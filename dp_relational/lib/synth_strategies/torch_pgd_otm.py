import numpy as np
import torch
from ..qm import QueryManager, QueryManagerBasic, QueryManagerTorch
from ..helpers import cdp_delta, cdp_eps, cdp_rho, get_per_round_privacy_budget, torch_cat_sparse_coo


from ..helpers import expround_torch, GM_torch_noise, GM_torch, mosek_optimize, mirror_descent_torch
from ..helpers import unbiased_sample_torch, unbiased_sample, display_top
from ..helpers import get_relationships_from_sparse
from ..helpers import largest_singular_value, gradient, projsplx_torch, one_to_many_project, one_to_many_sample, pgd_optimize_one_to_many

from tqdm import tqdm

import random

import gc

import time

@torch.no_grad()
def learn_relationship_vector_torch_pgd_otm(qm: QueryManagerTorch, epsilon_relationship=1.0, T=100,
                                            delta_relationship = 1e-5, subtable_size=100000, queries_to_reuse=None, iter_cb=lambda *args: None,
                                            k_new_queries=3, k_choose_from=300, exp_mech_alpha=0.2, choose_worst=True, verbose=False, device="cpu",
                                              slices_per_iter=1, expansion_ratio=2.5):
    """Implementation of new PGD based algorithm
     - Exponential mechanism to choose queries from the set 
     - Unbiased estimator algorithm
     - Optimal gradient descent algorithm.
    
    Information on parameters:
    qm: a query manager to produce query matrices
    epsilon_relationship: the privacy budget allocated to the relational table
    delta_relationship: the delta privacy parameter
    T: the number of iterations to run
    subtable_size: the size of the subtable to generate. This is related to the alpha parameter by sqrt(subtable_size / (n_syn1 * n_syn2))
    queries_to_reuse: the number of queries that we will evaluate in each iteration. Set to None to run all.
    k_new_queries: number of new queries to add to our set in each iteration
    k_choose_from: number of queries to evaluate when running the exponential mechanism

    This algorithm enforces a one to many relationship. This is also NOT TESTED!
    """
    assert k_new_queries <= k_choose_from
    assert 0 < exp_mech_alpha < 1
    assert expansion_ratio >= 1
    
    n_relationship_synt = qm.n_syn1
    m = n_relationship_synt # alias
    m_privacy = qm.n_relationship_orig
    
    assert qm.n_relationship_synth == qm.n_syn1
    assert qm.otm == True
    
    assert n_relationship_synt < qm.n_syn1 * qm.n_syn2

    # fraction of each table that should be taken    
    table1_size = np.sqrt(subtable_size / expansion_ratio)
    # size of each table to take
    table1_slice_size = int(np.clip(table1_size, 1, qm.n_syn1))
    table2_slice_size = int(np.clip(table1_size * expansion_ratio, 1, qm.n_syn2))
    cross_slice_size = table1_slice_size * table2_slice_size
    
    # convert to RDP
    rho_rel = cdp_rho(epsilon_relationship, delta_relationship)
    
    # privacy parameter
    epsilon0 = np.sqrt((2 * rho_rel) / (k_new_queries * T)) if T != 0 else 100000
    
    # exponential mechanism factor: product before the softmax
    exp_mech_factor = np.sqrt(exp_mech_alpha) * epsilon0 * (m_privacy / qm.rel_dataset.dmax)
    
    # gaussian mechanism standard deviation
    gm_stddev = (np.sqrt(2) / (np.sqrt(1 - exp_mech_alpha) * epsilon0)) * (qm.rel_dataset.dmax / m_privacy)

    # intialization
    unselected_workload = list(range(len(qm.workload_names)))
    
    # we are now storing old queries!
    selected_workloads = []
    noisy_ans_list = []
    
    def get_dataset_answer(workload_idx, table1_idxes, table2_idxes):
        """ Given a workload index and indexes of columns in both table 1 and table 2,
         return the current answer on the dataset for the workload. """
        w = qm.workload_names[workload_idx]
        # load the workload
        # size: num_queries x (nsyn1*nsyn2)
        true_answer = qm.get_true_answers(w)
        
        offsets_t1 = qm.get_offsets(w, 0, is_synth=True)
        offsets_t2 = qm.get_offsets(w, 1, is_synth=True)
        
        offsets = offsets_t1[table1_idxes] + offsets_t2[table2_idxes]
        values, counts = np.unique(offsets, return_counts=True)
        
        dataset_answer = torch.zeros([qm.workload_dict[w]["range_size"]])
        for val, count in zip(values, counts):
            dataset_answer[val] = count
        dataset_answer /= table1_idxes.shape[0]
        return true_answer, dataset_answer
    
    # initialize a b_round vector
    rand_idxes = torch.randint(0, qm.n_syn2, (qm.n_syn1,)) + torch.arange(0, qm.n_syn2 * qm.n_syn1, qm.n_syn2)[None, :] # torch.randperm(qm.n_syn1 * qm.n_syn2)[None, :n_relationship_synt] # TODO: this may run out of memory
    b_round = torch.sparse_coo_tensor(indices=rand_idxes, values=torch.ones([n_relationship_synt]),
                                      size=[qm.n_syn_cross], device=device).float().coalesce()
    
    for t in tqdm(range(T)):
        # Multiple slices
        for x_sli in range(slices_per_iter):
            timers = []
            timers.append((time.time(), "start"))
            
            table1_idxes, table2_idxes = get_relationships_from_sparse(qm, b_round)
            
            def generate_rand_slice_offsets():
                # choose random guaranteed indices
                slice_table_order = torch.randperm(qm.n_syn1, device=device)[:table1_slice_size].cpu()
                slice_table1 = torch.from_numpy(table1_idxes[slice_table_order])
                # identify guarantees
                table2_guaranteed = torch.unique(torch.from_numpy(table2_idxes[slice_table_order]))
                num_t2_guaranteed = torch.numel(table2_guaranteed)
                # choose a set to slice
                t2_randperm = torch.randperm(qm.n_syn2)
                slice_table2_rand = torch.masked_select(t2_randperm, torch.isin(t2_randperm, table2_guaranteed, invert=True))[:(table2_slice_size - num_t2_guaranteed)]
                
                slice_table2 = torch.cat((table2_guaranteed, slice_table2_rand))
                # identify which cells these are in b
                offsets_table1 = slice_table1.repeat_interleave(table2_slice_size) * qm.n_syn2
                offsets_table2 = slice_table2.repeat(table1_slice_size)
                offsets = offsets_table1 + offsets_table2
                
                return slice_table1.cpu(), slice_table2.cpu(), offsets.to(device)
            
            slice_table1, slice_table2, offsets = generate_rand_slice_offsets()
            
            # we will start optimising from here
            sub_num_relationships = int(torch.sparse.sum(torch.index_select(b_round, 0, offsets)).numpy(force=True))
            print(sub_num_relationships)
            if (sub_num_relationships < 1):
                continue
            timers.append((time.time(), "assorted_precomps"))

            # On the first slice, we need to select new workloads.
            if x_sli == 0:
                def exp_mech_new_workloads(uselected_workload):
                    """ Uses the exponential mechanism to select new workloads """
                    
                    exp_mech_workload_pool = random.sample(uselected_workload, k=min(k_choose_from, len(uselected_workload)))
                    
                    # get answers on this dataset
                    # if queries are being reused, it makes logical sense to choose worst
                    # queries on the whole dataset, not just the current slice.
                    # we should not save the query matrices at this point or we will run out of memory
                    true_and_dset_answers = [get_dataset_answer(i, table1_idxes, table2_idxes) for i in exp_mech_workload_pool]
                    
                    errors = [torch.sum(torch.abs(true_answer - dataset_answer)).numpy(force=True) for true_answer, dataset_answer in true_and_dset_answers]
                    
                    new_workloads = []
                    for x in range(k_new_queries):
                        # convert into numpy array
                        errors_np = np.array(errors)
                        
                        # now select from this set using the exponential mechanism
                        def softmax(v):
                            v_exp = np.exp(v - np.max(v))
                            return v_exp / np.sum(v_exp)
                        distribution = softmax(exp_mech_factor * errors_np)
                        
                        # sample from the distribution
                        def sample(dist):
                            cumulative_dist = np.cumsum(dist)
                            r = np.random.rand()
                            return np.searchsorted(cumulative_dist, r)
                        new_workload_idx = sample(distribution)
                        new_workload = exp_mech_workload_pool[new_workload_idx]
                        new_workloads.append(new_workload)
                        
                        # remove the workload from the pool
                        exp_mech_workload_pool.pop(new_workload_idx)
                        errors.pop(new_workload_idx) # double check
                    
                    return new_workloads
                
                new_workloads_this_iter = exp_mech_new_workloads(unselected_workload)
                timers.append((time.time(), "exponential mechanism"))
                
                for i in new_workloads_this_iter:
                    unselected_workload.remove(i)
                    selected_workloads.append(i)
                    
                    workload = qm.workload_names[i]
                    noisy_ans_list.append(GM_torch_noise(qm.get_true_answers(workload), gm_stddev))
            
            # initialize the Q_set from this list
            Q_set = torch.empty((0, cross_slice_size)).to_sparse_coo().to(device=device).float().coalesce()
    
            k_val = len(selected_workloads) if queries_to_reuse is None else min(queries_to_reuse, len(selected_workloads))
            errors = []
            
            timers.append((time.time(), "begin workload eval"))
            # On each iteration, evaluate all the workloads that we have stored answers for, and keep the ones with the worst errors.
            for i in range(len(selected_workloads)):
                workload_idx = selected_workloads[i]
                _, dataset_ans = get_dataset_answer(workload_idx, table1_idxes, table2_idxes) # we can't actually use the true answer here!
                true_ans = noisy_ans_list[i]
                errors.append((torch.sum(torch.abs(true_ans - dataset_ans)).numpy(force=True), i))
            top_errors = (sorted(errors) if choose_worst else random.sample(errors, len(errors)))[-k_val:]
            curr_workload_idxes = [i for err, i in top_errors]
            iter_selected_workloads = [selected_workloads[i] for i in curr_workload_idxes]
            iter_noisy_ans = torch.cat([noisy_ans_list[i] for i in curr_workload_idxes])
            timers.append((time.time(), "end workload eval"))
            
            # Get the query matrices for these workloads
            for i in iter_selected_workloads:
                curr_workload = qm.workload_names[i]
    
                curr_Qmat, curr_true_answer = qm.get_query_mat_sub_table(curr_workload, slice_table1, slice_table2)
                # curr_Qmat = torch.index_select(curr_Qmat_full, 1, offsets).coalesce()
                
                # del curr_Qmat_full
    
                Q_set = torch_cat_sparse_coo([Q_set, curr_Qmat], device=device)
                
                del curr_Qmat
            
            timers.append((time.time(), "build q mat"))
            # start with a random guess for b
            # TODO: think about using Algorithm L: https://en.wikipedia.org/wiki/Reservoir_sampling for this instead
            b_slice_rand_idxes = torch.randint(0, table2_slice_size, (table1_slice_size,)) + torch.arange(0, table2_slice_size * table1_slice_size, table2_slice_size)
            b_slice_rand_idxes = torch.stack((b_slice_rand_idxes, torch.zeros_like(b_slice_rand_idxes)))
            
            b_slice = torch.sparse_coo_tensor(indices=b_slice_rand_idxes, values=torch.ones([sub_num_relationships]),
                                            size=[cross_slice_size, 1], device=device).float().coalesce()
            Q_set = Q_set.coalesce()
            
            # run pgd.
            # The one to many function also enforces the one-to-many constraint in the projection.
            b_slice = pgd_optimize_one_to_many(Q_set, b_slice, iter_noisy_ans.to(device=device), sub_num_relationships, 20, table2_slice_size)
            
            timers.append((time.time(), "optimizer"))
            
            # put these back into the slice: this is slightly complicated!
            b_slice_round = one_to_many_sample(torch.squeeze(b_slice), table2_slice_size).to(device)
            timers.append((time.time(), "sample"))
            b_slice_round = b_slice_round.to_sparse()
            
            # create a mask
            mask = torch.sparse_coo_tensor(offsets[None, :], torch.ones_like(offsets), size=[qm.n_syn_cross], device=device).coalesce()
            b_round.coalesce()
            b_round = b_round - (mask * b_round) # now the area is filled with zeros
            # get nonzero indices in b_slice_round
            nz_indices = torch.squeeze(b_slice_round.indices())
            # lookup what offsets these were in the original tensor
            new_offsets = offsets[nz_indices]
            # print(new_offsets)
            # create new values
            new_values = torch.sparse_coo_tensor(new_offsets[None, :], torch.ones_like(new_offsets), size=[qm.n_syn_cross], device=device).coalesce()
            b_round = b_round + new_values
            timers.append((time.time(), "reinsert"))
            
            # clean TODO: is this necessary?
            del slice_table1
            del slice_table2
            del offsets
            del mask
            del Q_set
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # print(timers)
            timers_processed = [(int((timtup[0] - timers[i][0]) * 100000) / 100000, timtup[1]) for i, timtup in enumerate(timers[1:])]
            # print(f"slice {x_sli}")
            # print(timers_processed)
        
        iter_cb(qm, b_round, t)
    
    return b_round
