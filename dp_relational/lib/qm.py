from .dataset import RelationalDataset
import itertools
import numpy as np

import torch

from tqdm import tqdm

class QueryManager:
    """
    Query manager class.
    Given an input relational dataset, an input k for the desired k-way marginals
    and two pregenerated synthetic tables, returns an interface that allows for the
    calculation of "true" answers from the synthetic dataset.
    
    Also stores query vectors
    """
    def __init__(self, rel_dataset: RelationalDataset, k, df1_synth, df2_synth, otm=False, verbose=False) -> None:
        self.verbose = verbose
        
        self.rel_dataset = rel_dataset
        self.k = k
        self.df1_synth = df1_synth
        self.df2_synth = df2_synth
        
        self.n_syn1 = df1_synth.shape[0]
        self.n_syn2 = df2_synth.shape[0]
        self.n_syn_cross = self.n_syn1 * self.n_syn2
        
        self.otm = otm
        # print("t1", self.rel_dataset.table1.df.shape[0])
        # print("t2", self.rel_dataset.table2.df.shape[0])
        # print("rel", self.rel_dataset.df_rel.shape[0])
        # print(self.n_syn1)
        # print(self.n_syn2)
        orig_cross_size = self.rel_dataset.table1.df.shape[0] * self.rel_dataset.table2.df.shape[0]
        # print(np.sqrt(self.n_syn_cross / orig_cross_size))
        self.n_relationship_synth = self.n_syn1 if otm else int(self.rel_dataset.df_rel.shape[0] * np.sqrt(self.n_syn_cross / orig_cross_size))
        self.n_relationship_orig = self.rel_dataset.df_rel.shape[0]
        
        def make_cross_workloads():
            workload_names = []
            workload_dict = {}
            
            df1 = rel_dataset.table1.df
            df2 = rel_dataset.table2.df
            
            df1_col_dic = rel_dataset.table1.col_dict
            df2_col_dic = rel_dataset.table2.col_dict
            
            range_low = 0
            range_high = 0
            for k1 in range(1,k):
                df1_k1_col = list(itertools.combinations(df1.columns, k1))
                for k2 in range(1, k-k1+1):
                    df2_k2_col = list(itertools.combinations(df2.columns, k2))
                    if k1+k2 == k:
                        for col_comb_1 in df1_k1_col:
                            for col_comb_2 in df2_k2_col:
                                len_range = 1
                                uni_val_1 = []
                                
                                for col1 in col_comb_1:
                                    c1 = df1_col_dic[col1]['count']
                                    uni_val_1.append(c1)
                                    len_range *= c1
                                assert len(uni_val_1) == k1
                                
                                uni_val_2 = []
                                for col2 in col_comb_2:
                                    c2 = df2_col_dic[col2]['count']
                                    uni_val_2.append(c2)
                                    len_range *= c2
                                assert len(uni_val_2) == k2
                                
                                total_dim = len_range
                                    
                                t1_dimsizes = []
                                for subdim_size in uni_val_1:
                                    total_dim = total_dim/subdim_size
                                    t1_dimsizes.append(int(total_dim))
                                t2_dimsizes = []
                                for subdim_size in uni_val_2:
                                    total_dim = total_dim/subdim_size
                                    t2_dimsizes.append(int(total_dim))
                                
                                range_high = range_low + len_range - 1 
                                workload_names.append((col_comb_1 , col_comb_2))
                                workload_dict[(col_comb_1 , col_comb_2)] = {
                                    "dim_1": uni_val_1,
                                    "dim_2": uni_val_2,
                                    "range_low": range_low,
                                    "range_high": range_high,
                                    "range_size": len_range,
                                    "dim_sizes_1": t1_dimsizes,
                                    "dim_sizes_2": t2_dimsizes
                                }
                                
                                range_low = range_high + 1
            
            return (workload_names, workload_dict)
        
        self.workload_names, self.workload_dict = make_cross_workloads()
        
        last_workload = self.workload_names[-1]
        self.num_all_queries = self.workload_dict[last_workload]["range_high"] + 1
        
        def calculate_rand_ans():
            rand_ans = np.zeros(self.num_all_queries) # what answers would a random system give?
            for workload_name in self.workload_dict:
                w = self.workload_dict[workload_name]
                rand_ans[w['range_low']: (w['range_high']+1)] = 1.0/(w['range_high'] - w['range_low'] + 1)
            return rand_ans
        def calculate_true_ans():
            df_rel = rel_dataset.df_rel
            num_relationship = df_rel.shape[0]
            
            ID1s = df_rel.iloc[:][rel_dataset.rel_id1_col].values
            ID2s = df_rel.iloc[:][rel_dataset.rel_id2_col].values
            
            true_ans = np.zeros(self.num_all_queries) 
            for w in self.workload_names:
                w_dict = self.workload_dict[w]
                
                offsets_t1 = self.get_offsets(w, 0, is_synth=False)
                offsets_t2 = self.get_offsets(w, 1, is_synth=False)
                
                offsets = offsets_t1[ID1s] + offsets_t2[ID2s]
                values, counts = np.unique(offsets, return_counts=True)
                for val, count in zip(values, counts):
                    true_ans[w_dict["range_low"] + val] = count
            
            true_ans = true_ans/num_relationship
            
            return true_ans
        
        self.rand_ans = calculate_rand_ans()
        self.true_ans = self.calculate_true_ans()
        
    def calculate_ans_from_rel_dataset(self, df_rel, is_synth):
        num_relationship = df_rel.shape[0]
        
        ID1s = df_rel.iloc[:][self.rel_dataset.rel_id1_col].values
        ID2s = df_rel.iloc[:][self.rel_dataset.rel_id2_col].values
        
        ans = [] # np.zeros(self.num_all_queries) 
        for w in self.workload_names:
            # print(":", w)
            w_dict = self.workload_dict[w]
            
            offsets_t1 = self.get_offsets(w, 0, is_synth=is_synth)
            offsets_t2 = self.get_offsets(w, 1, is_synth=is_synth)
            
            offsets = offsets_t1[ID1s] + offsets_t2[ID2s]
            values, counts = np.unique(offsets, return_counts=True)
            ans.append(np.zeros((w_dict["range_size"], )))
            for val, count in zip(values, counts):
                ans[-1][val] = count / num_relationship
        return ans
    
    def calculate_true_ans(self):
        return self.calculate_ans_from_rel_dataset(self.rel_dataset.df_rel, is_synth=False)
    
    def query_ind(self, workload, val, zero_index=False):
        assert len(workload) == 2 #since only two tables
        assert len(workload[0]) + len(workload[1]) == self.k # k features?
        assert len(workload[0]) > 0 and len(workload[1]) > 0 # cross table queries
        
        assert len(val) == 2
        assert len(val[0]) == len(workload[0]) and len(val[1]) == len(workload[1])
        
        # TODO: add assert making sure val belongs to the unique value set of workload
        total_val = val[0] + val[1]
        
        q_ind = self.workload_dict[workload]['range_low']
        dim1 = self.workload_dict[workload]['dim_1']
        dim2 = self.workload_dict[workload]['dim_2']
        
        dim = dim1 + dim2
        
        total_dim = 1
        for i in dim:
            total_dim *= i
        
        # reshape
        for i in range(len(total_val)):
            total_dim = total_dim/dim[i]
            q_ind += total_dim * total_val[i]
            
        assert total_dim == 1
        assert q_ind <= self.workload_dict[workload]['range_high']
        
        if zero_index:
            q_ind -= self.workload_dict[workload]['range_low']
        
        return q_ind

    def get_offsets(self, workload, table_num, is_synth=True):
        assert table_num == 0 or table_num == 1
        
        dfs = [self.df1_synth, self.df2_synth] if is_synth else [self.rel_dataset.table1.df,self.rel_dataset.table2.df]
        df = dfs[table_num]

        # print(workload[table_num])
        # print(self.df1_synth.columns, )
        table = df[list(workload[table_num])]
        table = tuple(table[x].iloc[:].values for x in workload[table_num])
        
        dimsizes = self.workload_dict[workload][["dim_sizes_1", "dim_sizes_2"][table_num]]
        data_sizes = [self.n_syn1, self.n_syn2] if is_synth \
            else [self.rel_dataset.table1.df.shape[0],self.rel_dataset.table2.df.shape[0]]
        data_size = data_sizes[table_num]
        dim_counts = self.workload_dict[workload][["dim_1", "dim_2"][table_num]]

        offsets = np.zeros(shape=data_size, dtype=np.int_)
        for col in range(len(dimsizes)):
            # Synthesizers can emit category-coded columns as float dtype (e.g. 3.0).
            # Normalize to integer category indices and clamp to workload domain.
            values = np.asarray(table[col])
            if np.issubdtype(values.dtype, np.integer):
                values = values.astype(np.int_, copy=False)
            else:
                values = np.rint(values).astype(np.int_, copy=False)

            max_idx = max(int(dim_counts[col]) - 1, 0)
            values = np.clip(values, 0, max_idx)
            offsets += values * dimsizes[col]

        return offsets

class QueryManagerBasic(QueryManager):
    def __init__(self, rel_dataset: RelationalDataset, k, df1_synth, df2_synth, verbose=False) -> None:
        super().__init__(rel_dataset, k, df1_synth, df2_synth, verbose)
        
        if verbose:
            print("Constructing query matrix")
        # create the query matrix
        Q1 = np.zeros((self.num_all_queries, self.n_syn1 * self.n_syn2))

        for i in tqdm(range(self.n_syn1), disable=not verbose):
            row = self.df1_synth.iloc[i]
            #data indices
            ind_d = [i*self.n_syn2 + j for j in range(self.n_syn2)]
            
            for w in self.workload_names:
                val = list(row[list(w[0])])
                #query indices
                ind_q = self.query_ind_df1(w, val)
                
                Q1[np.ix_(ind_q, ind_d)] = 1
                
        Q2 = np.zeros((self.num_all_queries, self.n_syn1 * self.n_syn2))

        for j in tqdm(range(self.n_syn2)):
            row = self.df2_synth.iloc[j]
            #data indices
            ind_d = [i*self.n_syn2 + j for i in range(self.n_syn1)]
            
            for w in self.workload_names:
                val = list(row[list(w[1])])
                #query indices
                ind_q = self.query_ind_df2(w, val)
                
                Q2[np.ix_(ind_q, ind_d)] = 1
        
        Q = Q1 * Q2
        assert sum(sum(Q)) == (len(self.workload_names) * self.n_syn1 * self.n_syn2)
        self.Q = Q
    def query_ind_workload(self, workload):
        return list(range(self.workload_dict[workload]['range_low'],
                          self.workload_dict[workload]['range_high'] + 1))
    
    def query_ind_df1(self, workload, val1):    
        assert len(workload) == 2 #since only two tables
        assert len(workload[0]) + len(workload[1]) == self.k # k features?
        assert len(workload[0]) > 0 and len(workload[1]) > 0 # cross table queries
        
        assert len(val1) == len(workload[0])
        
        
        q_ind = self.workload_dict[workload]['range_low']
        dim1 = self.workload_dict[workload]['dim_1']
        dim2 = self.workload_dict[workload]['dim_2']
        
        dim = dim1 + dim2
        
        total_dim = 1
        for i in dim:
            total_dim *= i
        
        # reshape
        for i in range(len(val1)):
            total_dim = total_dim/dim[i]
            q_ind += total_dim * val1[i]
        
        q_ind_min = q_ind
        q_ind_max = q_ind
        
        for j in dim2:
            total_dim = total_dim/j
            q_ind_max += total_dim * (j-1)
        
        q = [i for i in range(int(q_ind_min), int(q_ind_max + 1))]
        
        assert max(q) <= self.workload_dict[workload]['range_high']
        
        return q
    
    def query_ind_df2(self, workload, val2):    
        assert len(workload) == 2 #since only two tables
        assert len(workload[0]) + len(workload[1]) == self.k # k features?
        assert len(workload[0]) > 0 and len(workload[1]) > 0 # cross table queries
        
        assert len(val2) == len(workload[1])
        
        
        q_ind = self.workload_dict[workload]['range_low']
        dim1 = self.workload_dict[workload]['dim_1']
        dim2 = self.workload_dict[workload]['dim_2']
        
        total_dim1 = 1
        for i in dim1:
            total_dim1 *= i
        
        q_ind_min = 0
        q_ind_max = 0
        
        for j in dim1:
            total_dim1 = total_dim1/j
            q_ind_max += total_dim1 * (j-1)
        q = [i for i in range(int(q_ind_min), int(q_ind_max + 1))]
        
        
        total_dim2 = 1
        
        for i in dim2:
            total_dim2 *= i
        
        q = [int(i * int(total_dim2)) for i in q]
        
        
        temp = q_ind
        for i in range(len(val2)):
            total_dim2 = total_dim2/dim2[i]
            temp += total_dim2 * val2[i]
        
        q = [int(i + temp) for i in q]
        
        assert max(q) <= self.workload_dict[workload]['range_high']
        return q

class QueryManagerTorch(QueryManager):
    """
    Query manager implementation in Pytorch, containing several optimizations.
        - Lazy generation of workload query vectors
        - Support for slicing to learn subsections of the query
        - Sparse Pytorch storage of query vectors (using COO)
    """
    def __init__(self, rel_dataset: RelationalDataset, k, df1_synth, df2_synth, device="cpu", otm=False, cache_query_matrices=False, verbose=False) -> None:
        super().__init__(rel_dataset, k, df1_synth, df2_synth, verbose=verbose, otm=otm)
        self.cache_query_matrices = cache_query_matrices
        self.device = device
        self.workload_query_answers = {}
        for workload in self.workload_names:
            self.workload_query_answers[workload] = None
        self.true_ans_tensor = torch.from_numpy(np.concatenate(self.true_ans)).float()
    def get_query_mat_full_table(self, workload):
        if self.workload_query_answers[workload] is not None:
            return self.workload_query_answers[workload]
        else:
            # generate the matrix
            range_low = self.workload_dict[workload]["range_low"]
            range_high = self.workload_dict[workload]["range_high"] + 1
            num_queries = range_high - range_low
            vec_len = self.n_syn1 * self.n_syn2
            true_vals = self.true_ans_tensor[range_low:range_high]
            
            indices_col = np.arange(0, vec_len)
            
            offsetst1 = self.get_offsets(workload, 0)
            offsetst2 = self.get_offsets(workload, 1)
            
            offsetst1fullvec = np.repeat(offsetst1, self.n_syn2)
            offsetst2fullvec = np.tile(offsetst2, self.n_syn1)
            
            indices_row = offsetst1fullvec + offsetst2fullvec
            indices = np.stack((indices_row, indices_col))
            
            # TODO: there is no good reason for this to work this way?? This should not be necessary
            query_mat = torch.sparse_coo_tensor(indices, np.ones(shape=(vec_len, )), size=(num_queries, vec_len), device=self.device).float().coalesce()
            # print(torch.sparse.sum(query_mat, dim=1))
            # print(vec_len)
            if self.cache_query_matrices:
                self.workload_query_answers[workload] = (query_mat, true_vals)
                return self.get_query_mat_full_table(workload)
            else:
                return (query_mat, true_vals)
    def get_query_mat_sub_table(self, workload, slices_t1, slices_t2):
        # generate the matrix
        size_t1 = slices_t1.shape[0]
        size_t2 = slices_t2.shape[0]
        
        range_low = self.workload_dict[workload]["range_low"]
        range_high = self.workload_dict[workload]["range_high"] + 1
        num_queries = range_high - range_low
        vec_len = size_t1 * size_t2
        true_vals = self.true_ans_tensor[range_low:range_high]
        
        indices_col = np.arange(0, vec_len)
        
        offsetst1 = self.get_offsets(workload, 0)[slices_t1]
        offsetst2 = self.get_offsets(workload, 1)[slices_t2]
        
        offsetst1fullvec = np.repeat(offsetst1, size_t2)
        offsetst2fullvec = np.tile(offsetst2, size_t1)
        
        indices_row = offsetst1fullvec + offsetst2fullvec
        indices = np.stack((indices_row, indices_col))
        
        # TODO: there is no good reason for this to work this way?? This should not be necessary
        query_mat = torch.sparse_coo_tensor(indices, np.ones(shape=(vec_len, )), size=(num_queries, vec_len), device=self.device).float().coalesce()
        return query_mat, true_vals
    def get_true_answers(self, workload):
        range_low = self.workload_dict[workload]["range_low"]
        range_high = self.workload_dict[workload]["range_high"] + 1
        
        true_vals = self.true_ans_tensor[range_low:range_high]
        
        return true_vals
    # def get_query_mat_partial(self, workload, table1_indices, table2_indices):
    #     # no point in caching these
    #     # our full_table query mat generation is so fast that we can just do this:
        
