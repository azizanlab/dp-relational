from dp_relational.lib.runner import ModelRunner

import dp_relational.lib.synth_data
import dp_relational.data.movies

import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("cuda_available", torch.cuda.is_available())
print("using device: ", device)

# EPSILON_PYTORCH_EXPERIMENT_DATASET = uuid.UUID('8e8d44c1-fe1e-11ee-88ce-a059507978f3')

def qm_generator_torch(rel_dataset, k, df1_synth, df2_synth):
    return dp_relational.lib.synth_data.QueryManagerTorch(rel_dataset, k=k, df1_synth=df1_synth, df2_synth=df2_synth, device=device)

def cross_generator_torch(qm, eps_rel, T):
    b_round = dp_relational.lib.synth_data.learn_relationship_vector_torch_masked(qm, eps_rel, T=T, T_mirror=150, verbose=True, device=device)
    relationship_syn = dp_relational.lib.synth_data.make_synthetic_rel_table_sparse(qm, b_round)
    return relationship_syn

runner = ModelRunner()
runner.update(dataset_generator=dp_relational.data.movies.dataset, n_syn1=776, n_syn2=1208,
              synth='mst', epsilon=3.0, eps1=1.0, eps2=1.0, k=2, dmax=10,
              qm_generator=qm_generator_torch, cross_generation_strategy=cross_generator_torch,
              T=20)

epsilons = [2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
run_count = 0
while True:
    for epsilon in epsilons:
        runner.update(epsilon=epsilon)
        runner.regenerate_qm = True
        results = runner.run(extra_params={ "T_mirror": 150, "run_set": 2 })
        print(runner.relationship_syn.shape[0])
        run_count += 1
        print(f"epsilon: {epsilon}, error_ave: {results['error_ave']}")
        print(f"###### COMPLETED {run_count} RUNS ######")