# %%
import numpy as np
import sys

CAUSICA_FOLDER = '/home/sandor_daniel/work/2024-05-07_active_bayesian_grn/Project-BayesDAG/src/'
sys.path.append(CAUSICA_FOLDER)
from causica.datasets.variables import Variables, Variable

from train_functions import train_bayesdag, train_gflow, load_standard, eval_graphs

# %%
timeseries = np.loadtxt(f'../gnw_example/Example_dream4_timeseries.tsv', skiprows=1)[:,1:]
timeseries_split = np.split(timeseries, range(21,210,21), axis=0)

ground_truth =load_standard(f'../gnw_example/Example_goldstandard.tsv')
# ground_truth = remove_cycles_from_true_graph(ground_truth)
known_subgraph_mask = np.ones(ground_truth.shape)

graph_args = {}
graph_args['num_variables'] = timeseries.shape[1]
graph_args['exp_edges'] = None
graph_args['exp_edges_per_node'] = None
graph_args['graph_type'] = None
graph_args['seed'] = 0

n_folds = len(timeseries_split)
n_folds = 1

vars = Variables([Variable(f'G{i}', True, 'continuous', lower=0, upper=1)
         for i in range(1,timeseries.shape[1]+1)])

synthetic_ER_observations = []
synthetic_ER_adj_mats = []

for i in range(5,9):
    synthetic_ER_observations.append(np.loadtxt(f'../synthetic_data/run_ER_70_140_mlp_sem_unequal_noise_{i}_seed/all.csv', delimiter=','))
    synthetic_ER_adj_mats.append(np.loadtxt(f'../synthetic_data/run_ER_70_140_mlp_sem_unequal_noise_{i}_seed/adj_matrix.csv', delimiter=','))


# %% [markdown]
# #### Bayes DAG

# %%
# bayes_dag_graphs = train_bayesdag(timeseries, ground_truth, known_subgraph_mask, vars, graph_args )
# eval_graphs(bayes_dag_graphs, ground_truth, vars, "Bayes DAG")

# for synth_data, synth_tuth in zip(synthetic_ER_observations, synthetic_ER_adj_mats):
#     known_subgraph_mask = np.ones(synth_tuth.shape)
#     vars = Variables([Variable(f'N{i}', True, 'continuous', lower=0, upper=1)
#          for i in range(1,synth_data.shape[1]+1)])
#     graph_args['num_variables'] = synth_data.shape[1]
#     bayes_dag_graphs = train_bayesdag(synth_data, synth_tuth, known_subgraph_mask, vars, graph_args)
#     eval_graphs(bayes_dag_graphs, synth_tuth, vars)

# %% [markdown]
# ### GFN

# %%
gflow_graphs = train_gflow(timeseries, delta=0.5125) 
eval_graphs(gflow_graphs, ground_truth, vars, "GFlow")

# for synth_data, synth_tuth in zip(synthetic_ER_observations, synthetic_ER_adj_mats):
#     gflow_graphs = train_gflow(synth_data, delta=0.6)
#     eval_graphs(gflow_graphs, synth_tuth, vars)

# %%
# # if no memory: 
# !fuser -v -k /dev/nvidia0


