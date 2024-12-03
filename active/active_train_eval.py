from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import os
import sys

CAUSICA_FOLDER = '/home/sandor_daniel/work/2024-05-07_active_bayesian_grn/Project-BayesDAG/src/'
RESULT_DIR = '/home/sandor_daniel/work/2024-05-07_active_bayesian_grn/results/'
ROOT_DIR = '/home/sandor_daniel/work/2024-05-07_active_bayesian_grn/'
GFLOW_DIR = '/home/sandor_daniel/work/2024-05-07_active_bayesian_grn/jax-dag-gflownet'
sys.path.append(ROOT_DIR)
sys.path.append(CAUSICA_FOLDER)
sys.path.append(GFLOW_DIR)
from causica.models.bayesdag.bayesdag_nonlinear import BayesDAGNonLinear
from causica.datasets.variables import Variables, Variable
from causica.datasets.dataset import Dataset, CausalDataset

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

from DAG_estimator import GFlowDAGEstimator

from utils import create_pdag, adjacency_to_edge_list, directed_shd, undirected_shd, directed_edge_f1_score, pdag_f1_score, pdag_shd, conf_matrix



# %%
def load_standard(file):
    standard = pd.read_csv(file, sep='\t', header=None)
    standard.replace([f'G{i}' for i in range(10)], [f'G0{i}' for i in range(10)], inplace=True)
    standard = standard.pivot(columns=[0], index=[1], values=[2])
    np.fill_diagonal(standard.values, 0)
    standard = standard.to_numpy()
    return standard

def remove_cycles_from_true_graph(true_graph):
    G = nx.from_numpy_array(true_graph, create_using=nx.DiGraph())
    for c in nx.simple_cycles(G):
        true_graph[c[0], c[1]] = 0
    return true_graph

timeseries = np.loadtxt(f'../gnw_example/Example_dream4_timeseries.tsv', skiprows=1)[:,1:]
timeseries_split = np.split(timeseries, range(21,210,21), axis=0)

ground_truth =load_standard(f'../gnw_example/Example_goldstandard.tsv')
ground_truth = remove_cycles_from_true_graph(ground_truth)
known_subgraph_mask = np.ones(ground_truth.shape)

train_data = np.vstack(timeseries_split)
val_data = None
test_data = train_data

train_mask = np.ones(train_data.shape)
val_mask = None
test_mask = np.ones(test_data.shape)

graph_args = {}
graph_args['num_variables'] = timeseries.shape[1]
graph_args['exp_edges'] = None
graph_args['exp_edges_per_node'] = None
graph_args['graph_type'] = None
graph_args['seed'] = 0

dataset = CausalDataset(train_data, 
                        train_mask, 
                        ground_truth, 
                        known_subgraph_mask, 
                        None, 
                        None, 
                        val_data=val_data,  
                        val_mask=val_mask,
                        test_data=test_data,
                        test_mask=test_mask,
                        graph_args=graph_args)

vars = Variables([Variable(f'G{i}', True, 'continuous', lower=0, upper=1)
         for i in range(1,timeseries.shape[1]+1)])

train_config_dict = {}
train_config_dict['batch_size'] = 4
train_config_dict['max_epochs'] = 10
# bd.run_train(dataset, train_config_dict)

def load_knockouts(file):
    ko = pd.read_csv(file, sep='\t', header=0)
    ko = ko.to_numpy()[:,1:]
    ko_dict = {i+1 : ko[i*21:(i+1)*21] for i in range(ko.shape[1])}
    return ko_dict


ko_dict = load_knockouts('../gnw_example/Example_knockout_timeseries.tsv')

def eval_graphs(graphs, ground_truth, vars, name):
    d_shds = []
    u_shds = []
    d_f1s = []
    p_f1s = []
    p_shds = []
    nnzs = []
    conf_matrices = []

    for g in graphs:
        pred = adjacency_to_edge_list(g)
        
        true_graph = adjacency_to_edge_list(ground_truth)

        d_shds.append(directed_shd(true_graph, pred))
        u_shds.append(undirected_shd(true_graph, pred))
        d_f1s.append(directed_edge_f1_score(true_graph, pred))
        nnzs.append(len(pred))
        conf_matrices.append(conf_matrix(true_graph, pred, len(vars)))
        true_pdag = create_pdag(remove_cycles_from_true_graph(ground_truth))
        pred_pdag = create_pdag(g)
        p_f1s.append(pdag_f1_score(true_pdag, pred_pdag))
        p_shds.append(pdag_shd(true_pdag, pred_pdag))

    with open("result_metrics.txt", "a") as results:
        results.write(f"{name}\n")
        results.write(f'directed_shd: {np.mean(d_shds)}\n')
        results.write(f'undirected_shd: {np.mean(u_shds)}\n')
        results.write(f'directed_edge_f1_score: {np.mean(d_f1s)}\n')
        results.write(f'pdag_f1_score: {np.mean(p_f1s)}\n')
        results.write(f'pdag_shd: {np.mean(p_shds)}\n')
        results.write(f'nnz: {np.mean(nnzs)}\n')
        results.write(f'conf_matrix: {np.mean(conf_matrices, axis=0)}\n')
        results.write("\n")



# %%
from active_DAG_learning_framework import active_learning_procedure
from DAG_estimator import BayesDAGEstimator

# bd_estimator = BayesDAGEstimator('BayesDAGEstimator', vars, RESULT_DIR, 'cuda:0', graph_args)
# active_learning_procedure('uniform', val_data, test_data, ko_dict, train_data, bd_estimator, pretrain_epochs=16, T=64)
# bd_graphs_unif = bd_estimator.sample_models(n_samples=64)
# eval_graphs(bd_graphs_unif, ground_truth, vars, 'Bayes DAG uniform')

# bd_estimator = BayesDAGEstimator('BayesDAGEstimator', vars, RESULT_DIR, 'cuda:0', graph_args)
# active_learning_procedure('entropy', val_data, test_data, ko_dict, train_data, bd_estimator, pretrain_epochs=16, T=64)
# bd_graphs_ee = bd_estimator.sample_models(n_samples=64)
# eval_graphs(bd_graphs_ee, ground_truth, vars, 'Bayes DAG entropy')

# bd_estimator = BayesDAGEstimator('BayesDAGEstimator', vars, RESULT_DIR, 'cuda:0', graph_args)
# active_learning_procedure('eces', val_data, test_data, ko_dict, train_data, bd_estimator, pretrain_epochs=16, T=64)
# bd_graphs_eces = bd_estimator.sample_models(n_samples=64)
# eval_graphs(bd_graphs_eces, ground_truth, vars, 'Bayes DAG eces')

# bd_estimator = BayesDAGEstimator('BayesDAGEstimator', vars, RESULT_DIR, 'cuda:0', graph_args)
# active_learning_procedure('bald', val_data, test_data, ko_dict, train_data, bd_estimator, pretrain_epochs=16, T=64)
# bd_graphs_bald = bd_estimator.sample_models(n_samples=64)
# eval_graphs(bd_graphs_bald, ground_truth, vars, 'Bayes DAG bald')

# bd_estimator = BayesDAGEstimator('BayesDAGEstimator', vars, RESULT_DIR, 'cuda:0', graph_args)
# active_learning_procedure('ebald', val_data, test_data, ko_dict, train_data, bd_estimator, pretrain_epochs=16, T=64)
# bd_graphs_ebald = bd_estimator.sample_models(n_samples=64)
# eval_graphs(bd_graphs_ebald, ground_truth, vars, 'Bayes DAG ebald')

# # # %%


train_data = pd.DataFrame(data=timeseries)
val_data =  None
test_data =  train_data

# gflow_estimator = GFlowDAGEstimator('GFlowEstimator', vars, RESULT_DIR, 'cuda:0', graph_args)
# active_learning_procedure('uniform', val_data, test_data, ko_dict, train_data, gflow_estimator, pretrain_epochs=10000, T=64)
# gflow_graphs_unif = gflow_estimator.sample_models(n_samples=64)
# eval_graphs(gflow_graphs_unif, ground_truth, vars, 'GFlow uniform')

# gflow_estimator = GFlowDAGEstimator('GFlowEstimator', vars, RESULT_DIR, 'cuda:0', graph_args)
# active_learning_procedure('entropy', val_data, test_data, ko_dict, train_data, gflow_estimator, pretrain_epochs=10000, T=64)
# gflow_graphs_ee = gflow_estimator.sample_models(n_samples=64)
# eval_graphs(gflow_graphs_ee, ground_truth, vars, 'GFlow entropy')

# gflow_estimator = GFlowDAGEstimator('GFlowEstimator', vars, RESULT_DIR, 'cuda:0', graph_args)
# active_learning_procedure('eces', val_data, test_data, ko_dict, train_data, gflow_estimator, pretrain_epochs=10000, T=64)
# gflow_graphs_eces = gflow_estimator.sample_models(n_samples=64)
# eval_graphs(gflow_graphs_eces, ground_truth, vars, 'GFlow eces')

# gflow_estimator = GFlowDAGEstimator('GFlowEstimator', vars, RESULT_DIR, 'cuda:0', graph_args)
# active_learning_procedure('bald', val_data, test_data, ko_dict, train_data, gflow_estimator, pretrain_epochs=10000, T=64)
# gflow_graphs_bald = gflow_estimator.sample_models(n_samples=64)
# eval_graphs(gflow_graphs_bald, ground_truth, vars, 'GFlow bald')

# gflow_estimator = GFlowDAGEstimator('GFlowEstimator', vars, RESULT_DIR, 'cuda:0', graph_args)
# active_learning_procedure('ebald', val_data, test_data, ko_dict, train_data, gflow_estimator, pretrain_epochs=10000, T=64)
# gflow_graphs_ebald = gflow_estimator.sample_models(n_samples=64)
# eval_graphs(gflow_graphs_ebald, ground_truth, vars, 'GFlow ebald')


