from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import os
import sys
import optax
import jax.numpy as jnp
import jax
from tqdm import trange
from argparse import Namespace

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

from dag_gflownet.env import GFlowNetDAGEnv
from dag_gflownet.gflownet import DAGGFlowNet
from dag_gflownet.utils.replay_buffer import ReplayBuffer
from dag_gflownet.utils.factories import get_scorer
from dag_gflownet.utils.gflownet import posterior_estimate
from dag_gflownet.utils.metrics import expected_shd, expected_edges, threshold_metrics
from dag_gflownet.utils import io

from utils import create_pdag, adjacency_to_edge_list, directed_shd, undirected_shd, directed_edge_f1_score, pdag_f1_score, pdag_shd, conf_matrix

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

def train_bayesdag(X, ground_truth, known_subgraph_mask, vars, graph_args, batch_size=16, max_epochs=64, num_chains=5, lambda_sparse=50, sparse_init=True):
    train_data = X
    val_data = None
    test_data = train_data

    train_mask = np.ones(train_data.shape)
    val_mask = None
    test_mask = np.ones(test_data.shape)

    dataset = CausalDataset(train_data, 
                        train_mask, 
                        remove_cycles_from_true_graph(ground_truth), 
                        known_subgraph_mask, 
                        None, 
                        None, 
                        val_data=val_data,  
                        val_mask=val_mask,
                        test_data=test_data,
                        test_mask=test_mask,
                        graph_args=graph_args)

    bd = BayesDAGNonLinear('Model1', vars, RESULT_DIR, 'cuda:0', num_chains=num_chains, lambda_sparse=lambda_sparse, sparse_init=sparse_init) # right number of nnz edges this way

    train_config_dict = {}
    train_config_dict['batch_size'] = batch_size
    train_config_dict['max_epochs'] = max_epochs
    bd.run_train(dataset, train_config_dict)
    dag_samples, is_dag = bd.get_adj_matrix(samples=2048)
    bayes_dag_graphs =[dag_samples[i] for i in range(dag_samples.shape[0]) if is_dag[i]]
    return bayes_dag_graphs

def train_gflow(X, delta, epochs=10000, batch_size=16):
    train_data = pd.DataFrame(X)

    gflownet = DAGGFlowNet(delta=delta) #0.5-0.8 érdemes
    optimizer = optax.adam(0.01)

    prefill = 1000
    num_iterations = 10000
    batch_size = 16

    scorer_args = Namespace(prior='uniform', graph='dream', data=train_data, prior_kwargs={}, scorer_kwargs={})
    scorer, data, graph = get_scorer(scorer_args)
    key = jax.random.PRNGKey(123)
    key, subkey = jax.random.split(key)


    env = GFlowNetDAGEnv(
        num_envs=1,
        scorer=scorer
    )

    replay = ReplayBuffer(
        10000,
        num_variables=env.num_variables
    )


    exploration_schedule = jax.jit(optax.linear_schedule(
        init_value=jnp.array(0.),
        end_value=jnp.array(1. - 0.1),
        transition_steps=num_iterations // 2,
        transition_begin=prefill,
    ))

    params, state = gflownet.init(
            subkey,
            optimizer,
            replay.dummy['adjacency'],
            replay.dummy['mask']
        )

    # Training loop
    indices = None
    observations = env.reset()

    with trange(prefill + num_iterations, desc='Training') as pbar:
        for iteration in pbar:
            # Sample actions, execute them, and save transitions in the replay buffer
            epsilon = exploration_schedule(iteration)
            actions, key, logs = gflownet.act(params.online, key, observations, epsilon)
            next_observations, delta_scores, dones, _ = env.step(np.asarray(actions))
            indices = replay.add(
                observations,
                actions,
                logs['is_exploration'],
                next_observations,
                delta_scores,
                dones,
                prev_indices=indices
            )
            observations = next_observations

            if iteration >= prefill:
                # Update the parameters of the GFlowNet
                samples = replay.sample(batch_size=batch_size)
                params, state, logs = gflownet.step(params, state, samples)

                pbar.set_postfix(loss=f"{logs['loss']:.2f}", epsilon=f"{epsilon:.2f}")

    # Evaluate the posterior estimate
    gfn_dags, _ = posterior_estimate(
        gflownet,
        params.online,
        env,
        key,
        num_samples=64,
        desc='Sampling from posterior'
    )

    return gfn_dags

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

        # print(len(true_graph))
        # print(len(pred))
        # print(conf_matrix(true_graph, pred, len(vars)))

        d_shds.append(directed_shd(true_graph, pred))
        u_shds.append(undirected_shd(true_graph, pred))
        d_f1s.append(directed_edge_f1_score(true_graph, pred))
        nnzs.append(len(pred))
        true_pdag = create_pdag(remove_cycles_from_true_graph(ground_truth))
        pred_pdag = create_pdag(g)
        p_f1s.append(pdag_f1_score(true_pdag, pred_pdag))
        p_shds.append(pdag_shd(true_pdag, pred_pdag))
        conf_matrices.append(conf_matrix(true_graph, pred, len(vars)))

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
