# %%
from __future__ import annotations
import h5py    
import numpy as np    
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import sys

CAUSICA_FOLDER = '/home/sandor_daniel/work/2024-05-07_active_bayesian_grn/Project-BayesDAG/src/'
RESULT_DIR = '/home/sandor_daniel/work/2024-05-07_active_bayesian_grn/results/'
ROOT_DIR = '/home/sandor_daniel/work/2024-05-07_active_bayesian_grn/'
gflow_dir = '/home/sandor_daniel/work/2024-05-07_active_bayesian_grn/jax-dag-gflownet'
sys.path.append(ROOT_DIR)
sys.path.append(CAUSICA_FOLDER)
sys.path.append(gflow_dir)
from causica.models.bayesdag.bayesdag_nonlinear import BayesDAGNonLinear
from causica.datasets.variables import Variables, Variable
from causica.datasets.dataset import Dataset, CausalDataset

f = h5py.File('rnaseq_calico/ad_worm_aging.h5ad','r')   
# for k in  f.keys():
#     print(k, f[k].keys())

# %%
row_filter = f['obs']['annotate_name']['codes'][()] == 0
col_filter = f['var']['gene_class']['codes'][()] == 0

age_codes = f['obs']['timepoint']['codes'][()]
age_categories = f['obs']['timepoint']['categories'][()]
age_bytes = age_categories[age_codes[row_filter]]

age = np.array(list(map(lambda a: float(str(a, encoding='utf-8')[1:]), age_bytes)))
expression_counts = f['layers']['denoised'][()][row_filter][:,col_filter]
genes = f['var']['gene_names'][()][col_filter]

X = np.hstack([age[:,None], expression_counts])
cols = np.append(['age'], [str(g, encoding='utf-8') for g in genes], axis=0)
df_X = pd.DataFrame(X, columns=cols)
# df_X.head()

# %% [markdown]
# # BayesDAG

# %%
train_X = X[:int(X.shape[0]*0.8)]
val_X = X[int(X.shape[0]*0.8):int(X.shape[0]*0.9)]
test_X = X[int(X.shape[0]*0.9):]

train_mask = np.ones(train_X.shape)
val_mask = np.ones(val_X.shape)
test_mask = np.ones(test_X.shape)

graph_args = {}
graph_args['num_variables'] = X.shape[1]
graph_args['exp_edges'] = None
graph_args['exp_edges_per_node'] = None
graph_args['graph_type'] = None
graph_args['seed'] = 0


dataset = Dataset(train_X, train_mask, 
                        val_data=val_X, val_mask=val_mask, 
                        test_data=test_X, test_mask=test_mask,
                        graph_args=graph_args)

train_config_dict = {}
train_config_dict['batch_size'] = 16
train_config_dict['max_epochs'] = 1

train_rounds = 3
dag_log = []

name = 'worm_aging_DAG'
vars = Variables([Variable('age', True, 'continuous', lower=1., upper=15.)] + [Variable(str(g, encoding='utf-8'), True, 'continuous', lower=0, upper=10e3) for g in genes])
device = 'cuda:0'


# for i in range(train_rounds):
#     bd = BayesDAGNonLinear(name, vars, RESULT_DIR, device)
#     bd.run_train(dataset, train_config_dict)
#     Ws, _, _ = bd.get_weighted_adj_matrix(samples=4)
#     dag_log.append(Ws)

# %%
# G = nx.from_numpy_array(np.abs(dag_log[0][0].cpu().numpy()) > 0.4, create_using=nx.DiGraph())
# G = nx.relabel_nodes(G, {i:c for i, c in enumerate(cols)})

# cmap = ['green'] + ['lightblue' for i in range(1, len(cols))]

# for i, gene in enumerate([str(g, encoding='utf-8') for g in genes]):
#     if G.has_edge('age', gene) or G.has_edge(gene, 'age') or np.any([G.has_edge(p, gene) for p in G.predecessors('age')]):
#         cmap[i+1] = 'lightgreen'


# nx.draw(G, node_color=cmap, with_labels=True)

# %% [markdown]
# # GFN
from dag_gflownet.env import GFlowNetDAGEnv
from dag_gflownet.gflownet import DAGGFlowNet
from dag_gflownet.utils.replay_buffer import ReplayBuffer
from dag_gflownet.utils.factories import get_scorer
from dag_gflownet.utils.gflownet import posterior_estimate
from dag_gflownet.utils.metrics import expected_shd, expected_edges, threshold_metrics
from dag_gflownet.utils import io
import optax
import jax.numpy as jnp
import jax
from tqdm import trange
from argparse import Namespace



# %%
gflownet = DAGGFlowNet()
optimizer = optax.adam(0.01)

prefill = 1000
num_iterations = 10000
batch_size = 32

scorer_args = Namespace(prior='uniform', graph='calico', data=df_X, prior_kwargs={}, scorer_kwargs={})
scorer, data, graph = get_scorer(scorer_args)
key = jax.random.PRNGKey(123)
key, subkey = jax.random.split(key)


env = GFlowNetDAGEnv(
    num_envs=8,
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

# %%
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
posterior, _ = posterior_estimate(
    gflownet,
    params.online,
    env,
    key,
    num_samples=1000,
    desc='Sampling from posterior'
)

# %%
np.save("results/gflow/gflow_posterior", posterior)


