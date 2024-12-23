import numpy as np
from sklearn.base import BaseEstimator
import torch
import sys
import copy
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



class DAGEstimator(BaseEstimator):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def fit(self, X):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

    def score(self, X):
        pass


class BayesDAGEstimator(DAGEstimator):
    def __init__(self, name, vars, results_dir, device, graph_args ):
        super().__init__(name)
        self.graph_args = graph_args
        self.train_config_dict = {}
        self.bd = BayesDAGNonLinear(name, vars, results_dir, device, num_chains=5, lambda_sparse=50, sparse_init=True) 
        print(vars)
        self.vars = copy.deepcopy(vars)
        self.graph_args = {}
        self.graph_args['exp_edges'] = None
        self.graph_args['exp_edges_per_node'] = None
        self.graph_args['graph_type'] = None
        self.graph_args['seed'] = 0
        self.graph_args['num_variables'] = len(vars)

    def fit(self, X, batch_size=16, max_epochs=16):
        train_mask = np.ones(X.shape)
        dataset = Dataset(X, train_mask, graph_args=self.graph_args)
        self.train_config_dict['batch_size'] = batch_size
        self.train_config_dict['max_epochs'] = max_epochs
        self.bd.run_train(dataset, self.train_config_dict)
        torch.cuda.empty_cache()

    def score(self, X, n_samples=4):
        X_torch = torch.tensor(X, dtype=torch.float32, device=self.bd.device)

        A_samples = self.bd.transform_adj(self.bd.p, detach_W=False)

        return self.bd.data_likelihood(X_torch, A_samples, X.shape[0])

    def sample_models(self, n_samples=4):
        Ws, _, _ = self.bd.get_weighted_adj_matrix(samples=n_samples)
        Ws = Ws.cpu().numpy() # Convert to numpy
        Ws = [np.absolute(W)/np.absolute(W).sum() for W in Ws] # Normalize the weights
        return Ws
    
    def get_vars(self):
        return self.vars


class GFlowDAGEstimator(DAGEstimator):
    def __init__(self, name, vars, results_dir, device, graph_args):
        super().__init__(name)
        self.vars = copy.deepcopy(vars)
        self.gflownet = DAGGFlowNet(delta=1.35)
        self.optimizer = optax.adam(0.01)

        self.prefill = 1000
        self.num_iterations = 10000
        self.batch_size = 16 #4
        self.replay_capacity = 10000
        self.min_exploration = 0.1
        self.num_envs = 1

        
        key = jax.random.PRNGKey(123)
        self.key, self.subkey = jax.random.split(key)
        
        self.exploration_schedule = jax.jit(optax.linear_schedule(
            init_value=jnp.array(0.),
            end_value=jnp.array(1. - self.min_exploration),
            transition_steps=self.num_iterations // 2,
            transition_begin=self.prefill,
        ))
        
        self.pretrain = True

    def init_env(self, X):
        scorer_args = Namespace(prior='uniform', graph='dream', data=X, prior_kwargs={}, scorer_kwargs={})
        self.scorer, self.data, self.graph = get_scorer(scorer_args)

        self.env = GFlowNetDAGEnv(
            num_envs=self.num_envs,
            scorer=self.scorer
        )

        self.replay = ReplayBuffer(
            self.replay_capacity,
            num_variables=self.env.num_variables
        )

        self.params, self.state = self.gflownet.init(
                self.subkey,
                self.optimizer,
                self.replay.dummy['adjacency'],
                self.replay.dummy['mask']
        )
        
        self.observations = self.env.reset()

    def fit(self, X, batch_size=4, max_epochs=500):

        if(self.pretrain):
            indices = None
            self.pretrain = False
            self.init_env(X)                                  
        else:
            self.prefill = 0
            indices = self.prev_indices

        with trange(self.prefill + max_epochs, desc='Training') as pbar:
            for iteration in pbar:
                # Sample actions, execute them, and save transitions in the replay buffer
                epsilon = self.exploration_schedule(iteration)
                actions, self.key, logs = self.gflownet.act(self.params.online, self.key, self.observations, epsilon)
                next_observations, delta_scores, dones, _ = self.env.step(np.asarray(actions))
                indices = self.replay.add(
                    self.observations,
                    actions,
                    logs['is_exploration'],
                    next_observations,
                    delta_scores,
                    dones,
                    prev_indices=indices
                )
                self.prev_indices = indices
                self.observations = next_observations

                if iteration >= self.prefill:
                    # Update the parameters of the GFlowNet
                    samples = self.replay.sample(batch_size=self.batch_size)
                    self.params, self.state, logs = self.gflownet.step(self.params, self.state, samples)

                    pbar.set_postfix(loss=f"{logs['loss']:.2f}", epsilon=f"{epsilon:.2f}")

        

    def score(self, X, n_samples=4):
        return -1

    def sample_models(self, n_samples=4):
        # Evaluate the posterior estimate
        Ws, _ = posterior_estimate(
            self.gflownet,
            self.params.online,
            self.env,
            self.key,
            num_samples=n_samples,
            desc='Sampling from posterior'
        )
        return Ws
    
    def get_vars(self):
        return self.vars