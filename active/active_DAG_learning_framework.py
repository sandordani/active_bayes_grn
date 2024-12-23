import torch
import numpy as np
import gc
from active_DAG_learner import ActiveDAGLearner

from DAG_acquisition_functions import uniform, edge_entropy, equivalence_class_entropy_sampling, bald, equivalence_class_bald_sampling
from utils import perform_ko, save_dag_log

acq_func_dict = {
        'uniform': uniform,
        'entropy': edge_entropy,
        'eces': equivalence_class_entropy_sampling,
        'bald': bald,
        'ebald': equivalence_class_bald_sampling
    }

def active_learning_procedure(
    query_strategy,
    X_val: np.ndarray,
    X_test: np.ndarray,
    X_pool: dict,
    X_init: np.ndarray,
    estimator,
    T: int = 16,
    n_query: int = 1,
    n_samples_per_query: int = 5,
    pretrain_epochs: int = 16,
):
    """Active Learning Procedure

    Attributes:
        query_strategy: Choose between Uniform(baseline), max_entropy, bald,
        X_val, y_val: Validation dataset,
        X_test, y_test: Test dataset,
        X_pool, y_pool: Query pool set,
        X_init, y_init: Initial training set data points,
        estimator: Neural Network architecture, e.g. CNN,
        T: Number of MC dropout iterations (repeat acqusition process T times),
        n_query: Number of points to query from X_pool,
        training: If False, run test without MC Dropout (default: True)
    """
    print(f'-------------------------------{query_strategy}---------------------------------')
    n_vars = X_init.shape[1]

    learner = ActiveDAGLearner(
        estimator=estimator,
        X_training=X_init,
        query_strategy=acq_func_dict[query_strategy],
        pretrain_epochs=pretrain_epochs,
    )
    
    save_dag_log(estimator.sample_models(n_samples=64), '../dag_logs', f'{estimator.name} - {query_strategy}', 0)

    for index in range(T):
        query_idx = learner.query(n_query=n_vars, T=T)
        #find the first n_query element from the back of query_idx that is in the pool in
        #reverse order
        query_idx = [i for i in query_idx[::-1] if i+1 in X_pool]
        if len(query_idx) == 0:
            break
        elif len(query_idx) > n_query:
            query_idx = query_idx[:n_query]
            with open(f'../dag_logs/{estimator.name} - {query_strategy}/choice_order', "a") as myfile:
                myfile.write(f"{index}: {query_idx}\n")

        new_X = []
        for i in query_idx:
            new_X_i, X_pool = perform_ko(X_pool, i)
            new_X.append(new_X_i)
        new_X = np.vstack(new_X)
        print('new_X', new_X.shape)

        learner.teach(new_X)
        save_dag_log(estimator.sample_models(n_samples=64), '../dag_logs', f'{estimator.name} - {query_strategy}', index+1)
        torch.cuda.empty_cache()
        gc.collect()
