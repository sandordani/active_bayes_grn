import torch
import numpy as np
from active_DAG_learner import ActiveDAGLearner

from DAG_acquisition_functions import uniform, equivalence_class_entropy_sampling
from utils import perform_ko

acq_func_dict = {
        'uniform': uniform,
        'eces': equivalence_class_entropy_sampling
    }

def active_learning_procedure(
    query_strategy,
    X_val: np.ndarray,
    X_test: np.ndarray,
    X_pool: dict,
    X_init: np.ndarray,
    estimator,
    T: int = 10,
    n_query: int = 1,
    n_samples_per_query: int = 5,
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


    learner = ActiveDAGLearner(
        estimator=estimator,
        X_training=X_init,
        query_strategy=acq_func_dict[query_strategy],
    )
    perf_hist = [learner.score(X_test)]

    for index in range(T):
        query_idx = learner.query(n_query=n_query, T=T)
        new_X = np.vstack([perform_ko(X_pool, i) for i in query_idx])
        learner.teach(new_X)

        model_accuracy_val = learner.score(X_val)
    #     if (index + 1) % 5 == 0:
    #         print(f"Val Accuracy after query {index+1}: {model_accuracy_val:0.4f}")
    #     perf_hist.append(model_accuracy_val)
    # model_accuracy_test = learner.score(X_test)
    # print(f"********** Test Accuracy per experiment: {model_accuracy_test} **********")
    return perf_hist, model_accuracy_test