import numpy as np
from utils import create_pdag, edge_list_to_adjacency, H, perform_ko

def uniform(vars, dag_samples, n_query: int = 1, T=1):
    query_idx = np.random.choice(range(len(vars)), size=n_query, replace=False)
    return query_idx

def equivalence_class_entropy_sampling(vars, dag_samples, n_query: int = 1, T=1):
    pdags = [create_pdag(d) for d in dag_samples]
    n_models = len(dag_samples)

    p_directed = np.array([edge_list_to_adjacency(p.directed_edge, vars) for p in pdags]).sum(axis=0) / len
    p_undirected = np.array([edge_list_to_adjacency(p.undirected_edge, vars) for p in pdags]).sum(axis=0) / len
    p_no_edge = np.array([~edge_list_to_adjacency(p, vars) for p in pdags]).sum().sum(axis=0) / len 

    direction_index = random.random_sample(size=p_undirected.shape) < p_undirected
    full_index = ~direction_index

    direction_dist = np.array([p_directed, p_directed.T])
    full_dist = np.array([p_undirected, p_directed, p_directed.T, p_no_edge])

    H_direction = H(direction_dist)
    H_full = H(full_dist)

    H = np.zeros_like(p_directed)
    H[direction_index] = H_direction[direction_index]
    H[full_index] = H_full[full_index]

    return np.argpartition(H, -n_query)[-n_query:]


def edge_entropy(vars, dag_samples, n_query: int = 1, T=1):
    n_models = len(dag_samples)

    p_directed = np.array([edge_list_to_adjacency(m.directed_edge, vars) for m in dag_samples]).sum(axis=0) / len
    p_no_edge = np.array([~edge_list_to_adjacency(p, vars) for p in pdags]).sum().sum(axis=0) / len

    distribution = np.array([p_directed, p_directed.T, p_no_edge])
    H = H(distribution)

    return np.argpartition(H, -n_query)[-n_query:]

def bald(vars, dag_samples, n_query: int = 1, T=1):
    n_models = len(dag_samples)

    ps_directed = np.array([edge_list_to_adjacency(d.directed_edge, vars) for d in dag_samples])
    ps_no_edge = np.array([~edge_list_to_adjacency(p, vars) for p in pdags])



    distribution = np.array([p_directed, p_directed.T, p_no_edge])
    H = H(distribution)
    E_H = E_H(distribution)

    bald = H - E_H

    return np.argpartition(H, -n_query)[-n_query:]

def equivalence_class_bald_sampling(learner, n_query: int = 1, T=1):
    pass