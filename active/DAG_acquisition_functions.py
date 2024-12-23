import numpy as np
from utils import create_pdag, edge_list_to_adjacency, H, perform_ko
import gc

def uniform(vars, dag_samples, n_query: int = 1, T=1):
    query_idx = np.random.choice(range(len(vars)), size=n_query, replace=False)
    return query_idx

def choose_based_on_edge_matrix(cert_matrix, edge_probs):
    edge_certainty = 1 - cert_matrix
    node_uncertainty = np.sum(cert_matrix, axis=0) 
    node_preference = (edge_certainty * edge_probs * node_uncertainty).sum(axis=1)

    return np.argsort(node_preference)

# input is list of var names, list of adjacency materices
def edge_entropy(vars, dag_samples, n_query: int = 1, T=1):
    n_models = len(dag_samples)

    p_directed = np.array(dag_samples).sum(axis=0) / n_models
    p_no_edge = np.array([1-m-m.T for m in dag_samples]).sum(axis=0) / n_models

    distribution = np.array([p_directed, p_directed.T, p_no_edge])
    edge_entropies = H(distribution)

    return choose_based_on_edge_matrix(edge_entropies, p_directed)[:n_query]
    

#kérdsé, hogy valségeket, hogyan lehet kezelni majd pdagba alakításkor
def equivalence_class_entropy_sampling(vars, dag_samples, n_query: int = 1, T=1):
    pdags = [create_pdag(d) for d in dag_samples]
    n_models = len(dag_samples)
    
    directed_edges = [edge_list_to_adjacency(p.directed_edges, np.arange(len(vars))) * d for p, d in zip(pdags, dag_samples)]
    undirected_edges = [edge_list_to_adjacency(p.undirected_edges, np.arange(len(vars))) * d for p, d in zip(pdags, dag_samples)]


    p_directed = np.array(directed_edges).sum(axis=0) / n_models
    p_undirected = np.array(undirected_edges).sum(axis=0) / n_models
    p_no_edge = np.array([1-d-u-d.T-u.T for d, u in zip(directed_edges, undirected_edges)]).sum(axis=0) / n_models 

    direction_index = np.random.random_sample(size=p_undirected.shape) < p_undirected
    full_index = ~direction_index

    direction_dist = np.array([p_directed, p_directed.T])
    full_dist = np.array([p_undirected, p_directed, p_directed.T, p_no_edge])

    H_direction = H(direction_dist)
    H_full = H(full_dist)

    entropy = np.zeros_like(p_directed)
    entropy[direction_index] = H_direction[direction_index]
    entropy[full_index] = H_full[full_index]

    return choose_based_on_edge_matrix(entropy, p_directed)[:n_query]


# input is list of var names, list of adjacency materices only makes sense for probabilistic dag samples
def bald(vars, dag_samples, n_query: int = 1, T=1):
    n_models = len(dag_samples)

    p_directed = np.array(dag_samples).sum(axis=0) / n_models
    p_no_edge = np.array([1-m-m.T for m in dag_samples]).sum(axis=0) / n_models

    distribution = np.array([p_directed, p_directed.T, p_no_edge])
    entropy = H(distribution)
    E_H = np.sum([H(np.array([dd, dd.T, 1-dd-dd.T])) for dd in dag_samples], axis=0) / n_models

    bald = entropy - E_H

    gc.collect()

    return choose_based_on_edge_matrix(bald, p_directed)[:n_query]

def equivalence_class_bald_sampling(vars, dag_samples, n_query: int = 1, T=1):
    pdags = [create_pdag(d) for d in dag_samples]
    n_models = len(dag_samples)
    
    directed_edges = [edge_list_to_adjacency(p.directed_edges, np.arange(len(vars))) * d for p, d in zip(pdags, dag_samples)]
    undirected_edges = [edge_list_to_adjacency(p.undirected_edges, np.arange(len(vars))) * d for p, d in zip(pdags, dag_samples)]

    p_directed = np.array(directed_edges).sum(axis=0) / n_models
    p_undirected = np.array(undirected_edges).sum(axis=0) / n_models
    p_no_edge = np.array([1-d-u-d.T-u.T for d, u in zip(directed_edges, undirected_edges)]).sum(axis=0) / n_models 

    direction_index = np.random.random_sample(size=p_undirected.shape) < p_undirected
    full_index = ~direction_index

    direction_dist = np.array([p_directed, p_directed.T])
    full_dist = np.array([p_undirected, p_directed, p_directed.T, p_no_edge])

    H_direction = H(direction_dist)
    H_full = H(full_dist)

    E_H_direction = np.sum([H(np.array([dd, dd.T])) for dd in directed_edges], axis=0) / n_models
    E_H_full = np.sum([H(np.array([dd, dd.T, 1-dd-dd.T])) for dd in dag_samples], axis=0) / n_models

    bald = np.zeros_like(p_directed)
    bald[direction_index] = H_direction[direction_index] - E_H_direction[direction_index]
    bald[full_index] = H_full[full_index] - E_H_full[full_index]

    gc.collect()

    return choose_based_on_edge_matrix(bald, p_directed)[:n_query]