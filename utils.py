import numpy as np
import pgmpy
import networkx as nx
import os

# from ebunch or pgmpy dag
def create_pdag(dag):
  dag = pgmpy.base.DAG(dag)

  # skeleton
  skeleton = list(dag.edges())
  skeleton.extend([(b, a) for (a,b) in skeleton])
  directed = []

  # immoralities
  immoralities = dag.get_immoralities()

  for (a, b) in immoralities:
    a_children = set(dag.get_children(a))
    b_children = set(dag.get_children(b))
    common_children = list(a_children.intersection(b_children))
    a_edges = [(a, c) for c in common_children]
    b_edges = [(b, c) for c in common_children]
    directed.extend(a_edges)
    directed.extend(b_edges)

  # meek rules
  next_dag = pgmpy.base.DAG()
  next_dag.add_nodes_from(dag.nodes())
  next_dag.add_edges_from(directed)
  previous_dag = None

  while next_dag != previous_dag:
    previous_dag = next_dag
    # 1
    for (a, b) in skeleton:
      if (a, b) in next_dag.edges() or (b, a) in next_dag.edges():
        continue
      if np.all([(c, b) not in skeleton for c in next_dag.get_parents(a)]):
        next_dag.add_edge(a,b)
    # 2
    for (a, b) in skeleton:
      if (a, b) in next_dag.edges() or (b, a) in next_dag.edges():
        continue
      if nx.has_path(next_dag, a, b):
        next_dag.add_edge(a,b)
    # 3
    for (a, b) in skeleton:
      if (a, b) in next_dag.edges() or (b, a) in next_dag.edges():
        continue
      for (i1, i2) in next_dag.get_immoralities():
        if i1 in next_dag.get_parents(b) and i2 in next_dag.get_parents(b) and (a, i1) in skeleton and (a, i2) in skeleton:
          next_dag.add_edge(a,b)
    #4
    for (a, b) in skeleton:
      if (a, b) in next_dag.edges() or (b, a) in next_dag.edges():
        continue
      for pa_b in next_dag.get_parents(b):
        if (pa_b,a) in skeleton:
          for pa_pa_b in next_dag.get_parents(pa_b):
            if (pa_pa_b, a) in skeleton:
              next_dag.add_edge(a,b)
              break

  directed.extend(list(next_dag.edges()))

  undirected = []
  for (a,b) in skeleton:
    if (a,b) not in undirected and (b,a) not in undirected:
      if (a,b) not in directed and (b,a) not in directed:
        undirected.append((a,b))


  pdag = pgmpy.base.PDAG(directed_ebunch=directed,
                         undirected_ebunch=undirected)

  return pdag

def edge_list_to_adjacency(edge_list, node_order):
  G = nx.DiGraph()
  G.add_nodes_from(node_order)
  G.add_edges_from(edge_list)
  return nx.adjacency_matrix(G, node_order).todense()

def adjacency_to_edge_list(adjacency):
  G = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
  return list(G.edges())

def H(P):
  logP = np.where(P != 0, np.log2(P), 0)
  return -np.sum(np.multiply(P, logP), axis=0)

def E_H(Ps):
  return np.mean(H(Ps), axis=0)

def sample_x_pool(X_pool, idx, n):
  selected = np.random.choice(range(len(X_pool[idx])), n)
  return X_pool[idx][selected]

# for knockout experiments from DREAM
def perform_ko(ko_dict, gene):
  knockout_data = ko_dict[gene+1]
  new_data = np.copy(ko_dict)
  del new_data[gene+1]
  return ko_dict[gene+1], new_data


def directed_edge_f1_score(true_DAG, pred_DAG):
  tp = np.sum([1 for (a,b) in true_DAG if (a,b) in pred_DAG])
  fp = np.sum([1 for (a,b) in pred_DAG if (a,b) not in true_DAG])
  fn = np.sum([1 for (a,b) in true_DAG if (a,b) not in pred_DAG])

  return tp / (tp + 0.5 * (fp + fn))

def undirected_edge_f1_score(true_DAG, pred_DAG):
  tp = np.sum([1 for (a,b) in true_DAG if (a,b) in pred_DAG or (b,a) in pred_DAG])
  fp = np.sum([1 for (a,b) in pred_DAG if (a,b) not in true_DAG and (b,a) not in true_DAG])
  fn = np.sum([1 for (a,b) in true_DAG if (a,b) not in pred_DAG and (b,a) not in pred_DAG])

  return tp / (tp + 0.5 * (fp + fn))

def directed_shd(truth_edges, pred_edges):
  shd = 0
  for (a,b) in truth_edges:
    if (a,b) not in pred_edges:
      shd += 1
  for (a,b) in pred_edges:
    if (a,b) not in truth_edges and (b,a) not in truth_edges:
      shd += 1
  return shd

def undirected_shd(truth_edges, pred_edges):
  shd = 0
  for (a,b) in truth_edges:
    if (a,b) not in pred_edges and (b,a) not in pred_edges :
      shd += 1
  for (a,b) in pred_edges:
    if (a,b) not in truth_edges and (b,a) not in truth_edges:
      shd += 1
  return shd


def pdag_f1_score(pdag_true, pdag_pred):
    true_directed = pdag_true.directed_edges
    pred_directed = pdag_pred.directed_edges
    true_undirected = pdag_true.undirected_edges
    pred_undirected = pdag_pred.undirected_edges

    tp_d = np.sum([1 for (a,b) in true_directed if (a,b) in pred_directed or (b,a) in pred_directed])
    fp_d = np.sum([1 for (a,b) in pred_directed if (a,b) not in true_directed and (b,a) not in true_directed])
    fn_d = np.sum([1 for (a,b) in true_directed if (a,b) not in pred_directed and (b,a) not in pred_directed])

    tp_u = np.sum([1 for (a,b) in true_undirected if (a,b) in pred_undirected])
    fp_u = np.sum([1 for (a,b) in pred_undirected if (a,b) not in true_undirected])
    fn_u = np.sum([1 for (a,b) in true_undirected if (a,b) not in pred_undirected])

    tp = tp_u + tp_d
    fp = fp_u + fp_d
    fn = fn_u + fn_d

    return tp / (tp + 0.5 * (fp + fn))

def pdag_shd(pdag_true, pdag_pred):
    pred_directed = pdag_pred.directed_edges
    truth_directed = pdag_true.directed_edges
    pred_undirected = pdag_pred.undirected_edges
    truth_undirected = pdag_true.undirected_edges
    d = directed_shd(pred_directed, truth_directed)
    u = undirected_shd(pred_undirected, truth_undirected)
    return d + u


def conf_matrix(truth_edges, pred_edges, n_nodes):
  tp = np.sum([1 for (a,b) in truth_edges if (a,b) in pred_edges or (b,a) in pred_edges])
  fp = np.sum([1 for (a,b) in pred_edges if (a,b) not in truth_edges and (b,a) not in truth_edges])
  fn = np.sum([1 for (a,b) in truth_edges if (a,b) not in pred_edges and (b,a) not in pred_edges])
  tn = n_nodes * n_nodes - tp - fp - fn

  return [[tp, fn], [fp, tn]]

def save_dag_log(log, path, model_name, epoch):
  for i, dag in enumerate(log):
    if not os.path.exists(f'{path}/{model_name}/epoch_{epoch}'):
      os.makedirs(f'{path}/{model_name}/epoch_{epoch}')
    np.save(f'{path}/{model_name}/epoch_{epoch}/dag_{i}.npy', dag)