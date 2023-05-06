import networkx as nx
import numpy as np

from graph import function_basis, convert2nx, get_subgraphs, new_norm
from sklearn.preprocessing import normalize

from tunning import merge_features


### GRAPH INVARIANTS - basic graph baseline

def convert_to_vectors_graph_invariants(graphs, labels):
    vectors = []
    for i in range(len(graphs)):
        if i % 50 == 0:
            print('#', end='\n')
        nx_graph = convert2nx(graphs[i], i)

        graph_diameter = nx.diameter(nx_graph)
        average_shortest_path = nx.average_shortest_path_length(nx_graph)
        n_nodes = nx_graph.number_of_nodes()
        node_degrees = [d for _, d in nx_graph.degree()]
        avg_degree =  sum(node_degrees) / float(n_nodes)

        vectors.append(
            [graph_diameter, average_shortest_path, n_nodes, avg_degree]
        )

    return np.array(vectors).astype(np.float32), np.array(labels)



### LDP - original paper baseline

def convert_to_vectors_ldp(dataset, graphs, labels, hyperparams):
    norm_flag = hyperparams['norm_flag']
    
    vectors = []
    for i in range(len(graphs)):
        if i % 50 == 0:
            print('#', end='\n')
        gi = convert2nx(graphs[i], i)
        subgraphs = get_subgraphs(gi)
        gi_s = [function_basis(gi, ['deg'], norm_flag=norm_flag) for gi in subgraphs]
        gi_s = [g for g in gi_s if g != None]
        vectors.append(gi_s)
    if norm_flag == 'no':
        vectors = new_norm(vectors)

    x_original = merge_features(
        dataset, 
        vectors, 
        n_bin=hyperparams['n_bin'], 
        his_norm_flag=hyperparams['his_norm_flag'], 
        cdf_flag=hyperparams['cdf_flag'], 
        uniform_flag=hyperparams['uniform_flag']
    )

    if norm_flag=='yes':
        x = normalize(x_original, axis = 1)
    else:
        x = x_original
    y = np.array(labels)
    return x, y
