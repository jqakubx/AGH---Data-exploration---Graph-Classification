import networkx as nx
import numpy as np
from networkx import NetworkXError

from graph import compute_node_features, convert2nx, get_subgraphs, new_norm
from sklearn.preprocessing import normalize

from tunning import merge_features, get_all_feature_keys


### GRAPH INVARIANTS - basic graph baseline

def convert_to_vectors_graph_invariants(graphs, labels):
    vectors = []
    for i in range(len(graphs)):
        # if i % 50 == 0:
        #     print('#', end='\n')
        nx_graph = convert2nx(graphs[i], i)

        try:
            graph_diameter = nx.diameter(nx_graph)
        except NetworkXError:
            graph_diameter = 1
        try:
            average_shortest_path = nx.average_shortest_path_length(nx_graph)
        except NetworkXError:
            average_shortest_path = 1
        n_nodes = nx_graph.number_of_nodes()
        node_degrees = [d for _, d in nx_graph.degree()]
        avg_degree =  sum(node_degrees) / float(n_nodes)

        vectors.append(
            [graph_diameter, average_shortest_path, n_nodes, avg_degree]
        )

    return np.array(vectors).astype(np.float32), np.array(labels)



### LDP - original paper baseline

def convert_to_vectors_ldp(graphs, labels, hyperparams, extended=False):
    norm_flag = hyperparams['norm_flag']

    node_descriptors = ['deg']
    aggregators = ['min', 'max', 'mean', 'std']

    if extended:
        node_descriptors += ['eccentricity', 'load_centrality', 'clustering']
        aggregators += ['skew', 'kurtosis']
    
    graphs_processed = []
    for i in range(len(graphs)):
        nx_graph = convert2nx(graphs[i], i)
        subgraphs = [g for g in get_subgraphs(nx_graph)]
        for g in subgraphs:
            compute_node_features(g, node_descriptors, aggregators, norm_flag)
        subgraphs = [g for g in subgraphs if g != None]
        graphs_processed.append(subgraphs)
    
    feature_keys = get_all_feature_keys(node_descriptors, aggregators)
    if norm_flag == 'no':
        graphs_processed = new_norm(graphs_processed, feature_keys) 

    x_original = merge_features(
        graphs_processed,
        feature_keys,
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
