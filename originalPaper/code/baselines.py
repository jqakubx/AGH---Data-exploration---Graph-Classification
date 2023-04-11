import argparse
import networkx as nx
import numpy as np
import pickle
import os

from graph_transform import load_graph_tudataset
from graph import load_graph, function_basis, convert2nx, get_subgraphs, new_norm, save_graphs_
from sklearn.preprocessing import normalize

from tunning import merge_features


### GRAPH INVARIANTS - TODO

def convert_to_vector_dataset_using_graph_invariants(dataset):
    raise NotImplemented()


### LDP (baseline from paper)
def calculate_and_save_ldp_baseline(dataset, norm_flag, graphs):
    graphs_ = []
    for i in range(len(graphs)):
        if i % 50 ==0: print('#'),
        gi = convert2nx(graphs[i], i)
        subgraphs = get_subgraphs(gi)
        gi_s = [function_basis(gi, ['deg'], norm_flag=norm_flag) for gi in subgraphs]
        gi_s = [g for g in gi_s if g != None]
        graphs_.append(gi_s)
    if norm_flag == 'no': graphs_ = new_norm(graphs_)
    save_graphs_(graphs_, dataset=dataset, norm_flag=norm_flag)
    return graphs_

def load_or_convert_to_vector_dataset_using_ldp_baseline(dataset, hyperparams):
    norm_flag = hyperparams['norm_flag']
    if dataset == "imdb_binary" or dataset == "imdb_multi":
        graphs, labels = load_graph(dataset)
        try:
            cached_data_path = os.path.join('../data/cache/', dataset, 'norm_flag_' + str(norm_flag), '') + 'graphs_',
            with open(cached_data_path, mode='rb') as cached_data_file:
                preprocessed_graphs = pickle.load(cached_data_file)
        except IOError:
            preprocessed_graphs = calculate_and_save_ldp_baseline(dataset, hyperparams['norm_flag'], graphs)
    else:
        graphs, labels = load_graph_tudataset(dataset)
        preprocessed_graphs = calculate_and_save_ldp_baseline(dataset, hyperparams['norm_flag'], graphs)

    x_original = merge_features(
        dataset, 
        preprocessed_graphs, 
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