import networkx as nx
from joblib import Parallel, delayed
import pickle
import os, sys
import time
import scipy.stats
import numpy as np

def make_direct(direct):
    # has side effect
    import os
    if not os.path.exists(direct):
            os.makedirs(direct)


def load_graph_pyg(graphs):
    pass


def load_graph(graph, debug='off', single_graph_flag=True):
    # print(f'Loading graph from dataset {graph}')
    file = os.path.join("../data/", graph + ".graph")
    f = open(file, 'rb')
    data = pickle.load(f, encoding='latin1')
    graphs, labels = data['graph'], data['labels']
    return graphs, labels


def convert2nx(graph, i, print_flag='False'):
    # graph: python dict
    keys = graph.keys()
    try:
        assert keys == range(len(graph.keys()))
    except AssertionError:
        # print('%s graph has non consecutive keys'%i)
        # print('Missing nodes are the follwing:')
        for i in range(max(graph.keys())):
            # if i not in graph.keys(): print(i, end=' ')
            pass
    # add nodes
    gi = nx.Graph()
    for i in keys: gi.add_node(i) # change from 1 to i.
    assert len(gi) == len(keys)

    # add edges
    for i in keys:
        for j in graph[i]['neighbors']:
            if j > i:
                gi.add_edge(i, j)
    for i in keys:
        # print graph[i]['label']
        if graph[i]['label']=='':
            gi._node[i]['label'] = 1
            # continue
        try:
            gi._node[i]['label'] = graph[i]['label'][0]
        except TypeError: # modifications for reddit_binary
            gi._node[i]['label'] = graph[i]['label']
        except IndexError:
            gi._node[i]['label'] = 0 # modification for imdb_binary
    assert len(gi._node) == len(graph.keys())
    gi.remove_edges_from(nx.selfloop_edges(gi, data=True))
    if print_flag=='True': print('graph: %s, nnodes: %s, n_edges: %s' %(i, len(gi), len(gi.edges)) )
    return gi


def aggregate_over_neighbors(graph, node_idx, descr, aggregators):
    node = graph._node[node_idx]
    assert descr in node.keys()

    neighbors = graph[node_idx].keys()
    neighbors_descr = [graph._node[j][descr] for j in neighbors]

    agg_lbls_funcs = {
        'mean':     (f'1_0_{descr}_mean',     np.mean),
        'min':      (f'1_0_{descr}_min',      np.min),
        'max':      (f'1_0_{descr}_max',      np.max),
        'std':      (f'1_0_{descr}_std',      np.std),
        'sum':      (f'1_0_{descr}_sum',      np.sum),
        'kurtosis': (f'1_0_{descr}_kurtosis', scipy.stats.kurtosis),
        'skew':     (f'1_0_{descr}_skew',     scipy.stats.skew)
    }

    for agg in aggregators:
        agg_lbl, agg_func = agg_lbls_funcs[agg]

        if len(neighbors_descr) == 0:
            node[agg_lbl] = 0
        else:
            node[agg_lbl] = agg_func(neighbors_descr)


def norm(g, key, flag):
    if flag=='no':
        return 1.0
    elif flag == 'yes':
        return float(np.max(np.abs(nx.get_node_attributes(g, key).values())) + 1e-6)


def get_node_descr_dict(g, descr):
    if descr == 'deg':
        return dict(nx.degree(g))
    else:
        raise Exception(f'Unsupported node descriptor "{descr}"')


def compute_descr_all_nodes(graph, descr, norm_flag):
    descr_dict = get_node_descr_dict(graph, descr)

    for node_idx in graph.nodes():
        node = graph._node[node_idx]
        node[descr] = descr_dict[node_idx]

    descr_norm = norm(graph, descr, norm_flag)

    for node_idx in graph.nodes():
        node = graph._node[node_idx]
        node[descr] /= descr_norm


def normalize_sum(graph, descr, norm_flag):
    if norm_flag == 'yes':
        attr = f'1_0_{descr}_sum'
        attr_norm = norm(graph, attr, norm_flag)
        for node_idx in graph.nodes():
            node = graph._node[node_idx]
            node[attr] /= attr_norm


def compute_node_features(graph, node_descriptors, aggregators, norm_flag='no'):
    # input: graph
    # output: graph with features computed

    if len(graph) < 3:
        return
    
    assert nx.is_connected(graph)

    for descr in node_descriptors:
        compute_descr_all_nodes(graph, descr, norm_flag)
        for node_idx in graph.nodes():
            aggregate_over_neighbors(graph, node_idx, descr, aggregators)
        normalize_sum(graph, descr, norm_flag)
            
    return graph


def get_subgraphs(g, threshold=1):
    assert str(type(g)) == "<class 'networkx.classes.graph.Graph'>"
    subgraphs = [g.subgraph(c).copy() for c in sorted(nx.connected_components(g), key=len, reverse=True)]
    subgraphs = [c for c in subgraphs if len(c) > threshold]
    return subgraphs


def new_norm(graphs_, bl_feat=['1_0_deg_min', '1_0_deg_max', '1_0_deg_mean', '1_0_deg_std', 'deg']):
    """Normalize graph function uniformly"""
    newnorm = dict(zip(bl_feat, [0] * 5))
    for attr in bl_feat:
        for gs in graphs_:
            for g in gs:
                tmp = max(nx.get_node_attributes(g, attr).values())
                if tmp > newnorm[attr]:
                    newnorm[attr] = tmp

    for gs in graphs_:
        for g in gs:
            for n in g.nodes():
                for attr in bl_feat:
                    g._node[n][attr] /= float(newnorm[attr])
                    assert g._node[n][attr] <=1
    return graphs_
