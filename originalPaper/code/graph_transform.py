import os

import numpy as np

from originalPaper.code.graph import load_graph

# That function work with some datasets from TUDatasted
# https://chrsmrrs.github.io/datasets/docs/datasets/
# To use it you only have  to copy data to folder data (for example data/redditbinary)
# To folder copy free files from dataset with _A.txt, _graph_indicator.txt and _graph_labels.txt
# As a parameter pass path, for example: redditbinary/REDDIT-BINARY
# In hyperparameter you have to set parameters to your dataset or use existing ones
# for example replace:
# - elif dataset == 'reddit_binary'
# + elif dataset == 'reddit_binary' or dataset == 'redditbinary/REDDIT-BINARY'

def load_graph_tudataset(graph_name):
    file_A = os.path.join("../data/", graph_name + "_A.txt")
    file_indicator = os.path.join("../data/", graph_name + "_graph_indicator.txt")
    file_labels = os.path.join("../data/", graph_name + "_graph_labels.txt")
    f_A = open(file_A, 'r')
    f_ind = open(file_indicator, 'r')
    f_lab = open(file_labels, 'r')

    labels = np.array([])
    graphs = dict()

    print("READ GRAPH")
    # Read labels
    label_lines = f_lab.readlines()
    for line in label_lines:
        labels = np.append(labels, int(line))

    # Read vertices and add them to graph
    vert_lines = f_ind.readlines()
    verts_in_graphs = dict()
    i = 0
    for line in vert_lines:
        graph = int(line) - 1
        if graph not in graphs:
            graphs[graph] = dict()
        graphs[graph][i] = dict()
        graphs[graph][i]['neighbors'] = list()
        graphs[graph][i]['label'] = ''
        verts_in_graphs[i] = graph
        i += 1

    # Read edges
    edge_lines = f_A.readlines()
    for line in edge_lines:
        from_v, to_v = line.strip().split(',')
        from_v, to_v = int(from_v) - 1, int(to_v) - 1
        graph = verts_in_graphs[from_v]
        graphs[graph][from_v]['neighbors'].append(to_v)
        graphs[graph][to_v]['neighbors'].append(from_v)

    for graph in graphs:
        for edge in graphs[graph]:
            graphs[graph][edge]['neighbors'] = list(dict.fromkeys(graphs[graph][edge]['neighbors']))
    print("GRAPH READED")
    return graphs, labels


if __name__ == '__main__':
    graphs, labels = load_graph('imdb_binary')
    print(graphs[0])
    print(labels)
    print(type(labels))
    # graphs, labels = load_graph_new_format("imdb_binary/IMDB-BINARY")
    # print(graphs[0])
    # print(labels)
    # print(type(labels))
    # for graph in graphs:
    #     keys = graphs[0].keys()
    #     print(keys)
    #     k = range(len(graphs[0].keys()))
    #     print(k)
    #     print(keys == k)
    #     print("****")
    # graphs, labels = load_graph_new_format("redditbinary/REDDIT-BINARY")
    # print(graphs[0])