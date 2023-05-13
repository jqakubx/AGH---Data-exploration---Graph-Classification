import argparse
import networkx as nx
import numpy as np

from baselines import convert_to_vectors_graph_invariants, convert_to_vectors_ldp
from graph import load_graph
from graph_transform import load_graph_tudataset

from svm import evaluate_svm
from svm_hyperparameter import load_best_svm_params
from random_forest import evaluate_random_forest


def evaluate(dataset, classifier, baseline, hyperparams):
    print("==========\n"\
        + "EVALUATION\n"\
        + "==========\n"
        + f"Classifier: {classifier}\n"
        + f"Baseline: {baseline}\n"
        + f"Dataset: {dataset}\n"
        + "---------------------"
        )

    if dataset == "imdb_binary" or dataset == "imdb_multi":
        graphs, labels = load_graph(dataset)
    else:
        graphs, labels = load_graph_tudataset(dataset)
    x, y = convert_graphs_to_vectors(dataset, graphs, labels, baseline, hyperparams=hyperparams)
    evaluate_with_classifier(classifier, dataset, x, y)


def convert_graphs_to_vectors(dataset, graphs, labels, baseline, hyperparams):
    if baseline == 'ldp':
        return convert_to_vectors_ldp(graphs, labels, hyperparams)
    elif baseline == 'graph_invariants':
        return convert_to_vectors_graph_invariants(graphs, labels)
    elif baseline == 'ldp_extended':
        return convert_to_vectors_ldp(graphs, labels, hyperparams, extended=True)
    else:
        raise Exception('Unsupported baseline')


def evaluate_with_classifier(classifier, dataset, x, y):
    if classifier == 'svm':
        try:
            svm_params = load_best_svm_params(dataset)
        except Exception:
            print(f"No best SVM params found for {dataset}, using defaults.")
            svm_params = {'kernel': 'linear', 'C': 100}
        evaluate_svm(x, y, svm_params, 10, n_eval=10)
    elif classifier == 'random_forest':
        evaluate_random_forest(x, y, 10, n_eval=10)
    else:
        raise Exception('Unsupported classifier')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='imdb_binary', help='dataset')
    parser.add_argument('--n_bin', type=int, default=50, help='number of bins')
    parser.add_argument('--norm_flag', type=str, default='no')

    # for fine tunning
    parser.add_argument('--nonlinear_flag', type=str, default='False',
                        help='True means turn on nonlinear kernel for SVM. In most dataset, linear kernel is already good enough. ')
    parser.add_argument('--uniform_flag', type=bool, default=True,
                        help='uniform or log scale when discretizing a distribution')
    args = parser.parse_args()
    dataset = args.dataset
    
    hyperparams = {
        'n_bin': args.n_bin, # number of bins for historgram
        'norm_flag': args.norm_flag,  # normalize before calling function_basis versus normalize after
        'nonlinear_kernel': args.nonlinear_flag, # SVM linear kernel versus nonlinear kernel

        # less important hyperparameter. Used for fine tunning
        'uniform_flag': args.uniform_flag, # unform versus log scale. True for imdb, False for reddit.
        'cdf_flag': True, # cdf versus pdf. True for most dataset.
        'his_norm_flag': 'yes'
    }

    # evaluate(dataset, classifier='svm', baseline='ldp', hyperparams=hyperparams)
    # evaluate(dataset, classifier='svm', baseline='graph_invariants', hyperparams=hyperparams)
    evaluate(dataset, classifier='random_forest', baseline='ldp', hyperparams=hyperparams)
    evaluate(dataset, classifier='random_forest', baseline='graph_invariants', hyperparams=hyperparams)
    # evaluate(dataset, classifier='random_forest', baseline='ldp', hyperparams=hyperparams)
