import auxiliarymethods.datasets as dp
from auxiliarymethods.reader import tud_to_networkx

if __name__ == '__main__':
    dataset = "REDDIT-BINARY"
    # Download dataset.
    dp.get_dataset(dataset)
    # Output dataset as a list of graphs.
    graph_db = tud_to_networkx(dataset)
    print(graph_db)