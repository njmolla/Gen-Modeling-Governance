import numpy as np
import networkx as nx

adjacency_matrix = np.ones((3,3))
graph = nx.from_numpy_array(adjacency_matrix,create_using=nx.DiGraph)
print(nx.is_weakly_connected(graph))

