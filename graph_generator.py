import networkx as nx

def random_graph_generator(nodecount, edgechance):
    Graph = nx.erdos_renyi_graph(nodecount,edgechance)
    return Graph

def generate_shape(input):
    if input == "triangle":
        Graph = nx.Graph()
        Graph.add_nodes_from([0,1,2])
        Graph.add_edges_from([(0,1),(1,2),(2,0)])
        print(f"Generated Triangle of Three Nodes")
        return Graph

    if input == "chain":
        Graph = nx.Graph()
        Graph.add_nodes_from([0,1,2,3,4])
        Graph.add_edges_from([(0,1),(1,2),(2,3),(3,4)])
        print(f"Generated Chain of Five Nodes")
        return Graph

    if input == "cross":
        Graph = nx.Graph()
        Graph.add_nodes_from([0,1,2,3,4,5])
        Graph.add_edges_from([(0,1),(1,2),(1,3),(1,4),(2,5)])
        print(f"Generated Cross of Six Nodes")
        return Graph

    if input == "square":
        Graph = nx.Graph()
        Graph.add_nodes_from([0,1,2,3])
        Graph.add_edges_from([(0,1),(1,2),(2,3),(3,0)])
        print(f"Generated Square of Four Nodes")
        return Graph

    else:
        print(f"Shape not recognized by the function")

