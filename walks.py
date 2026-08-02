import networkx as nx
import random
from classes import Pattern
def random_walk(Graph, walk_length=5):
    start = random.choice(list(Graph.nodes()))
    visited_nodes = []
    visited_nodes.append(start)
    for i in range(walk_length-1):
        next_node = random.choice(list(Graph.neighbors(start)))
        start = next_node
        visited_nodes.append(start)
    return visited_nodes

def anonymous_walk(visit_list):
    anon_walk=[]
    local_list=[]
    for idx,val in enumerate(visit_list):
        anon_walk.append(idx) if val not in local_list else anon_walk.append(local_list.index(val))
        local_list.append(val)
    return anon_walk

# Graph = random_graph_generator(10, 0.2)
# randomwalk_list = random_walk(Graph, 10)
# anonwalk_list = anonymous_walk(randomwalk_list)
# pattern = Pattern(randomwalk_list, anonwalk_list)
# print(f"Graph: {Graph.nodes()}")
# print(f"Random Walk Generated: {pattern.semantic_walk}")
# print(f"Anonymous Walk Generated: {pattern.anonymous_walk}")
