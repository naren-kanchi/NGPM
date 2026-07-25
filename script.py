import networkx as nx 
from classes import Pattern
from graph_generator import random_graph_generator, generate_shape
from walks import Pattern, random_walk, anonymous_walk

patterns=[]
Graph=random_graph_generator(100, 0.3)

for i in range(10):
    randomwalk_list = random_walk(Graph, 10)
    anonwalk_list = anonymous_walk(randomwalk_list)
    pattern = vars(Pattern(randomwalk_list, anonwalk_list))
    patterns.append(pattern)


