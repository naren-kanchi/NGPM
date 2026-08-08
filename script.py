import networkx as nx 
from classes import Pattern, PatternEncoder
from graph_generator import random_graph_generator, generate_shape
from walks import random_walk, anonymous_walk

node_count = 100
embedding_dim = 512

patterns=[]
fin_patterns=[]
Graph=random_graph_generator(node_count, 0.3)

for i in range(10):
    randomwalk_list = random_walk(Graph, 10)
    anonwalk_list = anonymous_walk(randomwalk_list)
    pattern = Pattern(randomwalk_list, anonwalk_list)
    patterns.append(vars(pattern))

encoder = PatternEncoder(node_count, embedding_dim)
for pattern in patterns:
    fin_patterns.append(encoder(pattern))

print(fin_patterns[0])