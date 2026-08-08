import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
import networkx as nx

from walks import random_walk, anonymous_walk
from train import GraphDataset

def load_mutag_data(num_walks=50, walk_length=5):
    """
    Loads MUTAG, converts to NetworkX, and extracts patterns.
    """
    dataset = TUDataset(root='./data/TUDataset', name='MUTAG')
    
    # MUTAG has 7 atom types. Node features (x) are one-hot encoded, 7 dimensions.
    # We will map them back to categorical integers (0 to 6) for the nn.Embedding layer.
    
    data_list = []
    
    print(f"Loading {len(dataset)} MUTAG graphs...")
    for i, data in enumerate(dataset):
        # Convert PyG graph to NetworkX for the tokenizer
        G = to_networkx(data, to_undirected=True)
        
        # Extract node labels (categorical index from one-hot)
        # data.x shape is (num_nodes, 7). argmax gets the integer category.
        node_labels = data.x.argmax(dim=1).tolist()
        
        patterns = []
        for _ in range(num_walks):
            # 1. Generate walk (node IDs)
            # Add a small retry mechanism for disconnected components if any (rare in MUTAG but safe)
            while True:
                try:
                    rw_ids = random_walk(G, walk_length=walk_length)
                    break
                except IndexError:
                    continue
                    
            # 2. Generate anonymous walk from node IDs
            aw = anonymous_walk(rw_ids)
            
            # 3. Map semantic walk to node labels!
            # The current GPM PatternEncoder uses an nn.Embedding, meaning it expects categorical integers.
            # If we pass raw node IDs, the model cannot generalize across graphs.
            # So we pass the atom types (node labels) instead.
            rw_labels = [node_labels[node_id] for node_id in rw_ids]
            
            patterns.append({
                "semantic_walk": rw_labels,
                "anonymous_walk": aw
            })
            
        data_list.append((patterns, data.y.item()))
        
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{len(dataset)} graphs.")
            
    print("Done generating patterns.")
    
    # We need to know the vocab size for the semantic embedding. 
    # MUTAG has 7 atom types.
    node_count = 7 
    
    return data_list, node_count

if __name__ == "__main__":
    # Smoke test
    data_list, node_count = load_mutag_data(num_walks=10, walk_length=5)
    print(f"Number of graphs: {len(data_list)}")
    print(f"Node vocabulary size: {node_count}")
    print(f"Example pattern: {data_list[0][0][0]}")
