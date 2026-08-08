import torch
import torch.nn as nn
import torch.optim as optim
import copy

from classes import GPMModel, Pattern
from graph_generator import random_graph_generator
from walks import random_walk, anonymous_walk
from train import GraphDataset

def generate_tiny_dataset():
    # Two classes of graphs: sparse and dense
    node_count = 20
    graphs_data = []
    
    # Class 0: Sparse graph
    g0 = random_graph_generator(node_count, 0.4)
    patterns_0 = []
    for _ in range(50): # 50 walks per graph
        while True:
            try:
                rw = random_walk(g0, walk_length=5)
                break
            except IndexError:
                continue
        aw = anonymous_walk(rw)
        patterns_0.append({"semantic_walk": rw, "anonymous_walk": aw})
    graphs_data.append((patterns_0, 0))

    # Class 1: Dense graph
    g1 = random_graph_generator(node_count, 0.9)
    patterns_1 = []
    for _ in range(50):
        while True:
            try:
                rw = random_walk(g1, walk_length=5)
                break
            except IndexError:
                continue
        aw = anonymous_walk(rw)
        patterns_1.append({"semantic_walk": rw, "anonymous_walk": aw})
    graphs_data.append((patterns_1, 1))
    
    return graphs_data, node_count

def run_sanity_test():
    torch.manual_seed(42)
    print("--- Starting End-to-End Sanity Test ---")
    
    graphs_data, node_count = generate_tiny_dataset()
    print(f"Generated {len(graphs_data)} graphs for tiny dataset.")
    print("Shapes at Stage 1 (Graph -> Pattern List):")
    print(f" - Graph 0 has {len(graphs_data[0][0])} patterns.")
    print(f" - Example pattern dict: {graphs_data[0][0][0]}")
    
    # Initialize tiny model
    embedding_dim = 32
    model = GPMModel(node_count=node_count, embedding_dim=embedding_dim, num_classes=2, aggregator_heads=2, aggregator_layers=1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Register forward hooks to capture tensor shapes
    shapes = {}
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                shapes[name] = output[0].shape
            else:
                shapes[name] = output.shape
        return hook

    model.encoder.register_forward_hook(hook_fn("Stage 2: Pattern Encoder Output (1 pattern)"))
    model.aggregator.register_forward_hook(hook_fn("Stage 3: Pattern Aggregator Output (Graph Embedding)"))
    model.classifier.register_forward_hook(hook_fn("Stage 4: Graph Classifier Output (Logits)"))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("\n--- Testing Forward Pass & Shapes ---")
    # Single forward pass for shape tracing
    p_list, label = graphs_data[0]
    out = model(p_list)
    print("Captured Shapes:")
    for name, shape in shapes.items():
        print(f" - {name}: {shape}")
    print(f" - Stage 5: Loss Output (scalar)")

    print("\n--- Verifying Gradients and Parameter Updates ---")
    loss = criterion(out.unsqueeze(0).to(device), torch.tensor([label]).to(device))
    loss.backward()
    
    has_grads = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"Gradients computed successfully for all parameters: {has_grads}")
    
    # Store old params to verify update
    old_params = [p.clone().detach() for p in model.parameters() if p.requires_grad]
    
    optimizer.step()
    optimizer.zero_grad()
    
    new_params = [p.clone().detach() for p in model.parameters() if p.requires_grad]
    params_changed = any(not torch.equal(o, n) for o, n in zip(old_params, new_params))
    print(f"Parameters updated after optimizer.step(): {params_changed}")

    print("\n--- Overfitting Tiny Dataset ---")
    dataset = GraphDataset(graphs_data)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
    
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        
        for batch in train_loader:
            patterns_list, label = batch[0], batch[1]
            label = torch.tensor([label], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            logits = model(patterns_list)
            logits = logits.unsqueeze(0).to(device)
            
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            
        acc = 100 * correct / len(train_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader):.4f} - Acc: {acc:.2f}%")

if __name__ == "__main__":
    run_sanity_test()
