"""
mutag_train.py

Full smoke test + training run for the GPM architecture on MUTAG.

Key adapter decision:
  - MUTAG node features are 7-dimensional one-hot vectors (7 atom types).
  - The current PatternEncoder uses nn.Embedding(node_count, embedding_dim).
    This operates on INTEGER indices, not raw float feature vectors.
  - LIMITATION: We therefore map each node's one-hot feature to its argmax
    integer index (0-6), which recovers the atom type as a categorical ID.
  - This means the model encodes atom TYPE, not the full feature vector.
    The one-hot-to-argmax mapping is lossless for MUTAG since the features
    are already pure one-hot. No information is discarded here.
  - Future: to exploit continuous/multi-hot features a linear projection
    layer would be needed inside PatternEncoder. That is a future modification.

MUTAG node vocabulary size = 7 (atom types: C, N, O, F, I, Cl, Br)
MUTAG edge vocabulary = 4 (bond types) — NOT used by current GPM.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

from classes import GPMModel
from walks import random_walk, anonymous_walk
from train import GraphDataset

# ─────────────────────────────────────────────────────────────────────────────
# 1. MUTAG ADAPTER
# ─────────────────────────────────────────────────────────────────────────────

# MUTAG has exactly 7 atom types encoded as one-hot node features.
MUTAG_NODE_VOCAB_SIZE = 7

def pyg_to_gpm_patterns(pyg_data, num_walks=50, walk_length=6):
    """
    Converts one PyG Data object from MUTAG into a GPM-compatible pattern list.

    Steps:
      1. Convert PyG Data -> NetworkX graph (for the walk tokenizer).
      2. Map each node's one-hot feature to its integer atom-type index.
      3. Run random walks over node IDs using the existing tokenizer.
      4. Map the walk's node IDs -> node labels (atom types).
      5. Generate anonymous walks from node IDs.

    Returns:
        patterns: List of {"semantic_walk": List[int], "anonymous_walk": List[int]}
        label:    Integer graph label (0 or 1)
    """
    G = to_networkx(pyg_data, to_undirected=True)

    # node_labels[i] = atom type integer (0..6) for node i
    node_labels = pyg_data.x.argmax(dim=1).tolist()

    patterns = []
    for _ in range(num_walks):
        # Retry if walk lands on an isolated node (safe guard)
        while True:
            try:
                rw_ids = random_walk(G, walk_length=walk_length)
                break
            except IndexError:
                continue

        # Semantic walk = atom-type sequence (passed to nn.Embedding)
        rw_labels = [node_labels[nid] for nid in rw_ids]

        # Anonymous walk = positional-pattern sequence (passed to nn.Embedding)
        aw = anonymous_walk(rw_ids)

        patterns.append({"semantic_walk": rw_labels, "anonymous_walk": aw})

    label = pyg_data.y.item()
    return patterns, label


def load_mutag(num_walks=50, walk_length=6, seed=42):
    """
    Downloads (or caches) MUTAG and converts all graphs.
    Returns a list of (patterns, label) tuples and the split sizes.
    """
    dataset = TUDataset(root='./data/TUDataset', name='MUTAG')
    print(f"MUTAG: {len(dataset)} graphs | {dataset.num_classes} classes "
          f"| node feature dim = {dataset.num_node_features}")

    random.seed(seed)
    torch.manual_seed(seed)

    all_data = []
    for i, data in enumerate(dataset):
        patterns, label = pyg_to_gpm_patterns(data, num_walks=num_walks, walk_length=walk_length)
        all_data.append((patterns, label))

    # 80 / 20 split (reproducible)
    random.shuffle(all_data)
    split = int(0.8 * len(all_data))
    train_data = all_data[:split]
    val_data   = all_data[split:]
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    return train_data, val_data


# ─────────────────────────────────────────────────────────────────────────────
# 2. SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke_test(num_smoke_graphs=3):
    print("\n" + "="*60)
    print("SMOKE TEST")
    print("="*60)

    dataset = TUDataset(root='./data/TUDataset', name='MUTAG')
    embedding_dim = 64

    model = GPMModel(
        node_count=MUTAG_NODE_VOCAB_SIZE,
        embedding_dim=embedding_dim,
        num_classes=2,
        aggregator_heads=4,
        aggregator_layers=2
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    for i in range(num_smoke_graphs):
        data = dataset[i]
        # Use small num_walks just for the smoke test
        patterns, label = pyg_to_gpm_patterns(data, num_walks=10, walk_length=6)

        print(f"\n  Graph {i}: label={label} | nodes={data.num_nodes} | edges={data.num_edges}")
        print(f"    Number of patterns generated : {len(patterns)}")
        print(f"    Example pattern dict         : {patterns[0]}")

        with torch.no_grad():
            # Encode a single pattern to check shape
            single_emb = model.encoder(patterns[0])
            print(f"    Single pattern embedding     : {single_emb.shape}")

            # Encode all patterns once, then reuse for both aggregation and classifier
            encoded = torch.stack([model.encoder(p) for p in patterns])
            graph_emb = model.aggregator(encoded)
            print(f"    Graph embedding (aggregated) : {graph_emb.shape}")

            logits = model.classifier(graph_emb)
            print(f"    Logits                       : {logits.shape}  values={logits.detach().cpu().numpy().round(4)}")

    print("\nSmoke test PASSED.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_mutag(num_epochs=30, lr=1e-3, num_walks=50, walk_length=6, embedding_dim=64, aggregator_layers=2, patience=10, seed=42):
    print("\n" + "="*60)
    print("MUTAG TRAINING RUN")
    print("="*60)

    train_data, val_data = load_mutag(num_walks=num_walks, walk_length=walk_length, seed=seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = GPMModel(
        node_count=MUTAG_NODE_VOCAB_SIZE,
        embedding_dim=embedding_dim,
        num_classes=2,
        aggregator_heads=4,
        aggregator_layers=aggregator_layers
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = torch.utils.data.DataLoader(
        GraphDataset(train_data), batch_size=1, shuffle=True, collate_fn=lambda x: x[0]
    )
    val_loader = torch.utils.data.DataLoader(
        GraphDataset(val_data), batch_size=1, shuffle=False, collate_fn=lambda x: x[0]
    )

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # ── Train ──────────────────────────────
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            patterns_list, label = batch[0], batch[1]
            label_t = torch.tensor([label], dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = model(patterns_list).unsqueeze(0).to(device)
            loss = criterion(logits, label_t)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (logits.argmax(1) == label_t).sum().item()
            total += 1

        train_loss = running_loss / total
        train_acc  = 100 * correct / total

        # ── Validate ───────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                patterns_list, label = batch[0], batch[1]
                label_t = torch.tensor([label], dtype=torch.long, device=device)
                logits = model(patterns_list).unsqueeze(0).to(device)
                val_loss += criterion(logits, label_t).item()
                val_correct += (logits.argmax(1) == label_t).sum().item()
                val_total += 1

        val_loss /= val_total
        val_acc   = 100 * val_correct / val_total

        print(f"Epoch {epoch+1:>3}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.1f}%")

        # ── Checkpoint / Early Stop ─────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc
            }, 'best_mutag_model.pth')
            print(f"  -> Best model saved (Val Acc: {val_acc:.1f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    model.load_state_dict(best_model_wts)
    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.1f}%")
    return model


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_smoke_test(num_smoke_graphs=3)
    train_mutag(
        num_epochs=50,
        lr=5e-4,
        num_walks=50,
        walk_length=6,
        embedding_dim=64,
        aggregator_layers=1,   # 1 layer is faster on CPU while still learning inter-pattern relations
        patience=15
    )
