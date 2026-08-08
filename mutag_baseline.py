"""
mutag_baseline.py

Honest, reproducible MUTAG baseline for the GPM architecture.

Configuration (CPU-friendly):
  - embedding_dim     = 32
  - nhead             = 4   (head_dim = 8)
  - encoder_layers    = 1
  - aggregator_layers = 1
  - dim_feedforward   = 128
  - walk_length       = 6
  - num_walks         = 35
  - epochs            = 40  (no early stopping)
  - class-weighted CrossEntropyLoss

All patterns are pre-computed ONCE before training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
from collections import Counter

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)

from classes import GPMModel
from walks import random_walk, anonymous_walk

# ── Config ────────────────────────────────────────────────────────────────────
SEED          = 42
EMBEDDING_DIM = 32
NHEAD         = 4          # 32 / 4 = 8 per head — valid
ENC_LAYERS    = 1
AGG_LAYERS    = 1
FEEDFORWARD   = 128
WALK_LENGTH   = 6
NUM_WALKS     = 35
NUM_EPOCHS    = 40
LR            = 5e-4
MUTAG_VOCAB   = 7          # 7 atom types (one-hot → argmax)

TRAIN_FRAC    = 0.70
VAL_FRAC      = 0.15
# TEST_FRAC   = remaining 0.15

# ─────────────────────────────────────────────────────────────────────────────
# 1. ADAPTER: PyG → GPM pattern list
# ─────────────────────────────────────────────────────────────────────────────

def pyg_to_patterns(pyg_data, num_walks=NUM_WALKS, walk_length=WALK_LENGTH):
    G = to_networkx(pyg_data, to_undirected=True)
    node_labels = pyg_data.x.argmax(dim=1).tolist()
    patterns = []
    for _ in range(num_walks):
        while True:
            try:
                rw_ids = random_walk(G, walk_length=walk_length)
                break
            except IndexError:
                continue
        rw_labels = [node_labels[nid] for nid in rw_ids]
        aw = anonymous_walk(rw_ids)
        patterns.append({"semantic_walk": rw_labels, "anonymous_walk": aw})
    return patterns


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD + PRE-COMPUTE (patterns generated once, before training)
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split():
    random.seed(SEED)
    torch.manual_seed(SEED)

    dataset = TUDataset(root='./data/TUDataset', name='MUTAG')
    n = len(dataset)

    print(f"\nMUTAG: {n} graphs | {dataset.num_classes} classes "
          f"| node feature dim = {dataset.num_node_features}")

    # Pre-compute all patterns once
    print(f"Pre-computing patterns ({NUM_WALKS} walks × {WALK_LENGTH} steps per graph)...")
    all_data = []
    for i, pyg_data in enumerate(dataset):
        patterns = pyg_to_patterns(pyg_data)
        label    = pyg_data.y.item()
        all_data.append((patterns, label))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n} done.")
    print("  Pre-computation complete.\n")

    # Shuffle with fixed seed and split
    random.shuffle(all_data)
    n_train = int(TRAIN_FRAC * n)
    n_val   = int(VAL_FRAC   * n)
    train_data = all_data[:n_train]
    val_data   = all_data[n_train: n_train + n_val]
    test_data  = all_data[n_train + n_val:]

    return train_data, val_data, test_data


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLASS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(train_data, num_classes=2):
    labels = [label for _, label in train_data]
    counts = Counter(labels)
    n      = len(labels)
    weights = torch.tensor(
        [n / (num_classes * counts[c]) for c in range(num_classes)],
        dtype=torch.float
    )
    return weights, counts


# ─────────────────────────────────────────────────────────────────────────────
# 4. EPOCH HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, data, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, preds, targets = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for patterns, label in data:
            label_t = torch.tensor([label], dtype=torch.long, device=device)
            logits  = model(patterns).unsqueeze(0).to(device)
            loss    = criterion(logits, label_t)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds.append(logits.argmax(1).item())
            targets.append(label)

    n   = len(data)
    acc = 100.0 * sum(p == t for p, t in zip(preds, targets)) / n
    return total_loss / n, acc, preds, targets


# ─────────────────────────────────────────────────────────────────────────────
# 5. TEST EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_test(preds, targets, majority_class):
    majority_preds = [majority_class] * len(targets)

    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)

    gpm_acc      = accuracy_score(targets, preds)
    baseline_acc = accuracy_score(targets, majority_preds)

    print(f"\n  Majority-class baseline accuracy : {baseline_acc*100:.1f}%")
    print(f"  GPM accuracy                     : {gpm_acc*100:.1f}%")
    print(f"  Delta over baseline              : {(gpm_acc - baseline_acc)*100:+.1f}%")

    print(f"\n  Macro  F1  : {f1_score(targets, preds, average='macro', zero_division=0):.4f}")
    print(f"  Weighted F1: {f1_score(targets, preds, average='weighted', zero_division=0):.4f}")
    print(f"  Precision  : {precision_score(targets, preds, average='macro', zero_division=0):.4f}")
    print(f"  Recall     : {recall_score(targets, preds, average='macro', zero_division=0):.4f}")

    cm = confusion_matrix(targets, preds)
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    print(f"    {cm}")

    return gpm_acc, baseline_acc


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    train_data, val_data, test_data = load_and_split()

    class_weights, class_counts = compute_class_weights(train_data)
    majority_class = max(class_counts, key=class_counts.get)

    # ── Print pre-training stats ──────────────────────────────────────────
    print("="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"  Train : {len(train_data)} graphs | "
          f"class 0: {class_counts[0]}  class 1: {class_counts[1]}")
    print(f"  Val   : {len(val_data)} graphs")
    print(f"  Test  : {len(test_data)} graphs")
    print(f"  Class weights -> 0: {class_weights[0]:.4f}  1: {class_weights[1]:.4f}")
    print(f"  Majority class: {majority_class}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model = GPMModel(
        node_count          = MUTAG_VOCAB,
        embedding_dim       = EMBEDDING_DIM,
        num_classes         = 2,
        nhead               = NHEAD,
        encoder_layers      = ENC_LAYERS,
        encoder_feedforward = FEEDFORWARD,
        aggregator_heads    = NHEAD,
        aggregator_layers   = AGG_LAYERS,
        aggregator_feedforward = FEEDFORWARD,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    final_test_preds, final_test_targets = None, None

    print("\n" + "="*60)
    print("TRAINING (no early stopping)")
    print("="*60)

    for epoch in range(NUM_EPOCHS):
        tr_loss, tr_acc, _, _               = run_epoch(model, train_data, criterion, optimizer, device, train=True)
        vl_loss, vl_acc, vl_preds, vl_tgts = run_epoch(model, val_data,   criterion, optimizer, device, train=False)

        print(f"Epoch {epoch+1:>3}/{NUM_EPOCHS} | "
              f"Train Loss {tr_loss:.4f}  Acc {tr_acc:.1f}% | "
              f"Val Loss {vl_loss:.4f}  Acc {vl_acc:.1f}%")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch, 'val_acc': vl_acc
            }, 'best_mutag_baseline.pth')

    # ── Final test evaluation with best checkpoint ─────────────────────
    model.load_state_dict(best_model_wts)
    _, _, test_preds, test_targets = run_epoch(model, test_data, criterion, optimizer, device, train=False)

    gpm_acc, baseline_acc = evaluate_test(test_preds, test_targets, majority_class)

    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    delta = (gpm_acc - baseline_acc) * 100
    if delta <= 1.0:
        print("  GPM did NOT meaningfully exceed the majority-class baseline.")
        print("  The current simplified pattern representation (atom-type IDs only,")
        print("  no edge features, fixed lambda) is the likely limiting factor.")
        print("  Recommended next step: inspect walk diversity before adding complexity.")
    else:
        print(f"  GPM exceeded the majority-class baseline by {delta:.1f} percentage points.")
        print("  Result is a valid starting point for further investigation.")


if __name__ == "__main__":
    main()
