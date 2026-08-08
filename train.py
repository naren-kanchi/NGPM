import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
from classes import GPMModel

# Example Dataset Interface (easily swappable with PyTorch Geometric or standard Dataset)
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        """
        data_list: List of tuples (patterns_list, label)
        - patterns_list: List of pattern dicts for a single graph.
        - label: Integer class label.
        """
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        return self.data_list[idx]

def train_gpm_model(model, train_loader, val_loader, num_epochs=50, lr=1e-3, patience=5, save_path="best_model.pth"):
    """
    Standard PyTorch training loop with early stopping, validation, and checkpointing.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Training Phase
        for batch in train_loader:
            # Batch size is typically 1 graph per iteration because each graph has a different
            # number of sampled walks. If batched processing of variable length sequences is needed,
            # collate_fn handling padding would be required.
            patterns_list, label = batch[0], batch[1]
            
            # Label should be a 1D tensor
            label = torch.tensor([label], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            # GPMModel expects a list of pattern dicts and outputs logits (num_classes,)
            logits = model(patterns_list)
            # Add batch dimension to logits to match target shape for CrossEntropyLoss: (1, num_classes)
            logits = logits.unsqueeze(0).to(device)
            
            loss = criterion(logits, label)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total_train += 1
            correct_train += (predicted == label).sum().item()
            
        train_acc = 100 * correct_train / total_train
        train_loss = running_loss / total_train
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch in val_loader:
                patterns_list, label = batch[0], batch[1]
                label = torch.tensor([label], dtype=torch.long).to(device)
                
                logits = model(patterns_list)
                logits = logits.unsqueeze(0).to(device)
                
                loss = criterion(logits, label)
                val_loss += loss.item()
                
                _, predicted = torch.max(logits.data, 1)
                total_val += 1
                correct_val += (predicted == label).sum().item()
                
        val_acc = 100 * correct_val / total_val
        val_loss = val_loss / total_val
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Checkpointing and Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved with Val Acc: {val_acc:.2f}%")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    # Load best model weights
    model.load_state_dict(best_model_wts)
    print("Training complete.")
    return model
