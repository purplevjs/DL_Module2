import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd
import copy
from transformers import RobertaTokenizer
from Model_Class import DistinguishModel
from Data_preprocess import data_preprocess
from Dataset_Class import TextDataset


# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


# ------------------------------
sentence_df = data_preprocess()

print(sentence_df.head())
# Extract data from DataFrame and convert to required format
texts = sentence_df['Sentence'].tolist()  # Extract text list

# Convert regression labels to tensor
regression_labels = torch.tensor(sentence_df['Importance'].values).float().reshape(-1, 1)

# Ensure Longstorage column values are numeric
sentence_df['Longstorage'] = pd.to_numeric(sentence_df['Longstorage'], errors='coerce')

# Fill NaN values with 0
sentence_df['Longstorage'] = sentence_df['Longstorage'].fillna(0)

# Convert to integer type
sentence_df['Longstorage'] = sentence_df['Longstorage'].astype(int)

# Convert classification labels to tensor
classification_labels = torch.tensor(sentence_df['Longstorage'].values).float().reshape(-1, 1)


# Create dataset
dataset = TextDataset(
    texts,
    regression_labels,
    classification_labels,
    tokenizer
)

# -------------------------------
# Define cross-validation parameters: five-fold cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Set training hyperparameters
batch_size = 32
num_epochs = 10000
patience = 50

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# For recording validation results summary for each fold
all_fold_results = []

# Collate function for data loader
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    # Ensure labels have correct dimensions (batch_size, 1)
    reg_labels = torch.tensor([item['reg_label'] for item in batch]).reshape(-1, 1)
    cls_labels = torch.tensor([item['cls_label'] for item in batch]).reshape(-1, 1)
    
    return input_ids, attention_mask, reg_labels, cls_labels

# Start cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset), start=1):
    print(f"Fold {fold}/{n_splits}")
    fold_dir = f"fold_{fold}"
    os.makedirs(fold_dir, exist_ok=True)
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Initialize DistinguishModel
    model = DistinguishModel(hidden_dim=1536, dropout_prob=0.2)
    model.to(device)
    
    # Set different learning rates for different parts
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.roberta.named_parameters() if p.requires_grad],
            'lr': 1e-5  # Smaller learning rate for trainable encoder layers
        },
        {
            'params': model.regression_head.parameters(),
            'lr': 1e-4  # Larger learning rate for task heads
        },
        {
            'params': model.classification_head.parameters(),
            'lr': 1e-4  # Larger learning rate for task heads
        }
    ]
    
    optimizer = torch.optim.Adam(optimizer_grouped_parameters)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.65, patience=3, verbose=True
    )
    
    loss_fn_reg = nn.MSELoss()
    loss_fn_cls = nn.BCEWithLogitsLoss()
    
    train_loss_history = []
    val_loss_history = []
    lr_history = []
    
    best_val_loss = float('inf')
    no_improve_count = 0
    best_model_state = None
    
    trained_epoch = 0
    
    # Start training each epoch
    for epoch in range(num_epochs):
                    
        model.train()
        train_epoch_loss = 0.0
        for batch in train_loader:
            input_ids, attention_mask, reg_labels_batch, cls_labels_batch = [b.to(device) for b in batch]
            
            # Forward pass
            optimizer.zero_grad()
            reg_pred, cls_pred, _ = model(input_ids, attention_mask)  # Get all prediction results
            
            # Calculate individual losses
            loss_reg = loss_fn_reg(reg_pred, reg_labels_batch)
            loss_cls = loss_fn_cls(cls_pred, cls_labels_batch)
            
            # Backward pass for regression task loss, retain computation graph
            loss_reg.backward(retain_graph=True)
            # Filter out classification head gradients
            for name, param in model.named_parameters():
                if 'classification_head' in name:
                    param.grad = None
            
            # Backward pass for classification task loss
            loss_cls.backward()  # No need for retain_graph=True, as this is the last backward pass
            # Filter out regression head gradients
            for name, param in model.named_parameters():
                if 'regression_head' in name:
                    param.grad = None
            
            # Update parameters
            optimizer.step()
            
            # Calculate total loss for recording only
            total_loss = loss_reg.item() + loss_cls.item()
            train_epoch_loss += total_loss * input_ids.size(0)
        avg_train_loss = train_epoch_loss / len(train_subset)
        train_loss_history.append(avg_train_loss)
        
        # Calculate average loss on validation set
        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, reg_labels_batch, cls_labels_batch = [b.to(device) for b in batch]
                reg_pred, cls_pred, _ = model(input_ids, attention_mask)
                loss_reg = loss_fn_reg(reg_pred, reg_labels_batch)
                loss_cls = loss_fn_cls(cls_pred, cls_labels_batch)
                loss = loss_reg + loss_cls
                val_epoch_loss += loss.item() * input_ids.size(0)
        avg_val_loss = val_epoch_loss / len(val_subset)
        val_loss_history.append(avg_val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        print(f"Fold {fold} Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")
        
        scheduler.step(avg_val_loss)
        
        # Early stopping check: save model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_count = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping triggered at epoch {epoch+1} for fold {fold}.")
                trained_epoch = epoch
                break
        
    # Load model with best state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Perform inference on validation set and save results
    model.eval()
    all_preds_reg = []
    all_preds_cls = []
    all_true_reg = []
    all_true_cls = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, reg_labels_batch, cls_labels_batch = [b.to(device) for b in batch]
            reg_pred, cls_pred, _ = model(input_ids, attention_mask)
            all_preds_reg.extend(reg_pred.squeeze(1).cpu().numpy())
            all_preds_cls.extend(torch.sigmoid(cls_pred).squeeze(1).cpu().numpy())
            all_true_reg.extend(reg_labels_batch.squeeze(1).cpu().numpy())
            all_true_cls.extend(cls_labels_batch.squeeze(1).cpu().numpy())
    
    df_val = pd.DataFrame({
        "Regression_Prediction": all_preds_reg,
        "Regression_True": all_true_reg,
        "Classification_Probability": all_preds_cls,
        "Classification_True": all_true_cls
    })
    
    csv_path = os.path.join(fold_dir, "validation_predictions_true.csv")
    df_val.to_csv(csv_path, index=False)
    print(f"Fold {fold} validation results saved to {csv_path}")
    
    # Plot and save training loss curve
    plt.figure()
    plt.plot(range(1, len(train_loss_history)+1), train_loss_history, marker='o', label='Train Loss')
    plt.plot(range(1, len(val_loss_history)+1), val_loss_history, marker='o', label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Fold {fold} Loss Curve")
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(fold_dir, "loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Fold {fold} loss curve saved to {loss_plot_path}")
    
    # Plot and save learning rate curve
    plt.figure()
    plt.plot(range(1, len(lr_history)+1), lr_history, marker='o', label='Learning Rate')
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title(f"Fold {fold} Learning Rate Curve")
    plt.legend()
    plt.grid(True)
    lr_plot_path = os.path.join(fold_dir, "lr_curve.png")
    plt.savefig(lr_plot_path)
    plt.close()
    print(f"Fold {fold} learning rate curve saved to {lr_plot_path}")
    
    # Save hyperparameters to text file
    hyperparams = {
        "model_name": "roberta-base",
        "hidden_dim": 512,
        "dropout_prob": 0.1,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr_roberta": optimizer.param_groups[0]['lr'],
        "lr_task_heads": optimizer.param_groups[1]['lr'],
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau (mode='min', factor=0.5, patience=2)",
        "early_stopping_patience": patience,
        "trained_epoch": trained_epoch
    }
    with open(os.path.join(fold_dir, "hyperparams.txt"), "w") as f:
        for key, value in hyperparams.items():
            f.write(f"{key}: {value}\n")
    print(f"Fold {fold} hyperparameters saved.")
    
    # Save model
    model_path = os.path.join(fold_dir, "distinguish_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Fold {fold} model saved to {model_path}\n")
    
    all_fold_results.append({
        "fold": fold,
        "best_val_loss": best_val_loss,
        "final_lr": lr_history[-1],
        "epochs_trained": trained_epoch
    })

# Save summary of all fold results to CSV file
results_df = pd.DataFrame(all_fold_results)
results_df.to_csv("cross_validation_results.csv", index=False)
print("Cross-validation summary saved to cross_validation_results.csv")