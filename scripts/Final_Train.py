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
# Collate function for data loader
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    # Ensure labels have correct dimensions (batch_size, 1)
    reg_labels = torch.tensor([item['reg_label'] for item in batch]).reshape(-1, 1)
    cls_labels = torch.tensor([item['cls_label'] for item in batch]).reshape(-1, 1)
    
    return input_ids, attention_mask, reg_labels, cls_labels

# Set training hyperparameters
batch_size = 32
num_epochs = 10000  # Set a large maximum number of epochs
patience = 25  # Early stopping patience value
min_delta = 2e-3  # Minimum improvement threshold
target_loss = 0.01  # Target loss value

# Remove KFold related code, directly create data loader
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)



# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize DistinguishModel
model = DistinguishModel(hidden_dim=1536, dropout_prob=0.25)
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
lr_history = []

best_loss = float('inf')
no_improve_count = 0
best_model_state = None

# Start training
for epoch in range(num_epochs):
    model.train()
    train_epoch_loss = 0.0
    
    for batch in train_loader:
        input_ids, attention_mask, reg_labels_batch, cls_labels_batch = [b.to(device) for b in batch]
        
        # Forward pass
        optimizer.zero_grad()
        reg_pred, cls_pred, _ = model(input_ids, attention_mask)
        
        # Calculate losses
        loss_reg = loss_fn_reg(reg_pred, reg_labels_batch)
        loss_cls = loss_fn_cls(cls_pred, cls_labels_batch)
        
        # Backward pass for regression task loss
        loss_reg.backward(retain_graph=True)
        for name, param in model.named_parameters():
            if 'classification_head' in name:
                param.grad = None
        
        # Backward pass for classification task loss
        loss_cls.backward()
        for name, param in model.named_parameters():
            if 'regression_head' in name:
                param.grad = None
        
        optimizer.step()
        
        total_loss = loss_reg.item() + loss_cls.item()
        train_epoch_loss += total_loss * input_ids.size(0)
    
    avg_train_loss = train_epoch_loss / len(dataset)
    train_loss_history.append(avg_train_loss)
    
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, LR: {current_lr:.6f}")
    
    scheduler.step(avg_train_loss)
    
    # Improved early stopping check
    if avg_train_loss < best_loss - min_delta:  # Only count as improvement if it exceeds threshold
        best_loss = avg_train_loss
        no_improve_count = 0
        best_model_state = copy.deepcopy(model.state_dict())
        print(f"Model improved! Best loss: {best_loss:.4f}")
    else:
        no_improve_count += 1
        print(f"No improvement for {no_improve_count} epochs")
    
    # Check if target loss is reached
    if avg_train_loss <= target_loss:
        print(f"Reached target loss of {target_loss}! Stopping training.")
        break
    
    # Early stopping check
    if no_improve_count >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        print(f"Best loss achieved: {best_loss:.4f}")
        break

# Use best model state
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Save loss and learning rate data to txt file
with open('training_history.txt', 'w') as f:
    f.write("Epoch\tTrain Loss\tLearning Rate\n")
    for epoch, (loss, lr) in enumerate(zip(train_loss_history, lr_history), 1):
        f.write(f"{epoch}\t{loss:.6f}\t{lr:.8f}\n")
    f.write(f"\nBest Loss Achieved: {best_loss:.6f}")
print("Training history saved to training_history.txt")

# Save training loss curve
plt.figure()
plt.plot(range(1, len(train_loss_history)+1), train_loss_history, marker='o', label='Train Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")
plt.close()

# Save learning rate curve
plt.figure()
plt.plot(range(1, len(lr_history)+1), lr_history, marker='o', label='Learning Rate')
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Curve")
plt.legend()
plt.grid(True)
plt.savefig("lr_curve.png")
plt.close()

# Save model
torch.save(model.state_dict(), "distinguish_model.pt")
print("Model saved to distinguish_model.pt")