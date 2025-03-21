"""
Sentence Analysis Models Setup
==============================
This file combines all the essential code from the original training files into a single setup script.
It provides functions to run different model training methods (RoBERTa, SVM, and naive approach)
for sentence analysis tasks (importance prediction and long-term storage classification).

Author: Your Name
Date: Current Date
"""

# Standard library imports
import os
import glob
import copy
from pathlib import Path

# Data processing imports
import pandas as pd
import numpy as np

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
)
import joblib

# Transformer imports
from transformers import RobertaModel, RobertaTokenizer

#============================================================================
# DATA PREPROCESSING
#============================================================================

def data_preprocess():
    """
    Load and preprocess data from CSV files in the Data folder.
    
    Returns:
        DataFrame: Processed DataFrame with columns ['Sentence', 'Importance', 'Longstorage']
    """
    csv_files = []
    folder = Path("./Data")  # Current directory 
    for file in folder.glob("*.csv"):
        csv_files.append(os.path.join(folder, file.name))
    print(f"Found {len(csv_files)} CSV files: {csv_files}")
    
    df_list = []
    for file in csv_files:
        # Read CSV file
        df = pd.read_csv(file)
        df_list.append(df)

    # Standardize column names for all dataframes
    for i in range(len(df_list)):
        df_list[i].columns = ['Sentence', 'Importance', 'Longstorage']
        
    # Combine all dataframes, ignore_index=True will reset the index
    combined_df = pd.concat(df_list, ignore_index=True, axis=0)

    # Shuffle the rows to ensure random distribution, frac=1 means take all data
    sentence_df = combined_df.sample(frac=1).reset_index(drop=True)
    
    # Create a mapping dictionary to standardize "yes"/"no" values
    replace_dict = {"是": 1, "yes": 1, "Yes": 1, "否": 0, "no": 0, "No": 0}
    
    # Apply the mapping to the 'Longstorage' column (third column)
    sentence_df.iloc[:, 2] = sentence_df.iloc[:, 2].replace(replace_dict)

    return sentence_df

#============================================================================
# PYTORCH DATASET CLASS
#============================================================================

class TextDataset(Dataset):
    """
    Custom PyTorch dataset for text processing with RoBERTa tokenizer.
    
    Args:
        texts (list): List of text strings
        reg_labels (tensor): Regression labels (importance)
        cls_labels (tensor): Classification labels (longstorage)
        tokenizer: Tokenizer instance
        max_length (int): Maximum sequence length for tokenization
    """
    def __init__(self, texts, reg_labels, cls_labels, tokenizer, max_length=128):
        self.texts = texts
        self.reg_labels = reg_labels
        self.cls_labels = cls_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize the text with padding and truncation
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Return a dictionary with all required data
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'reg_label': self.reg_labels[idx],
            'cls_label': self.cls_labels[idx]
        }

#============================================================================
# ROBERTA MODEL ARCHITECTURE
#============================================================================

class ResidualBlock(nn.Module):
    """
    Residual block for neural networks.
    Input and output dimensions are the same, with dimensionality reduction in the middle.
    
    Args:
        dim (int): Input/output dimension
        hidden_dim (int): Reduced dimension for the middle layer
        dropout_prob (float): Dropout probability
    """
    def __init__(self, dim, hidden_dim, dropout_prob=0.1):
        super(ResidualBlock, self).__init__()
        self.proj_down = nn.Linear(dim, hidden_dim)  # Downprojection
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.proj_up = nn.Linear(hidden_dim, dim)    # Upprojection
        self.dropout2 = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Save original input for the residual connection
        residual = x
        
        # Down-projection -> activation -> dropout -> up-projection -> dropout
        out = self.proj_down(x)        # Down: dim -> hidden_dim
        out = self.activation(out)
        out = self.dropout1(out)
        out = self.proj_up(out)        # Up: hidden_dim -> dim
        out = self.dropout2(out)
        
        # Residual connection and layer normalization
        out = out + residual           # Direct addition since dimensions match
        out = self.layer_norm(out)
        
        return out

class DistinguishModel(nn.Module):
    """
    Multi-task model based on RoBERTa for sentence analysis.
    Contains two task heads:
    1. Regression head for importance prediction
    2. Classification head for longstorage classification
    
    Args:
        hidden_dim (int): Hidden dimension for the residual blocks
        dropout_prob (float): Dropout probability
        pretrained_model_name (str): Name of the pretrained model
    """
    def __init__(self, hidden_dim=1536, dropout_prob=0.1, pretrained_model_name='roberta-base'):
        super(DistinguishModel, self).__init__()
        
        # RoBERTa encoder
        self.roberta = RobertaModel.from_pretrained(pretrained_model_name)
        
        # Freeze all parameters first
        for param in self.roberta.parameters():
            param.requires_grad = False
            
        # Unfreeze the last two encoder layers and pooler
        for i in range(10, 12):  # Last two layers
            for param in self.roberta.encoder.layer[i].parameters():
                param.requires_grad = True
        for param in self.roberta.pooler.parameters():
            param.requires_grad = True
        
        # Get RoBERTa's output dimension (usually 768)
        self.encoder_output_dim = self.roberta.config.hidden_size
        
        # Regression head for importance prediction - two-layer architecture
        self.regression_head = nn.ModuleList([
            # First layer: Maintains dimension with residual connection
            ResidualBlock(self.encoder_output_dim, hidden_dim, dropout_prob),
            # Second layer: Output layer with sigmoid to normalize output to [0,1]
            nn.Sequential(nn.Linear(self.encoder_output_dim, 1), nn.Sigmoid())
        ])
        
        # Classification head for longstorage binary classification - two-layer architecture
        self.classification_head = nn.ModuleList([
            # First layer: Maintains dimension with residual connection
            ResidualBlock(self.encoder_output_dim, hidden_dim, dropout_prob),
            # Second layer: Output layer with one output for binary classification
            nn.Linear(self.encoder_output_dim, 1)  # Binary classification, outputs logits
        ])
    
    def forward(self, input_ids, attention_mask=None):
        # Get RoBERTa's output
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embedding = outputs[1]  # [CLS] token representation
        
        # Get predictions through task heads
        reg_features = self.regression_head[0](sentence_embedding)
        regression_output = self.regression_head[1](reg_features)
        
        cls_features = self.classification_head[0](sentence_embedding)
        classification_logits = self.classification_head[1](cls_features)
        
        # Return three values: regression prediction, classification prediction, sentence embedding
        return regression_output, classification_logits, sentence_embedding

#============================================================================
# MODEL TRAINING UTILITY FUNCTIONS
#============================================================================

def collate_fn(batch):
    """
    Collate function for DataLoader to properly batch tokenized inputs.
    
    Args:
        batch: Batch of data from dataset
        
    Returns:
        tuple: Tensors for input_ids, attention_mask, regression labels, classification labels
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    # Ensure labels have correct dimensions (batch_size, 1)
    reg_labels = torch.tensor([item['reg_label'] for item in batch]).reshape(-1, 1)
    cls_labels = torch.tensor([item['cls_label'] for item in batch]).reshape(-1, 1)
    
    return input_ids, attention_mask, reg_labels, cls_labels

#============================================================================
# MODEL TRAINING IMPLEMENTATIONS
#============================================================================

def train_roberta_model(cross_validation=False, n_splits=5, batch_size=32, num_epochs=10000, 
                        patience=25, min_delta=2e-3, target_loss=0.01):
    """
    Train the RoBERTa-based multi-task model.
    
    Args:
        cross_validation (bool): Whether to use cross-validation
        n_splits (int): Number of folds if using cross-validation
        batch_size (int): Batch size for training
        num_epochs (int): Maximum number of epochs
        patience (int): Early stopping patience
        min_delta (float): Minimum improvement threshold for early stopping
        target_loss (float): Target loss to stop training early
        
    Returns:
        None: Results are saved to files
    """
    print("Loading and preprocessing data...")
    sentence_df = data_preprocess()
    
    print("Preparing dataset...")
    # Extract data from DataFrame
    texts = sentence_df['Sentence'].tolist()
    regression_labels = torch.tensor(sentence_df['Importance'].values).float().reshape(-1, 1)
    
    # Ensure Longstorage column has proper numeric values
    sentence_df['Longstorage'] = pd.to_numeric(sentence_df['Longstorage'], errors='coerce')
    sentence_df['Longstorage'] = sentence_df['Longstorage'].fillna(0).astype(int)
    classification_labels = torch.tensor(sentence_df['Longstorage'].values).float().reshape(-1, 1)
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Create dataset
    dataset = TextDataset(
        texts,
        regression_labels,
        classification_labels,
        tokenizer
    )
    
    # Determine device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if cross_validation:
        # Run cross-validation training
        print(f"Starting {n_splits}-fold cross-validation training...")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        all_fold_results = []
        
        # For each fold
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset), start=1):
            print(f"Fold {fold}/{n_splits}")
            fold_dir = f"fold_{fold}"
            os.makedirs(fold_dir, exist_ok=True)
            
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            
            # Initialize model
            model = DistinguishModel(hidden_dim=1536, dropout_prob=0.2)
            model.to(device)
            
            # Set different learning rates for different parts of the model
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in model.roberta.named_parameters() if p.requires_grad],
                    'lr': 1e-5  # Lower learning rate for encoder
                },
                {
                    'params': model.regression_head.parameters(),
                    'lr': 1e-4  # Higher learning rate for task heads
                },
                {
                    'params': model.classification_head.parameters(),
                    'lr': 1e-4  # Higher learning rate for task heads
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
            
            # Training loop
            for epoch in range(num_epochs):
                model.train()
                train_epoch_loss = 0.0
                
                # Process each batch
                for batch in train_loader:
                    input_ids, attention_mask, reg_labels_batch, cls_labels_batch = [b.to(device) for b in batch]
                    
                    # Forward pass
                    optimizer.zero_grad()
                    reg_pred, cls_pred, _ = model(input_ids, attention_mask)
                    
                    # Calculate losses
                    loss_reg = loss_fn_reg(reg_pred, reg_labels_batch)
                    loss_cls = loss_fn_cls(cls_pred, cls_labels_batch)
                    
                    # Backward pass for regression task, retain graph
                    loss_reg.backward(retain_graph=True)
                    # Zero out gradients for classification head
                    for name, param in model.named_parameters():
                        if 'classification_head' in name:
                            param.grad = None
                    
                    # Backward pass for classification task
                    loss_cls.backward()
                    # Zero out gradients for regression head
                    for name, param in model.named_parameters():
                        if 'regression_head' in name:
                            param.grad = None
                    
                    # Update parameters
                    optimizer.step()
                    
                    # Calculate total loss for monitoring
                    total_loss = loss_reg.item() + loss_cls.item()
                    train_epoch_loss += total_loss * input_ids.size(0)
                
                avg_train_loss = train_epoch_loss / len(train_subset)
                train_loss_history.append(avg_train_loss)
                
                # Validation
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
                
                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                lr_history.append(current_lr)
                
                print(f"Fold {fold} Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Early stopping check
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
            
            # Load best model state
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            # Validation inference
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
            
            # Save validation results
            df_val = pd.DataFrame({
                "Regression_Prediction": all_preds_reg,
                "Regression_True": all_true_reg,
                "Classification_Probability": all_preds_cls,
                "Classification_True": all_true_cls
            })
            
            csv_path = os.path.join(fold_dir, "validation_predictions_true.csv")
            df_val.to_csv(csv_path, index=False)
            print(f"Fold {fold} validation results saved to {csv_path}")
            
            # Plot and save training loss curves
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
            
            # Plot and save learning rate curves
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
            
            # Save hyperparameters
            hyperparams = {
                "model_name": "roberta-base",
                "hidden_dim": 1536,
                "dropout_prob": 0.2,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "lr_roberta": optimizer.param_groups[0]['lr'],
                "lr_task_heads": optimizer.param_groups[1]['lr'],
                "optimizer": "Adam",
                "scheduler": "ReduceLROnPlateau (mode='min', factor=0.65, patience=3)",
                "early_stopping_patience": patience,
                "trained_epoch": trained_epoch
            }
            
            with open(os.path.join(fold_dir, "hyperparams.txt"), "w") as f:
                for key, value in hyperparams.items():
                    f.write(f"{key}: {value}\n")
            
            # Save model
            model_path = os.path.join(fold_dir, "distinguish_model.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Fold {fold} model saved to {model_path}\n")
            
            # Collect fold results
            all_fold_results.append({
                "fold": fold,
                "best_val_loss": best_val_loss,
                "final_lr": lr_history[-1],
                "epochs_trained": trained_epoch
            })
        
        # Save cross-validation summary
        results_df = pd.DataFrame(all_fold_results)
        results_df.to_csv("cross_validation_results.csv", index=False)
        print("Cross-validation summary saved to cross_validation_results.csv")
    
    else:
        # Regular training without cross-validation
        print("Starting regular training...")
        
        # Create data loader
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        
        # Initialize model
        model = DistinguishModel(hidden_dim=1536, dropout_prob=0.25)
        model.to(device)
        
        # Set different learning rates for different parts of the model
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.roberta.named_parameters() if p.requires_grad],
                'lr': 1e-5  # Lower learning rate for encoder
            },
            {
                'params': model.regression_head.parameters(),
                'lr': 1e-4  # Higher learning rate for task heads
            },
            {
                'params': model.classification_head.parameters(),
                'lr': 1e-4  # Higher learning rate for task heads
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
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_epoch_loss = 0.0
            
            # Process each batch
            for batch in train_loader:
                input_ids, attention_mask, reg_labels_batch, cls_labels_batch = [b.to(device) for b in batch]
                
                # Forward pass
                optimizer.zero_grad()
                reg_pred, cls_pred, _ = model(input_ids, attention_mask)
                
                # Calculate losses
                loss_reg = loss_fn_reg(reg_pred, reg_labels_batch)
                loss_cls = loss_fn_cls(cls_pred, cls_labels_batch)
                
                # Backward pass for regression task, retain graph
                loss_reg.backward(retain_graph=True)
                # Zero out gradients for classification head
                for name, param in model.named_parameters():
                    if 'classification_head' in name:
                        param.grad = None
                
                # Backward pass for classification task
                loss_cls.backward()
                # Zero out gradients for regression head
                for name, param in model.named_parameters():
                    if 'regression_head' in name:
                        param.grad = None
                
                # Update parameters
                optimizer.step()
                
                # Calculate total loss for monitoring
                total_loss = loss_reg.item() + loss_cls.item()
                train_epoch_loss += total_loss * input_ids.size(0)
            
            avg_train_loss = train_epoch_loss / len(dataset)
            train_loss_history.append(avg_train_loss)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, LR: {current_lr:.6f}")
            
            # Learning rate scheduling
            scheduler.step(avg_train_loss)
            
            # Improved early stopping check
            if avg_train_loss < best_loss - min_delta:  # Only count as improvement if better by min_delta
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
        
        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Save training history
        with open('training_history.txt', 'w') as f:
            f.write("Epoch\tTrain Loss\tLearning Rate\n")
            for epoch, (loss, lr) in enumerate(zip(train_loss_history, lr_history), 1):
                f.write(f"{epoch}\t{loss:.6f}\t{lr:.8f}\n")
            f.write(f"\nBest Loss Achieved: {best_loss:.6f}")
        print("Training history saved to training_history.txt")
        
        # Plot and save training loss curve
        plt.figure()
        plt.plot(range(1, len(train_loss_history)+1), train_loss_history, marker='o', label='Train Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig("loss_curve.png")
        plt.close()
        
        # Plot and save learning rate curve
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

def train_svm_model():
    """
    Train a Support Vector Machine model for both regression and classification tasks.
    Uses TF-IDF vectorization for text feature extraction.
    
    Returns:
        None: Results are saved to files
    """
    print("Training SVM models...")
    
    # Create results directory
    results_dir = "svm_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Load and preprocess data
    sentence_df = data_preprocess()
    print("DataFrame columns:", sentence_df.columns.tolist())

    # Normalize the Longstorage column to binary values (0/1)
    def normalize_yes_no(value):
        if isinstance(value, (int, float)):
            return 1 if value == 1 else 0
        elif isinstance(value, str):
            value = value.strip().lower()
            if value in ['yes', 'y']:
                return 1
            elif value in ['no', 'n']:
                return 0
            else:
                # For unrecognized values, return -1 to indicate invalid
                print(f"Warning: Unrecognized value '{value}', setting to -1")
                return -1
        else:
            return -1

    # Apply conversion function
    sentence_df['Longstorage_binary'] = sentence_df['Longstorage'].apply(normalize_yes_no)

    # Remove entries with invalid labels
    invalid_count = (sentence_df['Longstorage_binary'] == -1).sum()
    if invalid_count > 0:
        print(f"Found {invalid_count} invalid values, removing from dataset")
        sentence_df = sentence_df[sentence_df['Longstorage_binary'] != -1]

    print("\nConverted Longstorage_binary unique values:", sentence_df['Longstorage_binary'].unique())
    print("Count per class:")
    print(sentence_df['Longstorage_binary'].value_counts())

    # Feature extraction with TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(sentence_df["Sentence"])

    # Prepare target variables
    y_reg_importance = sentence_df["Importance"].values  # Regression task: predict Importance
        # Prepare target variables
    y_reg_importance = sentence_df["Importance"].values  # Regression task: predict Importance
    y_clf_longstorage = sentence_df["Longstorage_binary"].values  # Classification task: predict Longstorage

    # Split data into train and test sets for each model
    X_train_importance, X_test_importance, y_train_importance, y_test_importance = train_test_split(
        X, y_reg_importance, test_size=0.2, random_state=42
    )

    X_train_longstorage, X_test_longstorage, y_train_longstorage, y_test_longstorage = train_test_split(
        X, y_clf_longstorage, test_size=0.2, random_state=42
    )

    # =========================================
    # Model 1: Support Vector Regression for Importance
    # =========================================
    print("\n===== Training Importance Regression Model =====")
    svr_importance = SVR(kernel="linear", C=1.0, epsilon=0.1)
    svr_importance.fit(X_train_importance, y_train_importance)
    y_pred_importance = svr_importance.predict(X_test_importance)

    # Calculate regression metrics
    mse_importance = mean_squared_error(y_test_importance, y_pred_importance)
    r2_importance = r2_score(y_test_importance, y_pred_importance)
    print("Importance SVR MSE:", mse_importance)
    print("Importance SVR R2:", r2_importance)

    # Save regression results
    reg_results_importance = pd.DataFrame({
        'True_Values': y_test_importance,
        'Predicted_Values': y_pred_importance,
        'Squared_Error': (y_test_importance - y_pred_importance) ** 2
    })
    reg_results_importance.to_csv(os.path.join(results_dir, 'importance_regression_results.csv'), index=False)

    # Visualize regression results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_importance, y_pred_importance, alpha=0.5)
    plt.plot([min(y_test_importance), max(y_test_importance)], [min(y_test_importance), max(y_test_importance)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('SVR for Importance: True vs Predicted Values')
    plt.savefig(os.path.join(results_dir, 'importance_regression_plot.png'))
    plt.close()

    # =========================================
    # Model 2: Support Vector Classifier for Longstorage
    # =========================================
    print("\n===== Training Longstorage Classification Model =====")
    svc = SVC(kernel="linear", C=1.0, probability=True)
    svc.fit(X_train_longstorage, y_train_longstorage)
    y_pred_longstorage = svc.predict(X_test_longstorage)
    y_pred_longstorage_prob = svc.predict_proba(X_test_longstorage)  # Get probability predictions

    # Calculate classification metrics
    acc = accuracy_score(y_test_longstorage, y_pred_longstorage)
    report = classification_report(y_test_longstorage, y_pred_longstorage)
    print("Longstorage SVC Classification Accuracy:", acc)
    print("\nLongstorage Classification Report:")
    print(report)

    # Save classification results
    clf_results = pd.DataFrame({
        'True_Labels': y_test_longstorage,
        'Predicted_Labels': y_pred_longstorage,
        'Probability_Class_0': y_pred_longstorage_prob[:, 0],
        'Probability_Class_1': y_pred_longstorage_prob[:, 1]
    })
    clf_results.to_csv(os.path.join(results_dir, 'longstorage_classification_results.csv'), index=False)

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test_longstorage, y_pred_longstorage_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Longstorage Classification')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_dir, 'longstorage_roc_curve.png'))
    plt.close()

    # Plot confusion matrix
    cm = confusion_matrix(y_test_longstorage, y_pred_longstorage)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No (0)', 'Yes (1)'],
                yticklabels=['No (0)', 'Yes (1)'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for Longstorage Classification')
    plt.savefig(os.path.join(results_dir, 'longstorage_confusion_matrix.png'))
    plt.close()

    # Save model metrics
    metrics = {
        'Importance_SVR_MSE': mse_importance,
        'Importance_SVR_R2': r2_importance,
        'Longstorage_SVC_Accuracy': acc
    }
    with open(os.path.join(results_dir, 'model_metrics.txt'), 'w') as f:
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value}\n")
        f.write("\nLongstorage Classification Report:\n")
        f.write(report)

    # Save models and preprocessing information
    joblib.dump(svr_importance, os.path.join(results_dir, 'importance_svr_model.pkl'))
    joblib.dump(svc, os.path.join(results_dir, 'longstorage_svc_model.pkl'))
    joblib.dump(vectorizer, os.path.join(results_dir, 'tfidf_vectorizer.pkl'))

    # Create mapping dictionary for reference
    longstorage_mapping = {}
    for val in sentence_df['Longstorage'].unique():
        longstorage_mapping[str(val)] = normalize_yes_no(val)

    # Save mapping information
    mapping_df = pd.DataFrame({
        'Original_Value': list(longstorage_mapping.keys()),
        'Mapped_To': list(longstorage_mapping.values())
    })
    mapping_df.to_csv(os.path.join(results_dir, 'yes_no_mapping.csv'), index=False)

    # Save true value data
    print("\n===== Saving Original True Value Data =====")
    true_values_df = pd.DataFrame({
        'Importance': sentence_df["Importance"].values,
        'Longstorage': sentence_df["Longstorage"].values,
        'Longstorage_binary': sentence_df["Longstorage_binary"].values
    })
    true_values_df.to_csv(os.path.join(results_dir, 'true_values_data.csv'), index=False)

    # Save train and test set true values
    train_importance_df = pd.DataFrame({'True_Importance': y_train_importance})
    test_importance_df = pd.DataFrame({'True_Importance': y_test_importance})
    train_importance_df.to_csv(os.path.join(results_dir, 'train_importance_true_values.csv'), index=False)
    test_importance_df.to_csv(os.path.join(results_dir, 'test_importance_true_values.csv'), index=False)

    train_longstorage_df = pd.DataFrame({'True_Longstorage': y_train_longstorage})
    test_longstorage_df = pd.DataFrame({'True_Longstorage': y_test_longstorage})
    train_longstorage_df.to_csv(os.path.join(results_dir, 'train_longstorage_true_values.csv'), index=False)
    test_longstorage_df.to_csv(os.path.join(results_dir, 'test_longstorage_true_values.csv'), index=False)

    print(f"All results saved to {results_dir} folder")

def train_naive_approach():
    """
    Train a simple word-matching baseline model for the classification task.
    This approach checks if any predefined simple words appear in the text.
    
    Returns:
        None: Results are saved to files
    """
    print("Training naive word-matching approach...")
    
    # Create results folder
    results_folder = "naive_approach_results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Define simple words for matching
    simple_words = [
        'have', 'own', 'possess', 'hold', 'like', 'love', 'prefer', 'fancy', 
        'admire', 'enjoy', 'savor', 'relish', 'appreciate', 'delight in'
    ]

    # Load data
    sentence_df = data_preprocess()
    print("DataFrame columns:", sentence_df.columns.tolist())

    # Get column names for text and labels
    text_column = sentence_df.columns[0]  # First column for text
    label_column = sentence_df.columns[2]  # Third column for classification label

    # Function to normalize yes/no to binary
    def normalize_yes_no(value):
        if isinstance(value, (int, float)):
            return 1 if value == 1 else 0
        elif isinstance(value, str):
            value = value.strip().lower()
            if value in ['yes', 'y', 'si', 'sí', 'oui', 'ja', '是', '是的']:
                return 1
            elif value in ['no', 'n', 'non', 'nein', '否', '不']:
                return 0
            else:
                print(f"Warning: Unrecognized value '{value}', setting to -1")
                return -1
        else:
            return -1

    # Convert labels to binary format
    sentence_df['binary_label'] = sentence_df[label_column].apply(normalize_yes_no)

    # Remove entries with invalid labels
    sentence_df = sentence_df[sentence_df['binary_label'] != -1]

    # Function to check if any simple word is in the text
    def contains_simple_word(text):
        if not isinstance(text, str):
            return 0
        
        text_lower = text.lower()
        for word in simple_words:
            if word in text_lower:
                return 1
        return 0

    # Create predictions based on simple word matching
    sentence_df['prediction'] = sentence_df[text_column].apply(contains_simple_word)

    # Get true values and predictions
    true_values = sentence_df['binary_label']
    predictions = sentence_df['prediction']

    # Calculate metrics
    accuracy = accuracy_score(true_values, predictions)
    precision = precision_score(true_values, predictions, zero_division=0)
    recall = recall_score(true_values, predictions, zero_division=0)
    f1 = f1_score(true_values, predictions, zero_division=0)

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Create confusion matrix
    cm = confusion_matrix(true_values, predictions)
    print("\nConfusion Matrix:")
    print(cm)

    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No (0)', 'Yes (1)'],
                yticklabels=['No (0)', 'Yes (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Simple Word Matching Approach')
    plt.savefig(os.path.join(results_folder, 'confusion_matrix.png'))
    plt.close()

    # Save predictions and true values to CSV
    results_df = pd.DataFrame({
        'Text': sentence_df[text_column],
        'True_Label': true_values,
        'Predicted_Label': predictions,
        'Correct_Prediction': true_values == predictions
    })
    results_df.to_csv(os.path.join(results_folder, 'prediction_results.csv'), index=False)

    # Save metrics to text file
    with open(os.path.join(results_folder, 'metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))

    # Additional visualizations

    # 1. Pie chart of true label distribution
    plt.figure(figsize=(8, 6))
    true_counts = true_values.value_counts()
    plt.pie(true_counts, labels=['No (0)', 'Yes (1)'] if 0 in true_counts.index else ['Yes (1)', 'No (0)'], 
            autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen'])
    plt.title('Distribution of True Labels')
    plt.savefig(os.path.join(results_folder, 'true_label_distribution.png'))
    plt.close()

    # 2. Bar chart comparing prediction success by class
    by_class = results_df.groupby('True_Label')['Correct_Prediction'].mean() * 100
    plt.figure(figsize=(8, 6))
    by_class.plot(kind='bar', color=['lightblue', 'lightgreen'])
    plt.xlabel('True Label')
    plt.ylabel('Prediction Accuracy (%)')
    plt.title('Prediction Accuracy by Class')
    plt.xticks([0, 1], ['No (0)', 'Yes (1)'])
    plt.ylim(0, 100)
    for i, value in enumerate(by_class):
        plt.text(i, value + 2, f"{value:.1f}%", ha='center')
    plt.savefig(os.path.join(results_folder, 'accuracy_by_class.png'))
    plt.close()

    print(f"All results saved to folder: {results_folder}")

#============================================================================
# MAIN FUNCTION
#============================================================================
