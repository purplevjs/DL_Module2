import torch
import numpy as np
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

class DistinguishModel(nn.Module):
    def __init__(self, hidden_dim=1536, dropout_prob=0.1, pretrained_model_name='roberta-base'):
        super(DistinguishModel, self).__init__()
        
        # RoBERTa encoder
        self.roberta = RobertaModel.from_pretrained(pretrained_model_name)
        
        # Freeze all layers except the last two encoder layers and pooler
        # 1. First freeze all parameters
        for param in self.roberta.parameters():
            param.requires_grad = False
            
        # 2. Unfreeze the last two encoder layers and pooler
        for i in range(10, 12):  # Last two layers
            for param in self.roberta.encoder.layer[i].parameters():
                param.requires_grad = True
        for param in self.roberta.pooler.parameters():
            param.requires_grad = True
        
        # Get RoBERTa's output dimension (usually 768)
        self.encoder_output_dim = self.roberta.config.hidden_size
        
        # Regression task head - two-layer structure
        self.regression_head = nn.ModuleList([
            # First layer: maintain dimension, add residual connection
            ResidualBlock(self.encoder_output_dim, hidden_dim, dropout_prob),
            # Second layer: output layer
            nn.Sequential(nn.Linear(self.encoder_output_dim, 1), nn.Sigmoid())
        ])
        
        # Classification task head - two-layer structure
        self.classification_head = nn.ModuleList([
            # First layer: maintain dimension, add residual connection
            ResidualBlock(self.encoder_output_dim, hidden_dim, dropout_prob),
            # Second layer: output layer
            nn.Linear(self.encoder_output_dim, 1)  # Binary classification, output logits
        ])
    
    def forward(self, input_ids, attention_mask=None):
        # Get RoBERTa's output
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embedding = outputs[1]  # [CLS] representation
        
        # Get predictions through task heads
        reg_features = self.regression_head[0](sentence_embedding)
        regression_output = self.regression_head[1](reg_features)
        
        cls_features = self.classification_head[0](sentence_embedding)
        classification_logits = self.classification_head[1](cls_features)
        
        # Return three values: regression prediction, classification prediction, sentence vector
        return regression_output, classification_logits, sentence_embedding

class ResidualBlock(nn.Module):
    """
    Residual block: input and output dimensions are the same, using dimension reduction and expansion for feature transformation
    """
    def __init__(self, dim, hidden_dim, dropout_prob=0.1):
        super(ResidualBlock, self).__init__()
        self.proj_down = nn.Linear(dim, hidden_dim)  # Dimension reduction projection
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.proj_up = nn.Linear(hidden_dim, dim)    # Dimension expansion projection
        self.dropout2 = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Save original input for residual connection
        residual = x
        
        # Dimension reduction -> activation -> dropout -> dimension expansion -> dropout
        out = self.proj_down(x)        # Reduction: dim -> hidden_dim
        out = self.activation(out)
        out = self.dropout1(out)
        out = self.proj_up(out)        # Expansion: hidden_dim -> dim
        out = self.dropout2(out)
        
        # Residual connection and layer normalization
        out = out + residual           # Direct addition because dimensions match
        out = self.layer_norm(out)
        
        return out

def train_model(model, train_loader, val_loader, num_epochs, device):
    """
    Function to train the model
    """
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
    loss_fn_reg = nn.MSELoss()
    loss_fn_cls = nn.BCEWithLogitsLoss()
    
    # Store sentence vectors during training
    sentence_vectors_dict = {}
    
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            input_ids, attention_mask, reg_labels, cls_labels = [b.to(device) for b in batch]
            
            # Forward pass
            reg_pred, cls_pred, sentence_vectors = model(input_ids, attention_mask)
            
            # Calculate losses
            loss_reg = loss_fn_reg(reg_pred, reg_labels)
            loss_cls = loss_fn_cls(cls_pred, cls_labels)
            
            # Backward pass (use retain_graph=True to keep computation graph from first backward pass)
            optimizer.zero_grad()
            loss_reg.backward(retain_graph=True)
            # Filter out gradients for classification head
            for name, param in model.named_parameters():
                if 'classification_head' in name:
                    param.grad = None
                    
            loss_cls.backward()
            # Filter out gradients for regression head
            for name, param in model.named_parameters():
                if 'regression_head' in name:
                    param.grad = None
                    
            optimizer.step()
            
            # Store sentence vectors (e.g., once per epoch)
            if batch_idx == 0:  # Can adjust storage frequency as needed
                vectors = sentence_vectors.detach().cpu().numpy()
                batch_ids = input_ids.cpu().numpy()
                for idx, vector in enumerate(vectors):
                    sentence_vectors_dict[f"epoch_{epoch}_batch_{batch_idx}_idx_{idx}"] = vector

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss_reg = 0
            val_loss_cls = 0
            for batch in val_loader:
                input_ids, attention_mask, reg_labels, cls_labels = [b.to(device) for b in batch]
                reg_pred, cls_pred, _ = model(input_ids, attention_mask)
                val_loss_reg += loss_fn_reg(reg_pred, reg_labels).item()
                val_loss_cls += loss_fn_cls(cls_pred, cls_labels).item()
            
            print(f"Epoch {epoch}: Val Reg Loss: {val_loss_reg/len(val_loader):.4f}, "
                  f"Val Cls Loss: {val_loss_cls/len(val_loader):.4f}")
    
    return sentence_vectors_dict

def save_sentence_vectors(vectors_dict, output_file):
    """
    Save sentence vectors to file
    """
    np.save(output_file, vectors_dict)

# Usage example:
"""
# 1. Initialize model
model = CombinedModel()
model.to(device)

# 2. Train model
sentence_vectors = train_model(model, train_loader, val_loader, num_epochs=10, device=device)

# 3. Save sentence vectors
save_sentence_vectors(sentence_vectors, 'sentence_vectors.npy')
"""