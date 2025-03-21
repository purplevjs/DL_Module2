import torch
import numpy as np
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

class DistinguishModel(nn.Module):
    def __init__(self, hidden_dim=1536, dropout_prob=0.1, pretrained_model_name='roberta-base'):
        super(DistinguishModel, self).__init__()
        
        # RoBERTa编码器
        self.roberta = RobertaModel.from_pretrained(pretrained_model_name)
        
        # 冻结除了最后两层encoder和pooler之外的所有层
        # 1. 首先冻结所有参数
        for param in self.roberta.parameters():
            param.requires_grad = False
            
        # 2. 解冻最后两层encoder和pooler
        for i in range(10, 12):  # 最后两层
            for param in self.roberta.encoder.layer[i].parameters():
                param.requires_grad = True
        for param in self.roberta.pooler.parameters():
            param.requires_grad = True
        
        # 获取RoBERTa的输出维度 (通常是768)
        self.encoder_output_dim = self.roberta.config.hidden_size
        
        # 回归任务头 - 两层结构
        self.regression_head = nn.ModuleList([
            # 第一层：保持维度，添加残差连接
            ResidualBlock(self.encoder_output_dim, hidden_dim, dropout_prob),
            # 第二层：输出层
            nn.Sequential(nn.Linear(self.encoder_output_dim, 1), nn.Sigmoid())
        ])
        
        # 分类任务头 - 两层结构
        self.classification_head = nn.ModuleList([
            # 第一层：保持维度，添加残差连接
            ResidualBlock(self.encoder_output_dim, hidden_dim, dropout_prob),
            # 第二层：输出层
            nn.Linear(self.encoder_output_dim, 1)  # 二分类，输出logits
        ])
    
    def forward(self, input_ids, attention_mask=None):
        # 获取RoBERTa的输出
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embedding = outputs[1]  # [CLS]表示
        
        # 通过任务头获取预测结果
        reg_features = self.regression_head[0](sentence_embedding)
        regression_output = self.regression_head[1](reg_features)
        
        cls_features = self.classification_head[0](sentence_embedding)
        classification_logits = self.classification_head[1](cls_features)
        
        # 返回三个值：回归预测、分类预测、句向量
        return regression_output, classification_logits, sentence_embedding

class ResidualBlock(nn.Module):
    """
    残差块：输入和输出维度相同，中间使用降维和升维进行特征转换
    """
    def __init__(self, dim, hidden_dim, dropout_prob=0.1):
        super(ResidualBlock, self).__init__()
        self.proj_down = nn.Linear(dim, hidden_dim)  # 降维投影
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.proj_up = nn.Linear(hidden_dim, dim)    # 升维投影
        self.dropout2 = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # 保存原始输入用于残差连接
        residual = x
        
        # 降维 -> 激活 -> dropout -> 升维 -> dropout
        out = self.proj_down(x)        # 降维：dim -> hidden_dim
        out = self.activation(out)
        out = self.dropout1(out)
        out = self.proj_up(out)        # 升维：hidden_dim -> dim
        out = self.dropout2(out)
        
        # 残差连接和层归一化
        out = out + residual           # 因为维度相同，可以直接相加
        out = self.layer_norm(out)
        
        return out

def train_model(model, train_loader, val_loader, num_epochs, device):
    """
    训练模型的函数
    """
    # 为不同部分设置不同的学习率
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.roberta.named_parameters() if p.requires_grad],
            'lr': 1e-5  # encoder的可训练层使用较小的学习率
        },
        {
            'params': model.regression_head.parameters(),
            'lr': 1e-4  # 任务头使用较大的学习率
        },
        {
            'params': model.classification_head.parameters(),
            'lr': 1e-4  # 任务头使用较大的学习率
        }
    ]
    
    optimizer = torch.optim.Adam(optimizer_grouped_parameters)
    loss_fn_reg = nn.MSELoss()
    loss_fn_cls = nn.BCEWithLogitsLoss()
    
    # 存储训练过程中的句向量
    sentence_vectors_dict = {}
    
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            input_ids, attention_mask, reg_labels, cls_labels = [b.to(device) for b in batch]
            
            # 前向传播
            reg_pred, cls_pred, sentence_vectors = model(input_ids, attention_mask)
            
            # 计算损失
            loss_reg = loss_fn_reg(reg_pred, reg_labels)
            loss_cls = loss_fn_cls(cls_pred, cls_labels)
            
            # 反向传播（使用retain_graph=True保留第一次反向传播的计算图）
            optimizer.zero_grad()
            loss_reg.backward(retain_graph=True)
            # 过滤分类头的梯度
            for name, param in model.named_parameters():
                if 'classification_head' in name:
                    param.grad = None
                    
            loss_cls.backward()
            # 过滤回归头的梯度
            for name, param in model.named_parameters():
                if 'regression_head' in name:
                    param.grad = None
                    
            optimizer.step()
            
            # 存储句向量（例如每个epoch存储一次）
            if batch_idx == 0:  # 可以根据需要调整存储频率
                vectors = sentence_vectors.detach().cpu().numpy()
                batch_ids = input_ids.cpu().numpy()
                for idx, vector in enumerate(vectors):
                    sentence_vectors_dict[f"epoch_{epoch}_batch_{batch_idx}_idx_{idx}"] = vector

        # 验证
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
    保存句向量到文件
    """
    np.save(output_file, vectors_dict)

# 使用示例：
"""
# 1. 初始化模型
model = CombinedModel()
model.to(device)

# 2. 训练模型
sentence_vectors = train_model(model, train_loader, val_loader, num_epochs=10, device=device)

# 3. 保存句向量
save_sentence_vectors(sentence_vectors, 'sentence_vectors.npy')
"""