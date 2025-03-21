from torch.utils.data import Dataset


# Create custom dataset class to handle text data
class TextDataset(Dataset):
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
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'reg_label': self.reg_labels[idx],
            'cls_label': self.cls_labels[idx]
        }
