import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

class FNNTransformerDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

class DistilBERTClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for param in self.encoder.parameters():
            param.requires_grad = False  # freeze encoder
        
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = encoder_outputs.last_hidden_state[:, 0, :]
        cls_token = self.dropout(cls_token)
        logits = self.classifier(cls_token)
        return logits

class DistilBERTModelWrapper:
    def __init__(self, device=None):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBERTClassifier().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=1e-5)

    def encode_texts(self, texts, max_length=512):
        return self.tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

    def create_dataloader(self, texts, labels, batch_size=16, shuffle=True):
        encodings = self.encode_texts(texts)
        dataset = FNNTransformerDataset(encodings, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train(self, train_loader, val_loader, epochs=3):
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)
            print(f"Epoch {epoch+1}: Train loss = {avg_train_loss:.4f}, Val loss = {val_loss:.4f}, Val acc = {val_metrics['accuracy']:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        preds, targets = [], []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                preds.extend(predictions.cpu().numpy())
                targets.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(targets, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='binary')
        metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
        print(f"Eval — Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return avg_loss, metrics

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {filepath}")
