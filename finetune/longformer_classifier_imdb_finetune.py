import torch
from torch import nn
from transformers import LongformerTokenizer, LongformerForSequenceClassification  # Changed import
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from transformers import AdamW

class IMDBDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_length=4096):  # Updated max_length for Longformer
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        target = self.targets[idx]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(target, dtype=torch.long)
        }

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0, path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        print(f'Model saved to {self.path}')

def train_model(model, train_loader, val_loader, test_loader, device, epochs=10):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=3, path='ckpts/best_longformer_imdb_model.pt')
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'test_loss': [], 'test_acc': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
        # for batch in train_loader: 
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss/len(train_loader)
        train_accuracy = 100*train_correct/train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            # for batch in val_loader:
            for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss/len(val_loader)
        val_accuracy = 100*val_correct/val_total
        
        # Test set evaluation
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        
        # Store metrics
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_accuracy)
        
        print(f'Epoch {epoch + 1}:')
        print(f'Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')
        
        # Early stopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Load best model
    model.load_state_dict(torch.load('ckpts/best_longformer_imdb_model.pt'))
    return history

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total_loss/len(data_loader), 100*correct/total

def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.plot(epochs, history['test_loss'], 'g-', label='Test Loss')
    plt.title('Training, Validation, and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.plot(epochs, history['test_acc'], 'g-', label='Test Accuracy')
    plt.title('Training, Validation, and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ckpts/training_history_longformer_imdb.png')
    plt.close()

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load IMDB dataset
    df = pd.read_csv('dataset/IMDB Dataset.csv')
    reviews = df['review'].values
    labels = (df['sentiment'] == 'positive').astype(int).values
    
    # Split the dataset into train, validation, and test sets
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        reviews, labels, test_size=0.3, random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )
    
    # Replace BigBird tokenizer and model with Longformer
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    model = LongformerForSequenceClassification.from_pretrained(
        'allenai/longformer-base-4096',
        num_labels=2
    ).to(device)
    
    # Create datasets
    train_dataset = IMDBDataset(train_texts, train_labels, tokenizer)
    val_dataset = IMDBDataset(val_texts, val_labels, tokenizer)
    test_dataset = IMDBDataset(test_texts, test_labels, tokenizer)

    # Update batch size due to BigBird's memory requirements
    batch_size = 4  # Reduced from 16 due to larger model size
    
    # Create dataloaders with updated batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Train the model
    history = train_model(model, train_loader, val_loader, test_loader, device)
    
    # Plot training history
    plot_training_history(history)
    
    # Final evaluation on test set
    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f'\nFinal Test Set Performance:')
    print(f'Loss: {test_loss:.4f}')
    print(f'Accuracy: {test_accuracy:.2f}%')
    
    # Save the model and tokenizer
    # model.save_pretrained('bert_imdb_model')
    # tokenizer.save_pretrained('bert_imdb_tokenizer')

if __name__ == "__main__":
    main()