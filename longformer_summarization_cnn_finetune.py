import torch
from torch import nn
from transformers import LongformerTokenizer, LongformerForTokenClassification  # Changed imports
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AdamW
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

class SummarizationDataset(Dataset):
    def __init__(self, articles, highlights, tokenizer, max_length=4096):
        self.articles = articles
        self.highlights = highlights
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        article = str(self.articles[idx])
        highlight = str(self.highlights[idx])

        # Tokenize article into sentences
        article_sentences = sent_tokenize(article)
        highlight_sentences = sent_tokenize(highlight)

        # Create sentence-level labels
        sentence_labels = []
        for sent in article_sentences:
            # Simple overlap-based labeling
            is_important = any(self._sentence_overlap(sent, h_sent) > 0.5 
                             for h_sent in highlight_sentences)
            sentence_labels.append(1 if is_important else 0)

        # Tokenize article
        encoding = self.tokenizer.encode_plus(
            article,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Convert sentence labels to token labels
        token_labels = torch.zeros(self.max_length)
        current_pos = 1  # Skip [CLS] token
        
        for sent, label in zip(article_sentences, sentence_labels):
            sent_tokens = self.tokenizer.encode(sent, add_special_tokens=False)
            sent_length = len(sent_tokens)
            if current_pos + sent_length < self.max_length:
                token_labels[current_pos:current_pos + sent_length] = label
                current_pos += sent_length

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': token_labels
        }
    
    def _sentence_overlap(self, sent1, sent2):
        # Simple word overlap ratio
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())
        if not words1 or not words2:
            return 0
        return len(words1.intersection(words2)) / min(len(words1), len(words2))

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
    criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(patience=3, path='ckpts/best_longformer_summarization_model.pt')
    
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits.squeeze(-1), labels.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs.logits.squeeze(-1)) > 0.5).float()
            mask = attention_mask.bool()
            train_correct += ((predictions == labels).float() * mask).sum().item()
            train_total += mask.sum().item()
        
        avg_train_loss = train_loss/len(train_loader)
        train_accuracy = 100 * train_correct/train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits.squeeze(-1), labels.float())
                
                val_loss += loss.item()
                predictions = (torch.sigmoid(outputs.logits.squeeze(-1)) > 0.5).float()
                mask = attention_mask.bool()
                val_correct += ((predictions == labels).float() * mask).sum().item()
                val_total += mask.sum().item()
        
        avg_val_loss = val_loss/len(val_loader)
        val_accuracy = 100 * val_correct/val_total
        
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
        
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    model.load_state_dict(torch.load('ckpts/best_longformer_summarization_model.pt'))
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
            loss = criterion(outputs.logits.squeeze(-1), labels.float())
            
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs.logits.squeeze(-1)) > 0.5).float()
            mask = attention_mask.bool()
            correct += ((predictions == labels).float() * mask).sum().item()
            total += mask.sum().item()
    
    return total_loss/len(data_loader), 100 * correct/total

def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.plot(epochs, history['test_loss'], 'g-', label='Test Loss')
    plt.title('Training, Validation, and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.plot(epochs, history['test_acc'], 'g-', label='Test Accuracy')
    plt.title('Training, Validation, and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ckpts/training_history_longformer_summarization.png')
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load CNN/DailyMail dataset
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    
    # Initialize tokenizer and model with Longformer
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    model = LongformerForTokenClassification.from_pretrained(
        'allenai/longformer-base-4096',
        num_labels=1  # Binary classification for each token
    ).to(device)
    
    # Create datasets
    train_dataset = SummarizationDataset(
        dataset['train']['article'][:1000],  # Limiting for initial testing
        dataset['train']['highlights'][:1000],
        tokenizer
    )
    val_dataset = SummarizationDataset(
        dataset['validation']['article'][:100],
        dataset['validation']['highlights'][:100],
        tokenizer
    )
    test_dataset = SummarizationDataset(
        dataset['test']['article'][:100],
        dataset['test']['highlights'][:100],
        tokenizer
    )

    # Create dataloaders
    batch_size = 4  # Small batch size due to long sequences
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Train the model
    history = train_model(model, train_loader, val_loader, test_loader, device)
    
    # Plot training history
    plot_training_history(history)
    
    # Final evaluation
    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f'\nFinal Test Set Performance:')
    print(f'Loss: {test_loss:.4f}')
    print(f'Accuracy: {test_accuracy:.2f}%')
    
    # Save the model and tokenizer
    # model.save_pretrained('bigbird_summarization_model')
    # tokenizer.save_pretrained('bigbird_summarization_tokenizer')

if __name__ == "__main__":
    main()