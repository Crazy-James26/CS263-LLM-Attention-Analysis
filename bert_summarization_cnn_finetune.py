import torch
from torch import nn
from transformers import BertTokenizer, BertForTokenClassification  # Changed from BigBird
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics import roc_auc_score, ndcg_score
import os
import random
import numpy as np
import torch.backends.cudnn as cudnn
nltk.download('punkt')

class SummarizationDataset(Dataset):
    def __init__(self, articles, highlights, tokenizer, max_length=512):  # Changed from 4096
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

        # # Create sentence-level labels
        # sentence_labels = [0] * len(article_sentences)
        # # print("\nlabed highlights:\n")
        # for h_sent in highlight_sentences:
        #     # Find the most related sentence in article_sentences
        #     max_overlap = 0
        #     best_sentence_idx = -1
            
        #     for i, sent in enumerate(article_sentences):
        #         overlap = self._sentence_overlap(sent, h_sent)
        #         if overlap > max_overlap:
        #             max_overlap = overlap
        #             best_sentence_idx = i
            
        #     # Mark the most related sentence as important
        #     if best_sentence_idx >= 0:
        #         sentence_labels[best_sentence_idx] = 1
        #         # print(h_sent)
        #         # print(article_sentences[best_sentence_idx], "\n")
        # # print(sentence_labels)

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

def compute_class_weights(dataset):
    positive_count = 0
    negative_count = 0

    for idx in tqdm(range(len(dataset)), desc='Computing class weights'):
        sample = dataset[idx]
        labels = sample['labels']
        attention_mask = sample['attention_mask']
        mask = attention_mask.bool()
        positive_count += labels[mask].sum().item()
        negative_count += mask.sum().item() - labels[mask].sum().item()

    return positive_count, negative_count

def train_model(model, train_loader, val_loader, test_loader, device, criterion, epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    early_stopping = EarlyStopping(patience=3, path='ckpts/best_bert_summarization_model.pt')
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_ndcg': [],
        'test_loss': [], 'test_acc': [], 'test_auc': [], 'test_ndcg': []
    }
    
    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            loss = criterion(logits, labels.float())
            loss = (loss * attention_mask).sum() / attention_mask.sum()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (torch.sigmoid(logits) > 0.5).float()
            mask = attention_mask.bool()
            train_correct += ((predictions == labels).float() * mask).sum().item()
            train_total += mask.sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation
        avg_val_loss, val_accuracy, val_auc, val_ndcg = evaluate_model(model, val_loader, criterion, device)
        
        # Test set evaluation
        test_loss, test_accuracy, test_auc, test_ndcg = evaluate_model(model, test_loader, criterion, device)
        
        # Store metrics
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        history['val_auc'].append(val_auc)
        history['val_ndcg'].append(val_ndcg)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_accuracy)
        history['test_auc'].append(test_auc)
        history['test_ndcg'].append(test_ndcg)
        
        print(f'\nEpoch {epoch + 1}:')
        print(f'Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
        print(f'Validation AUC: {val_auc:.4f}, NDCG: {val_ndcg:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')
        print(f'Test AUC: {test_auc:.4f}, NDCG: {test_ndcg:.4f}')
        
        # Save the model checkpoint at the end of the epoch
        checkpoint_path = f'ckpts/bert_summarization_epoch_{epoch + 1}.pt'
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Model checkpoint saved to {checkpoint_path}')
        
        # Early stopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Load the best model saved by early stopping
    model.load_state_dict(torch.load('ckpts/best_bert_summarization_model.pt'))
    return history

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            loss = criterion(logits, labels.float())
            loss = (loss * attention_mask).sum() / attention_mask.sum()

            total_loss += loss.item()
            predictions = torch.sigmoid(logits)
            pred_labels = (predictions > 0.5).float()
            mask = attention_mask.bool()
            correct += ((pred_labels == labels).float() * mask).sum().item()
            total += mask.sum().item()

            # Accumulate for AUC calculation
            all_labels.extend(labels[mask].cpu().numpy())
            all_predictions.extend(predictions[mask].cpu().numpy())

    # Compute AUC
    try:
        auc_score = roc_auc_score(all_labels, all_predictions)
    except ValueError:
        # Handle the case when only one class is present in y_true
        auc_score = float('nan')

    # Compute NDCG
    ndcg = ndcg_score([all_labels], [all_predictions])

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy, auc_score, ndcg

def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(20, 10))
    
    # Plot Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.plot(epochs, history['test_loss'], 'g-', label='Test Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.plot(epochs, history['test_acc'], 'g-', label='Test Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Plot AUC
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['val_auc'], 'r-', label='Validation AUC')
    plt.plot(epochs, history['test_auc'], 'g-', label='Test AUC')
    plt.title('AUC Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()

    # Plot NDCG
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['val_ndcg'], 'r-', label='Validation NDCG')
    plt.plot(epochs, history['test_ndcg'], 'g-', label='Test NDCG')
    plt.title('NDCG Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('NDCG')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ckpts/training_history_bert_summarization.png')
    plt.close()

def main():
    # **Set Random Seeds for Reproducibility**
    seed = 42  # You can use any integer as the seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs('ckpts', exist_ok=True)
    
    # Load CNN/DailyMail dataset
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Changed from BigBird
    model = BertForTokenClassification.from_pretrained(
        'bert-base-uncased',  # Changed from BigBird
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

    # **Compute class weights based on the training data**
    positive_count, negative_count = compute_class_weights(train_dataset)
    print(f'Positive samples: {positive_count}, Negative samples: {negative_count}')

    # **Calculate pos_weight for BCEWithLogitsLoss**
    pos_weight = negative_count / positive_count
    pos_weight_tensor = torch.tensor(pos_weight).to(device)
    print(f'Calculated pos_weight: {pos_weight}')

    # Create dataloaders
    batch_size = 4  # Small batch size due to long sequences
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Define the loss function with the calculated pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # **Initial Evaluation Before Training**
    print('\nInitial Evaluation on Test Set Before Fine-tuning:')
    initial_test_loss, initial_test_accuracy, initial_test_auc, initial_test_ndcg = evaluate_model(model, test_loader, criterion, device)
    print(f'Loss: {initial_test_loss:.4f}')
    print(f'Accuracy: {initial_test_accuracy:.2f}%')
    print(f'AUC: {initial_test_auc:.4f}')
    print(f'NDCG: {initial_test_ndcg:.4f}')
    
    # **Initial Evaluation on Validation Set (Optional)**
    print('\nInitial Evaluation on Validation Set Before Fine-tuning:')
    initial_val_loss, initial_val_accuracy, initial_val_auc, initial_val_ndcg = evaluate_model(model, val_loader, criterion, device)
    print(f'Loss: {initial_val_loss:.4f}')
    print(f'Accuracy: {initial_val_accuracy:.2f}%')
    print(f'AUC: {initial_val_auc:.4f}')
    print(f'NDCG: {initial_val_ndcg:.4f}')
    
    # Train the model
    history = train_model(model, train_loader, val_loader, test_loader, device, criterion, epochs=10)
    
    # Plot training history
    plot_training_history(history)
    
    # Final evaluation after training
    test_loss, test_accuracy, test_auc, test_ndcg = evaluate_model(model, test_loader, criterion, device)
    print('\nFinal Test Set Performance After Fine-tuning:')
    print(f'Loss: {test_loss:.4f}')
    print(f'Accuracy: {test_accuracy:.2f}%')
    print(f'AUC: {test_auc:.4f}')
    print(f'NDCG: {test_ndcg:.4f}')
    
    # Save the model and tokenizer
    model.save_pretrained('bert_summarization_model')
    tokenizer.save_pretrained('bert_summarization_tokenizer')

if __name__ == "__main__":
    main()