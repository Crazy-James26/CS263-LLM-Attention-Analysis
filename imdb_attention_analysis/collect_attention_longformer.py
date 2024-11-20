import torch
from torch import nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import BigBirdTokenizer, BigBirdForSequenceClassification
from transformers import LongformerTokenizer, LongformerForSequenceClassification 
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import seaborn as sns
import networkx as nx
import scipy
from scipy.stats import pearsonr
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import nltk

# Ensure necessary NLTK resources are available
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize VADER Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

class IMDBDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_length=512):
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
            'labels': torch.tensor(target, dtype=torch.long),
            'review': review
        }
        
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
model = LongformerForSequenceClassification.from_pretrained(
    'allenai/longformer-base-4096',
    num_labels=2
).to(device)


model.load_state_dict(torch.load('ckpts/best_longformer_imdb_model.pt'))

df = pd.read_csv('dataset/IMDB Dataset.csv')
reviews = df['review'].values
labels = (df['sentiment'] == 'positive').astype(int).values

train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    reviews, labels, test_size=0.3, random_state=42
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42
)

# Create datasets
train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, max_length=4096)
val_dataset = IMDBDataset(val_texts, val_labels, tokenizer, max_length=4096)
test_dataset = IMDBDataset(test_texts, test_labels, tokenizer, max_length=4096)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=1)

cls_token_id = tokenizer.cls_token_id
sep_token_id = tokenizer.sep_token_id
pad_token_id = tokenizer.pad_token_id
punctuation_ids = [tokenizer.convert_tokens_to_ids(p) for p in [".", ",", "!", "?", ":", ";", "-", "..."]]

all_results = []
for test_data in tqdm(test_loader, total=len(test_loader)):
    try:
        input_ids = test_data['input_ids'].to(device)
        attention_mask = test_data['attention_mask'].to(device)
        label = test_data['labels']
        review = test_data['review']
        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask, output_attentions=True)
        _attentions = [att.detach().cpu().numpy() for att in output.attentions]
        attentions_mat = np.asarray(_attentions)[:,0]
        predicted = output['logits']
    #     print("processing tokens...")
        text = review[0]
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        stop_words = set(stopwords.words("english"))
        content_tokens = [word for word, pos in pos_tags if pos.startswith("NN") or pos.startswith("VB") or pos.startswith("JJ") and word.lower() not in stop_words]

        input_ids = input_ids.to("cpu")

        non_special_token_mask = (input_ids != cls_token_id) & (input_ids != sep_token_id) & (input_ids != pad_token_id)
        non_punctuation_mask = ~torch.isin(input_ids, torch.tensor(punctuation_ids))
        content_word_mask = torch.tensor([1 if tokenizer.decode(id).strip() in content_tokens else 0 for id in input_ids[0]], dtype=torch.bool)

        # Combine all masks
        valid_token_mask = (non_special_token_mask & non_punctuation_mask & content_word_mask).flatten()

        # Extract attention window size
        attention_window = model.config.attention_window[0]

        # Filter CLS attention to content words across layers
        cls_attention_over_layers = []
        for layer_attention in output.attentions:
            # Extract CLS token's global attention in this layer
            cls_attention = layer_attention[:, :, :, 0]  # Shape: (batch_size, num_heads, sequence_length)

            # Apply valid token mask
            cls_attention_to_content_words = cls_attention[:, :, valid_token_mask]  # Shape: (batch_size, num_heads, num_valid_tokens)

            # Average attention across heads
            mean_cls_attention = cls_attention_to_content_words.mean(dim=1)  # Shape: (batch_size, num_valid_tokens)

            # Append mean attention
            cls_attention_over_layers.append(mean_cls_attention.cpu().squeeze().detach().numpy())  # Shape: (num_valid_tokens,)

        # Concatenate attentions across layers
        concatenated_attention = np.flip(np.vstack(cls_attention_over_layers), axis=0)  # Shape: (num_layers, num_valid_tokens)

        del output
        torch.cuda.empty_cache()


        # Extract valid tokens for x-axis labels
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0][valid_token_mask])

        # Process tokens for visualization
        token_labels = []
        for token in tokens:
            if token[0] == 'Ġ':  # Adjust for tokenizer-specific prefixes
                token_labels.append(token[1:].lower())
            else:
                token_labels.append(token.lower())

        labels = {}
        for word in token_labels:
            sentiment_score = sia.polarity_scores(word)['compound']
            labels[word] = sentiment_score

        indices = concatenated_attention.argmax(axis=1)
        max_tokens = []
        max_tokens_scores = []
        max_att_scores = []
        for k, idx in enumerate(indices):
            max_tokens.append(token_labels[idx])
            max_tokens_scores.append(labels[token_labels[idx]])
            max_att_scores.append(concatenated_attention[k, idx])

        all_results.append({
            "sentence": review,
            "label": label,
            "prediction": predicted,
            'max_tokens': max_tokens,
            'max_token_sentiment_scores': max_tokens_scores,
            'max_tokens_attention_scores': max_att_scores
        })
    except:
        continue

with open("output/longformer_imdb_test_results.pickle", 'wb') as f:
    pickle.dump(all_results, f)

