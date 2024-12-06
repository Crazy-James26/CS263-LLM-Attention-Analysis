import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from attention_graph_util import *
import seaborn as sns
import itertools 
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
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


import os

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure necessary NLTK resources are available
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('vader_lexicon')

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

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
).to(device)

model.load_state_dict(torch.load('../ckpts/best_bert_imdb_model.pt'))

df = pd.read_csv('../dataset/IMDB Dataset.csv')
reviews = df['review'].values
labels = (df['sentiment'] == 'positive').astype(int).values

train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    reviews, labels, test_size=0.3, random_state=42
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42
)

# Create datasets
train_dataset = IMDBDataset(train_texts, train_labels, tokenizer)
val_dataset = IMDBDataset(val_texts, val_labels, tokenizer)
test_dataset = IMDBDataset(test_texts, test_labels, tokenizer)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=1)

cls_token_id = tokenizer.cls_token_id
sep_token_id = tokenizer.sep_token_id
pad_token_id = tokenizer.pad_token_id
punctuation_ids = [tokenizer.convert_tokens_to_ids(p) for p in [".", ",", "!", "?", ":", ";", "-", "..."]]

all_concat_max_cls_att = []

for j, test_data in tqdm(enumerate(test_loader), total=len(test_loader)):
    input_ids = test_data['input_ids'].to(device)
    attention_mask = test_data['attention_mask'].to(device)
    label = test_data['labels']
    review = test_data['review']
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask, output_attentions=True)
    _attentions = [att.detach().cpu().numpy() for att in output.attentions]
    attentions_mat = np.asarray(_attentions)[:, 0]
    predicted = output['logits']
    del output
    torch.cuda.empty_cache()

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
    valid_token_mask = non_special_token_mask & non_punctuation_mask & content_word_mask

    # Filter CLS attention to only content words across layers
    cls_attention_over_layers = []


    for layer_attention in _attentions:
        # CLS token's attention in this layer
        cls_attention = layer_attention[:, :, 0, :]
        # Apply the combined mask
        cls_attention_to_content_words = cls_attention[:, :, valid_token_mask[0]]
        cls_attention_over_layers.append(cls_attention_to_content_words.mean(axis=1))  # Average over heads

    # Concatenate all layer attentions vertically
    concatenated_attention = np.vstack(cls_attention_over_layers)

    # Extract valid tokens for x-axis labels
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0][valid_token_mask[0]])
    #     print("finished processing tokens...")

    token_labels = tokens

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
        
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # res_att_mat = attentions_mat.sum(axis=1)/attentions_mat.shape[1]
    # res_att_mat = res_att_mat + np.eye(res_att_mat.shape[1])[None,...]
    # res_att_mat = res_att_mat / res_att_mat.sum(axis=-1)[...,None]

    # joint_attentions = compute_joint_attention(res_att_mat, add_residual=False)
    # joint_att_adjmat, joint_labels_to_index = get_adjmat(mat=joint_attentions, input_tokens=tokens)
    
    s_position = 0
    t_positions = np.where(valid_token_mask[0])[0][indices]
    # max_cls_joint_attentions = joint_attentions[:, s_position, t_positions]
    attentions_mat = attentions_mat.mean(axis=1)
    max_cls_attentions = attentions_mat[:, s_position, t_positions]
    concat_max_cls_att = max_cls_attentions[np.newaxis, :]
    
    all_concat_max_cls_att.append(concat_max_cls_att)
    

    if (j % 500) == 1:
        print(f"Iteration {j}")
        np.savez_compressed("/u/scratch/z/zhengton/CS263/output/bert_concat_max_cls_attentions_2.npz",
                           np.vstack(all_concat_max_cls_att))

np.savez_compressed("/u/scratch/z/zhengton/CS263/output/bert_concat_max_cls_attentions_2.npz",
                           np.vstack(all_concat_max_cls_att))