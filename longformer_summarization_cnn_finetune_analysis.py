import torch
from transformers import LongformerTokenizer, LongformerForTokenClassification
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
import nltk
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

nltk.download('punkt')
nltk.download('punkt_tab')

def get_attention_scores(model, input_ids, attention_mask):
    """Get attention scores from the model."""
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        attentions = outputs.attentions  # Get attention scores
    return attentions

def word_importance_scores(predictions, input_length):
    """Calculate word importance scores based on predictions."""
    importance_scores = predictions[:input_length] / predictions[:input_length].sum()
    return importance_scores

def compute_correlation(attention_scores, importance_scores):
    """Compute correlation between attention scores and importance scores."""
    correlations = []
    avg_attention = np.mean(attention_scores, axis=1).squeeze()

    for layer in range(avg_attention.shape[0]):
        layer_correlation = []
        for token_idx in range(avg_attention.shape[1]):
            corr, _ = pearsonr(avg_attention[layer, token_idx], importance_scores)
            layer_correlation.append(corr)
        correlations.append(layer_correlation)

    return correlations

def visualize_correlation(correlations, importance_scores, tokens, save_path):
    """Visualize correlations and importance scores."""
    # Create a grid for the subplots
    fig = plt.figure(figsize=(24, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.8, 0.2])  # 80% for the first plot and 20% for the second
    
    # Correlation Heatmap
    ax0 = fig.add_subplot(gs[0])
    sns.heatmap(correlations, ax=ax0, cmap='coolwarm', xticklabels=tokens, 
                yticklabels=[f'Layer {i+1}' for i in range(len(correlations))])
    ax0.set_title('Correlation between Attention Scores and Word Importance')
    ax0.set_xlabel('Token')
    ax0.set_ylabel('Layer')

    # Importance Scores Heatmap
    ax1 = fig.add_subplot(gs[1])
    importance_scores_reshaped = importance_scores.reshape(1, -1)  # Reshape for heatmap
    sns.heatmap(importance_scores_reshaped, ax=ax1, cmap='coolwarm', 
                xticklabels=tokens, yticklabels=['Importance Score'])
    ax1.set_title('Word Importance Scores')
    ax1.set_xlabel('Token')
    ax1.set_ylabel('Importance Score')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def attention_transform(attentions, global_attentions=None, global_index=[]):
    last_attention_weights = attentions
    global_attention_weights = attentions[:, :, :, :len(global_index)]
    local_attention_weights = attentions[:, :, :, len(global_index):]
    batch_size, num_heads, seq_len, window_size = local_attention_weights.shape

    if global_attentions is not None:
        global_attention_weights_t = global_attentions
        global_attention_weights_t = global_attention_weights_t[:, :, :seq_len, :len(global_index)]

    combined_attns =[]
    # Draw heatmap for each attention head
    for head in range(num_heads):
        global_attn = torch.zeros((seq_len, seq_len))  # Ensure tensor is on the same device
        local_attn = torch.zeros((seq_len, seq_len))  # Ensure tensor is on the same device
        half_window = int(window_size / 2)

        # Draw global attention 
        for n, i in enumerate(global_index):
            global_attn[:, i] += global_attention_weights[0, head, :, n]
            if global_attentions is not None:
                global_attn[i, :] += global_attention_weights_t[0, head, :, n]

        # Draw local attention 
        for i in range(seq_len):
            if i > half_window and seq_len - i > half_window + 1:
                local_attn[i, i - half_window:i + half_window + 1] = local_attention_weights[0, head, i, :]
            elif i <= half_window:
                local_attn[i, :i + half_window + 1] = local_attention_weights[0, head, i, half_window - i:]
            else:
                local_attn[i, i - half_window:] = local_attention_weights[0, head, i, :half_window + (seq_len - i)]

        combined_attn = global_attn + local_attn
        combined_attns.append(combined_attn)

    combined_attns=torch.stack(combined_attns)
    return combined_attns

def summarize_and_analyze(model, tokenizer, article, device, max_length=4096):
    """Summarize a single article using the tested model."""
    # Tokenize article into sentences
    article = str(article)
    sentences = sent_tokenize(article)
    
    # Encode the article
    encoding = tokenizer.encode_plus(
        article,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits[0]
    # print(torch.sigmoid(logits.squeeze(-1)))

    # Convert logits to binary labels
    slogits = torch.sigmoid(logits.squeeze(-1))
    predictions = (slogits > 0.5).int().cpu().numpy()
    # print(predictions)
    
    # Extract important sentences based on predictions
    summary = []
    current_pos = 1  # Skip [CLS] token
    
    for sent in sentences:
        sent_tokens = tokenizer.encode(sent, add_special_tokens=False)
        sent_length = len(sent_tokens)
        if current_pos + sent_length < max_length:
            sent_prediction = predictions[current_pos:current_pos + sent_length]
            if sent_prediction.mean() > 0.5:  # If the sentence is mostly predicted as important
                summary.append(sent)
            current_pos += sent_length

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0, :current_pos].cpu().numpy())
    importance_scores = word_importance_scores(slogits[:current_pos].cpu().numpy(), current_pos)
    attentions = torch.stack(
        [attention_transform(attn[:, :, :current_pos, :]) for attn in outputs.attentions]
    ).cpu().numpy()
    correlations = compute_correlation(attentions, importance_scores)
    return '\n'.join(summary), correlations, importance_scores, tokens

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Load the tokenizer and model
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    model = LongformerForTokenClassification.from_pretrained('allenai/longformer-base-4096', num_labels=1, output_attentions=True)
    model_path = './ckpts/best_longformer_summarization_model.pt'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()
    
    # Load CNN/DailyMail dataset
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    test_dataset = {
        'article': dataset['test']['article'][:100],
        'highlights': dataset['test']['highlights'][:100],
    }
    
    # Use the first article from the dataset as an example
    article = test_dataset['article'][0]
    highlights_gt = test_dataset['highlights'][0]
    
    # Generate summary
    highlights, correlations, importance_scores, tokens = summarize_and_analyze(model, tokenizer, article, device)
    
    print("Original Article:\n", article)
    print("\nOriginal highlight:\n", highlights_gt)
    print("\nGenerated highlight:\n", highlights)

    visualize_correlation(correlations, importance_scores, tokens, './ckpts/longformer_summarization_correlations.png')

if __name__ == "__main__":
    main()