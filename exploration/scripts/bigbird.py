import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BigBirdTokenizer, BigBirdModel

# Define the BigBird model
class BigBirdClassifier(nn.Module):
    def __init__(self, block_size=64):
        super(BigBirdClassifier, self).__init__()
        self.bigbird = BigBirdModel.from_pretrained(
            "google/bigbird-roberta-base",
            block_size=block_size,
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bigbird(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids, output_attentions=True)
        return outputs.attentions  # Return attention from all layers

# Initialize the model
model = BigBirdClassifier(block_size=8)
model.eval()  # Set to evaluation mode

# Load the tokenizer
tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')

# Input sentence
sentence = "This is an example sentence."
sentence = sentence * 15  # Repeat the sentence to increase length
inputs = tokenizer(sentence, return_tensors="pt")

# Get model outputs
with torch.no_grad():
    attentions = model(input_ids=inputs['input_ids'], 
                       attention_mask=inputs['attention_mask'], 
                       token_type_ids=inputs.get('token_type_ids'))

# Get the attention matrix of the last layer
last_attention_weights = attentions[-1]  # Take the last layer's attention

# Visualize the attention of all heads
num_heads = last_attention_weights.shape[1]  # Number of attention heads
print(num_heads)

# Create a plotting window with a 3x4 layout
fig, axes = plt.subplots(3, 4, figsize=(20, 15))  # Adjust figure size to fit all subplots

# Convert tokens to words
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].numpy())

# Draw heatmap for each attention head
for head in range(num_heads):
    attention_head = last_attention_weights[0, head].detach().numpy()  # Get the current attention head for the first sample
    ax = axes[head // 4, head % 4]  # Determine subplot position
    sns.heatmap(attention_head, cmap='viridis', ax=ax, cbar=True)

    # Set title and labels
    ax.set_title(f'Head {head + 1}')
    ax.set_xlabel('Tokens')
    ax.set_ylabel('Tokens')

    # Set x and y axis tick labels to words
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, rotation=0)

# Adjust layout
plt.tight_layout()

# Save the image
plt.savefig("bigbert_attention_heatmap_3x4.png")
plt.close()  # Close the current figure to free memory

print("Attention heatmaps saved!")