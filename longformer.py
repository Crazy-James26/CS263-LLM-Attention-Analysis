import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import LongformerTokenizer, LongformerModel

# Define the Longformer model
class LongformerClassifier(nn.Module):
    def __init__(self, attention_window=512):
        super(LongformerClassifier, self).__init__()
        self.longformer = LongformerModel.from_pretrained(
            "allenai/longformer-base-4096",
            attention_window=attention_window,
        )

    def forward(self, input_ids, attention_mask, global_attention_mask):
        outputs = self.longformer(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            global_attention_mask=global_attention_mask,
            output_attentions=True,
        )
        return outputs.attentions, outputs.global_attentions # Return attention from all layers. 
        # outputs.attentions: refer to https://huggingface.co/docs/transformers/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerQuestionAnsweringModelOutput.attentions
        # outputs.global_attentions: refer to https://huggingface.co/docs/transformers/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerQuestionAnsweringModelOutput.global_attentions

# Initialize the model
model = LongformerClassifier(attention_window=16)
model.eval()  # Set to evaluation mode

# Load the tokenizer
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

# Input sentence
sentence = "This is an example sentence."
sentence = sentence * 15  # Repeat the sentence to increase length
inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

# Create the global_attention_mask, marking the first 3 tokens as global tokens
global_attention_mask = torch.zeros_like(inputs['input_ids'])
global_index = [20, 40, 60, 80]
for i in global_index:
    global_attention_mask[:, i] = 1

# Get model outputs
with torch.no_grad():
    attentions, global_attentions = model(input_ids=inputs['input_ids'], 
                       attention_mask=inputs['attention_mask'],
                       global_attention_mask=global_attention_mask)

# Get the attention matrix of the last layer
last_attention_weights = attentions[-1]
global_attention_weights = last_attention_weights[:, :, :, :len(global_index)]
local_attention_weights = last_attention_weights[:, :, :, len(global_index):]
print(last_attention_weights.shape)
batch_size, num_heads, seq_len, _ = last_attention_weights.shape

global_attention_weights_t = global_attentions[-1]
print(global_attention_weights_t.shape)
global_attention_weights_t = global_attention_weights_t[:, :, :seq_len, :len(global_index)]

# Create a plotting window
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Convert tokens to words
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].numpy())

# Draw heatmap for each attention head
for head in range(num_heads):
    global_attn = torch.zeros((seq_len, seq_len))
    local_attn = torch.zeros((seq_len, seq_len))
    half_window = int(model.longformer.config.attention_window[head] / 2)

    # draw global attention 
    for n, i in enumerate(global_index):
        global_attn[:, i] += global_attention_weights[0, head, :, n]
        global_attn[i, :] += global_attention_weights_t[0, head, :, n]

    # draw local attention 
    for i in range(seq_len):
        if i > half_window and seq_len - i > half_window + 1:
            local_attn[i, i - half_window:i + half_window + 1] = local_attention_weights[0, head, i, :]
        elif i <= half_window:
            local_attn[i, :i + half_window + 1] = local_attention_weights[0, head, i, half_window - i:]
        else:
            local_attn[i, i - half_window:] = local_attention_weights[0, head, i, :half_window + (seq_len - i)]

    combined_attn = global_attn + local_attn

    ax = axes[head // 4, head % 4]
    sns.heatmap(combined_attn.detach().numpy(), cmap='viridis', ax=ax, cbar=True)

    ax.set_title(f'Head {head + 1}')
    ax.set_xlabel('Tokens')
    ax.set_ylabel('Tokens')

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, rotation=0)

plt.tight_layout()
plt.savefig("longformer_attention_heatmap_3x4.png")
plt.close()

print("Attention heatmaps saved!")