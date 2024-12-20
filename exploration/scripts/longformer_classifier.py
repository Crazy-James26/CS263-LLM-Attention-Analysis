import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import LongformerTokenizer, LongformerModel

# Define the Longformer model with classification head
class LongformerClassifier(nn.Module):
    def __init__(self, num_classes, attention_window=512, num_hidden_layers=12):
        super(LongformerClassifier, self).__init__()
        self.longformer = LongformerModel.from_pretrained(
            "allenai/longformer-base-4096",
            attention_window=attention_window,
            num_hidden_layers = num_hidden_layers,
            output_attentions=True  # Enable attention output
        )
        self.dropout = nn.Dropout(0.3)  # Dropout layer
        self.fc = nn.Linear(self.longformer.config.hidden_size, num_classes)  # Fully connected layer

    def forward(self, input_ids, attention_mask, global_attention_mask):
        outputs = self.longformer(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            global_attention_mask=global_attention_mask
        )
        pooled_output = outputs.pooler_output  # Get the pooled output
        pooled_output = self.dropout(pooled_output)  # Apply dropout
        logits = self.fc(pooled_output)  # Get logits from MLP
        return logits, outputs.attentions, outputs.global_attentions  # Return logits and attention weights

# Function to visualize attention
def visualize_attention(attentions, global_attentions, tokens, save_file_name):
    last_attention_weights = attentions
    global_attention_weights = last_attention_weights[:, :, :, :len(global_index)]
    local_attention_weights = last_attention_weights[:, :, :, len(global_index):]
    batch_size, num_heads, seq_len, _ = last_attention_weights.shape

    global_attention_weights_t = global_attentions
    global_attention_weights_t = global_attention_weights_t[:, :, :seq_len, :len(global_index)]

    # Create a plotting window
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    # Draw heatmap for each attention head
    for head in range(num_heads):
        print(head)
        global_attn = torch.zeros((seq_len, seq_len))  # Use the same device
        local_attn = torch.zeros((seq_len, seq_len))  # Use the same device
        half_window = int(model.longformer.config.attention_window[0] / 2)

        # Draw global attention 
        for n, i in enumerate(global_index):
            global_attn[:, i] += global_attention_weights[0, head, :, n]
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

        ax = axes[head // 4, head % 4]
        sns.heatmap(combined_attn.numpy(), cmap='viridis', ax=ax, cbar=True)

        ax.set_title(f'Head {head + 1}')
        ax.set_xlabel('Tokens')
        ax.set_ylabel('Tokens')

        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, rotation=0)

    plt.tight_layout()
    plt.savefig(save_file_name)
    plt.close()
    print("Attention heatmap saved!")

# Example usage
if __name__ == "__main__":
    # Check if a GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = 2  # Change based on your classification task (e.g., binary classification)
    model = LongformerClassifier(num_classes=num_classes, attention_window=16, num_hidden_layers=12).to(device)  # Move model to GPU
    model.eval()  # Set to evaluation mode

    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    sentence = "This is an example sentence."
    sentence = sentence * 15  # Repeat the sentence to increase length
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

    # Move input tensors to the same device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Create the global_attention_mask, marking specific tokens as global tokens
    global_attention_mask = torch.zeros_like(input_ids).to(device)
    global_index = [20, 40, 60, 80]  # Example global attention indices
    for i in global_index:
        global_attention_mask[:, i] = 1

    # Get model outputs
    with torch.no_grad():
        logits, attentions, global_attentions = model(input_ids=input_ids, 
                                                       attention_mask=attention_mask,
                                                       global_attention_mask=global_attention_mask)

    # Print the logits
    print("Logits:", logits)

    # Convert tokens to words for visualization
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu().numpy())  # Move to CPU for conversion

    # Visualize the attention
    visualize_attention(attentions[-1].detach().cpu(), global_attentions[-1].detach().cpu(), tokens, "longformer_classifier_attention_heatmap.png")