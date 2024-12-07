import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BigBirdTokenizer, BigBirdModel

# Define the BigBird model with summary generation capability
class BigBirdSummaryGenerator(nn.Module):
    def __init__(self, block_size=64, num_hidden_layers=12):
        super(BigBirdSummaryGenerator, self).__init__()
        self.bigbird = BigBirdModel.from_pretrained(
            "google/bigbird-roberta-base",
            block_size=block_size,
            num_hidden_layers = num_hidden_layers,
            output_attentions=True  # Enable attention output
        )
        self.fc = nn.Linear(self.bigbird.config.hidden_size, self.bigbird.config.vocab_size)  # Fully connected layer for generating token logits

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bigbird(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
        hidden_states = outputs.last_hidden_state  # Get the last hidden states
        logits = self.fc(hidden_states)  # Generate logits for the vocabulary
        return logits, outputs.attentions  # Return logits and attention weights

# Function to visualize attention
def visualize_attention(attentions, tokens, save_file_name):
    num_heads = attentions.shape[1]
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    for head in range(num_heads):
        print(head)
        attention_head = attentions[0, head]
        ax = axes[head // 4, head % 4]
        sns.heatmap(attention_head.numpy(), cmap='viridis', ax=ax, cbar=True)

        ax.set_title(f'Head {head + 1}')
        ax.set_xlabel('Tokens')
        ax.set_ylabel('Tokens')

        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, rotation=0)

    plt.tight_layout()
    plt.savefig(save_file_name)
    plt.close()  # Close to free memory
    print("Attention heatmap saved!")

# Example usage
if __name__ == "__main__":
    # Check if a GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BigBirdSummaryGenerator(block_size=8, num_hidden_layers=12).to(device)  # Move model to GPU
    model.eval()

    tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
    sentence = "This is an example sentence."
    sentence = sentence * 15  # Repeat the sentence to increase length
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

    # Move input tensors to the same device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Get model outputs
    with torch.no_grad():
        logits, attentions = model(input_ids=input_ids, 
                                   attention_mask=attention_mask)
        
    # Use softmax to get probabilities and then get the predicted indices
    predicted_indices = torch.argmax(logits, dim=-1)

    # Convert indices to tokens
    summary_tokens = tokenizer.convert_ids_to_tokens(predicted_indices[0].cpu())  # Move to CPU for conversion
    # Limit to max_length and filter out padding tokens
    max_length = 50
    summary = tokenizer.convert_tokens_to_string([token for token in summary_tokens if token not in ["[PAD]", "[UNK]"]][:max_length])
    print("Generated Summary:", summary)

    # Convert tokens to words for visualization
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu().numpy())  # Move to CPU for conversion

    # Visualize the attention of last layer
    visualize_attention(attentions[-1].detach().cpu(), tokens, "bigbird_summary_generator_attention_heatmap.png")