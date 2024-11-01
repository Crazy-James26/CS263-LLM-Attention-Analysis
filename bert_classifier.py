import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel

# Define the classification model with MLP and attention return
class BertClassifier(nn.Module):
    def __init__(self, num_classes, num_hidden_layers=12):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased",
            attn_implementation="eager",
            num_hidden_layers = num_hidden_layers,
            output_attentions=True,  # Enable attention output
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        
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

    num_classes = 2
    model = BertClassifier(num_classes=num_classes, num_hidden_layers=12).to(device)  # Move model to GPU
    model.eval()  # Set to evaluation mode

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sentence = "This is an example sentence."
    sentence = sentence * 5  # Repeat the sentence to increase length
    inputs = tokenizer(sentence, return_tensors="pt")

    # Move input tensors to the same device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    token_type_ids = inputs.get('token_type_ids', None).to(device) if 'token_type_ids' in inputs else None

    with torch.no_grad():
        logits, attentions = model(input_ids=input_ids, 
                                   attention_mask=attention_mask, 
                                   token_type_ids=token_type_ids)

    # Convert tokens to words for visualization
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu().numpy())  # Move to CPU for conversion

    # Print the logits
    print("Logits:", logits)

    # Visualize the attention of last layer
    visualize_attention(attentions[-1].detach().cpu(), tokens, "bert_classifier_attention_heatmap.png")