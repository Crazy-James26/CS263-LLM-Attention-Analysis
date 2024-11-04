import torch
from transformers import BertTokenizer, BertModel
from captum.attr import LayerGradCam
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

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

    def forward(self, inputs_embeds, attention_mask, token_type_ids=None):
        outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        
        return logits  # Return logits and attention weights


# Load a pre-trained BERT model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2
model = BertClassifier(num_classes=num_classes, num_hidden_layers=12).to(device)  # Move model to GPU
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.eval()

# Input text for analysis
text = "This is a test sentence to visualize attention using Grad-CAM in BERT."
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)
token_type_ids = inputs.get('token_type_ids', None).to(device) if 'token_type_ids' in inputs else None
inputs_embeds = model.bert.embeddings(input_ids)


# Initialize Captum's LayerGradCam
layer = model.bert.encoder.layer[-1].attention.self  # Last attention layer
layer_gradcam = LayerGradCam(model, layer.query)  # Using query matrixs
# Calculate attributions without a target class
attributions = layer_gradcam.attribute(inputs=inputs_embeds, additional_forward_args=(attention_mask, token_type_ids), target=0, attr_dim_summation=False)
# Sum across heads and normalize attributions
attributions = attributions.sum(dim=-1).squeeze().detach().cpu().numpy()
attributions = attributions / np.max(attributions)

# Tokenize text and create a visualization
tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

# Visualize attributions as a heatmap
def plot_token_attributions(tokens, attributions):
    fig, ax = plt.subplots(figsize=(10, 1))
    ax.imshow(attributions[np.newaxis, :], cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticks([])
    plt.colorbar(ax.imshow(attributions[np.newaxis, :], cmap="viridis", aspect="auto"))
    plt.show()

plot_token_attributions(tokens, attributions)
