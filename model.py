import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import yaml
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from utils import load_configs, prepare_saving_dir, prepare_tensorboard, get_optimizer, get_scheduler, save_checkpoint, \
    load_checkpoint, visualize_predictions
import tqdm
import torchmetrics
import lightning as L


class KinaseSubstrateInteractionModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        base_model_name = configs.model.model_name

        # Load pretrained ESM model
        config = AutoConfig.from_pretrained(base_model_name)
        self.esm_model = AutoModel.from_pretrained(base_model_name, config=config)

        # Get correct hidden size from model config
        hidden_size = self.esm_model.config.hidden_size  # Ensure correct size for 650M model
        classifier_hidden = configs.model.classifier_hidden if hasattr(configs.model, 'classifier_hidden') else hidden_size // 2

        # Freeze all layers except the last two
        for param in self.esm_model.parameters():
            param.requires_grad = False
        for param in list(self.esm_model.parameters())[-2:]:
            param.requires_grad = True

        # Classifier for binary interaction prediction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(configs.model.classifier_dropout_rate),
            nn.Linear(classifier_hidden, 1)
        )

    def forward(self, input_ids, attention_mask):
        output = self.esm_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        logits = self.classifier(output)
        return logits

def prepare_model(configs):
    tokenizer = AutoTokenizer.from_pretrained(configs.model.model_name)
    model = KinaseSubstrateInteractionModel(configs)
    return tokenizer, model


if __name__ == '__main__':
    from box import Box
    import yaml

    config_file_path = 'configs/config.yaml'
    with open(config_file_path, 'r') as file:
        config_data = yaml.safe_load(file)
    test_configs = Box(config_data)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = prepare_model(test_configs)
    model.to(device)

    # Define sample kinase and substrate sequences
    kinase_sequence = "MVLSPADKTNVKAAWGKVGAHAGEY"
    substrate_sequence = "KVLSPADKTNVKAAWGKVGAHAGEY"
    combined_sequence = f"{kinase_sequence}<EOS>{substrate_sequence}"
    labels = torch.tensor([[1]], dtype=torch.float)  # Binary label: 1 (interaction), 0 (no interaction)

    # Tokenize combined sequence
    inputs = tokenizer(combined_sequence, return_tensors="pt", padding=True, truncation=True, max_length=128,
                       add_special_tokens=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    labels = labels.to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        print(f"Logits: {logits}")
