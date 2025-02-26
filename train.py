import os
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModel
from utils import load_configs, prepare_saving_dir, prepare_tensorboard, get_optimizer, get_scheduler, save_checkpoint, \
    load_checkpoint
from dataset_kinase import prepare_dataloader
from model_kinase import KinaseSubstrateInteractionModel
import torchmetrics
import tqdm
# import lightning as L


def train_model(configs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # fabric = L.Fabric(precision="bf16-mixed")
    # fabric.launch()

    # Load tokenizer and dataloaders
    train_loader = prepare_dataloader(configs.train_settings.train_path, configs.model.model_name,
                                      configs.train_settings.batch_size, mode='train')
    test_loader = prepare_dataloader(configs.test_settings.test_path, configs.model.model_name,
                                     configs.test_settings.batch_size, shuffle=False, mode='test')

    # Initialize model
    model = KinaseSubstrateInteractionModel(configs).to(device)
    optimizer = get_optimizer(model, configs)
    scheduler = get_scheduler(optimizer, configs)

    # Training settings
    num_epochs = configs.train_settings.num_epochs
    grad_clip_norm = configs.train_settings.grad_clip_norm

    # Logging setup
    result_path, checkpoint_path = prepare_saving_dir(configs)
    train_writer, test_writer = prepare_tensorboard(result_path)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        accuracy = torchmetrics.Accuracy(task="binary").to(device)
        f1_score = torchmetrics.F1Score(task="binary", average='macro').to(device)

        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            inputs, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch[
                "labels"].to(device)
            labels = torch.nan_to_num(labels, nan=1.0)

            optimizer.zero_grad()
            logits = model(inputs, attention_mask)
            loss = torch.nn.BCEWithLogitsLoss()(logits.squeeze(), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predictions = torch.sigmoid(logits).round().squeeze(dim=-1)
            accuracy.update(predictions, labels)
            f1_score.update(predictions, labels)

        avg_loss = running_loss / len(train_loader)
        epoch_acc = accuracy.compute().cpu().item()
        epoch_f1 = f1_score.compute().cpu().item()
        accuracy.reset()
        f1_score.reset()

        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Accuracy={epoch_acc:.4f}, F1 Score={epoch_f1:.4f}")

        if (epoch + 1) % configs.checkpoints_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)

        # Evaluate on the test set after each epoch
        model.eval()
        test_loss = 0.0
        test_accuracy = torchmetrics.Accuracy(task="binary").to(device)
        test_f1 = torchmetrics.F1Score(task="binary", average='macro').to(device)

        with torch.no_grad():
            for batch in tqdm.tqdm(test_loader, desc=f"Evaluating Epoch {epoch + 1}"):
                inputs, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
                logits = model(inputs, attention_mask)
                loss = torch.nn.BCEWithLogitsLoss()(logits.squeeze(), labels)
                test_loss += loss.item()
                predictions = torch.sigmoid(logits).round()
                test_accuracy.update(predictions, labels)
                test_f1.update(predictions, labels)

        avg_test_loss = test_loss / len(test_loader)
        test_acc = test_accuracy.compute().cpu().item()
        test_f1_score = test_f1.compute().cpu().item()
        test_accuracy.reset()
        test_f1.reset()

        print(f"Test Set - Loss={avg_test_loss:.4f}, Accuracy={test_acc:.4f}, F1 Score={test_f1_score:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", default='./configs/config.yaml')
    args = parser.parse_args()

    with open(args.config_path) as file:
        config_data = yaml.safe_load(file)
    configs = load_configs(config_data)

    train_model(configs)
