import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import random
from distance_base_negative import compute_negative_samples
import numpy as np


def calculate_class_weights(label_frequencies):
    total = sum(label_frequencies.values())
    return {k: total / v for k, v in label_frequencies.items()}


def independent_exponential_smoothing(weights, alpha=0.9):
    smoothed_weights = {}
    for key, value in weights.items():
        smoothed_weights[key] = alpha * value + (1 - alpha) * np.mean(list(weights.values()))
    return smoothed_weights


def random_pick(samples, max_samples, random_seed):
    np.random.seed(random_seed)
    return np.random.choice(samples, size=min(max_samples, len(samples)), replace=False).tolist()


def prepare_kinase_interaction_samples(dataset_path, task_token, max_length, max_samples, random_seed, logging,
                                       mode='train'):
    positive_df = pd.read_csv(dataset_path)
    positive_samples = positive_df[['sequence', 'kinase_sequence', 'kinase_name']].values.tolist()

    negative_samples = []
    if mode == 'train':
        negative_samples, _ = compute_negative_samples(positive_df, max_length=max_length, k=10)

    samples = positive_samples + negative_samples
    label_index_mapping = {'disconnect': 0, 'connect': 1}
    class_weights = calculate_class_weights({"connect": len(positive_samples), "disconnect": len(negative_samples)})
    class_weights = independent_exponential_smoothing(class_weights)

    new_samples = []
    for protein_sequence, kinase_sequence, kinase_name in samples:
        label = 1 if [protein_sequence, kinase_sequence, kinase_name] in positive_samples else 0
        while len(protein_sequence) + len(kinase_sequence) > max_length - 3:
            protein_sequence = protein_sequence[:int(len(protein_sequence) / 1.1)]
            kinase_sequence = kinase_sequence[:int(len(kinase_sequence) / 1.1)]
        new_samples.append(
            (task_token, [[protein_sequence[:max_length - 2], kinase_sequence[:max_length - 2]]], [label], 'null',
             np.full(2, class_weights[label]))
        )

    new_samples = random_pick(new_samples, max_samples, random_seed)
    logging.info(f'{task_token}: remaining samples: {len(new_samples)}')
    return new_samples, label_index_mapping


class KinaseSubstrateDataset(Dataset):
    def __init__(self, file_path, tokenizer_name, max_length=128, include_negatives=True, mode='train'):
        """
        Dataset class for loading kinase-substrate interaction data with optional negative samples.

        Args:
            file_path (str): Path to the CSV file.
            tokenizer_name (str): Name of the tokenizer model to use.
            max_length (int): Maximum length for tokenized sequences.
            include_negatives (bool): Whether to generate and include negative samples (only for training).
            mode (str): Whether the dataset is for 'train' or 'test'.
        """
        self.data = pd.read_csv(file_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        if mode == 'train' and include_negatives:
            negative_samples, _ = compute_negative_samples(self.data, max_length=max_length, k=10)
            neg_df = pd.DataFrame(negative_samples,
                                  columns=['sequence', 'kinase_sequence', 'kinase_name', 'label'])
            self.data = pd.concat([self.data, neg_df], ignore_index=True)

        if mode == 'test':
            self.data['label'] = 1  # Assign all test samples a positive label
        else:
            self.label_mapping = {'disconnect': 0, 'connect': 1}
            self.data['label'] = self.data['label'].map(self.label_mapping)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        kinase_sequence = row['kinase_sequence']
        substrate_sequence = row['sequence']
        label = torch.tensor(row['label'], dtype=torch.float)

        combined_sequence = f"{kinase_sequence}<EOS>{substrate_sequence}"
        inputs = self.tokenizer(combined_sequence, return_tensors="pt", padding="max_length", truncation=True,
                                max_length=self.max_length, add_special_tokens=True)

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": label
        }


def prepare_dataloader(file_path, tokenizer_name, batch_size=32, max_length=128, shuffle=True, num_workers=4,
                       include_negatives=True, mode='train'):
    dataset = KinaseSubstrateDataset(file_path, tokenizer_name, max_length, include_negatives, mode)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

if __name__ == '__main__':
    train_file = 'kinase_data/positive_new_train_all_columns_cdhit_0.7_kinase_domain.csv'  # Update with actual file path
    test_file = 'kinase_data/positive_data_test_cdhit_70_no_duplicates_v2_pair.csv'  # Update with actual file path
    tokenizer_name = 'facebook/esm2_t12_35M_UR50D'

    train_loader = prepare_dataloader(train_file, tokenizer_name, batch_size=32)
    test_loader = prepare_dataloader(test_file, tokenizer_name, batch_size=32, shuffle=False)

    print("Train dataset size:", len(train_loader.dataset))
    print("Test dataset size:", len(test_loader.dataset))
