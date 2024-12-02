import pandas as pd
from Bio import SeqIO
import pandas as pd
import random

import pandas as pd
import itertools

# Load your data
unique_file_path = "zero_shot/negative/unique_sapiens_ids.csv"
train_file_path = "zero_shot/zero_shot_data_random_hidden/positive_test_data_zero_shot.csv"



unique_ids_df = pd.read_csv(unique_file_path)
train_df = pd.read_csv(train_file_path)

# Extract unique IDs and sequences

# Extract unique IDs, sequences, and domains
unique_ids_sequences_domains = unique_ids_df[['Uniprotid', 'Sequence']].drop_duplicates()
train_df = train_df.drop_duplicates(subset=['kinase_sequence', 'kinase_domain'])
kinase_sequences = train_df[['kinase_sequence', 'kinase_domain']].drop_duplicates()

# Set the target number of negative samples
negative_count = 1500

# Randomly sample unique IDs and assign kinase sequences with domains
negative_samples = []
for _ in range(negative_count):
    sampled_row = unique_ids_sequences_domains.sample(n=1).iloc[0]
    sampled_kinase = kinase_sequences.sample(n=1).iloc[0]
    negative_samples.append({
        "UniprotID": sampled_row['Uniprotid'],
        "Sequence": sampled_row['Sequence'],
        "KinaseSequence": sampled_kinase['kinase_sequence'],
        "Domain": sampled_kinase['kinase_domain']
    })


# Convert to DataFrame
negative_data_df = pd.DataFrame(negative_samples)

# Save to CSV
negative_output_path = "zero_shot/zero_shot_data_random_hidden/negative_test_data.csv"
negative_data_df.to_csv(negative_output_path, index=False)

print(f"Negative data saved to {negative_output_path} with {negative_count} samples.")