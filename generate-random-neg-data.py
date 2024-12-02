import pandas as pd
import random

# Load the filtered unique CSV and interaction data
filtered_unique_df = pd.read_csv('data/valid_cdhit_0.7_no_duplicates.csv')
interactions_to_remove_df = pd.read_csv('data/valid_kinase_to_remove.csv')

# Find matching rows in the filtered data based on Uniprotid and kinase, and remove them
filtered_unique_df_filtered = filtered_unique_df[
    ~filtered_unique_df[['Uniprotid', 'kinase_sequence']].apply(tuple, axis=1).isin(interactions_to_remove_df[['Uniprotid', 'kinase_sequence']].apply(tuple, axis=1))
]

# Generate negative data by randomly pairing Uniprotid and kinase from filtered_unique_df_filtered
unique_uniprotids = filtered_unique_df_filtered['Uniprotid'].unique()
unique_kinases = filtered_unique_df_filtered['kinase_sequence'].unique()

negative_samples = []
while len(negative_samples) < 1700:
    uniprotid = random.choice(unique_uniprotids)
    kinase = random.choice(unique_kinases)
    if not ((filtered_unique_df_filtered['Uniprotid'] == uniprotid) & (filtered_unique_df_filtered['kinase_sequence'] == kinase)).any():
        negative_samples.append((uniprotid, kinase))

# Create a DataFrame for negative samples
negative_data_df = pd.DataFrame(negative_samples, columns=['Uniprotid', 'kinase_sequence'])

# Create full negative data by manually finding sequences and kinase sequences
negative_data_full = []
for _, row in negative_data_df.iterrows():
    uniprotid = row['Uniprotid']
    kinase = row['kinase_sequence']

    sequence = filtered_unique_df_filtered.loc[filtered_unique_df_filtered['Uniprotid'] == uniprotid, 'Sequence'].values[0]
    kinase_sequence = filtered_unique_df_filtered.loc[filtered_unique_df_filtered['kinase_sequence'] == kinase, 'kinase_sequence'].values[0]
    negative_data_full.append([uniprotid, sequence, kinase_sequence])

negative_data_full_df = pd.DataFrame(negative_data_full, columns=['Uniprotid', 'Sequence','kinase_sequence'])
negative_data_full_df.to_csv('valid_negative.csv')

# train_df_neg = negative_data_full_df.sample(frac=0.8, random_state=1)
# remaining_df = negative_data_full_df.drop(train_df_neg.index)
# validation_df_neg = remaining_df.sample(frac=0.5, random_state=1)
# test_df_neg = remaining_df.drop(validation_df_neg.index)


# train_df_pos = filtered_unique_df.sample(frac=0.8, random_state=1)
# remaining_df = filtered_unique_df.drop(train_df_pos.index)
# validation_df_pos = remaining_df.sample(frac=0.5, random_state=1)
# test_df_pos = remaining_df.drop(validation_df_pos.index)



# Save the training and validation dataframes to new CSV files
# train_df_neg.to_csv('negative_data_train.csv', index=False)
# validation_df_neg.to_csv('negative_data_validation.csv', index=False)
# test_df_neg.to_csv('negative_data_test.csv', index=False)


# train_df_pos.to_csv('positive_data_train_cdhit_70.csv', index=False)
# validation_df_pos.to_csv('positive_data_validation_cdhit_70.csv', index=False)
# test_df_pos.to_csv('positive_data_test_cdhit_70.csv', index=False)

print('done')