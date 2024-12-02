import pandas as pd

interaction_data = pd.read_csv('data/valid_filtered_unique.csv')

cdhit_file_path = 'data/cdhit/valid_cluster.txt'

with open(cdhit_file_path, 'r') as f:
    clusters = {}
    current_cluster = None

    for line in f:
        if line.startswith('>Cluster'):
            current_cluster = line.strip().split(' ')[1]
            clusters[current_cluster] = []
        elif '>' in line:
            uniprot_id = line.split('>')[1].split('...')[0]
            clusters[current_cluster].append(uniprot_id)

filtered_clusters = {k: v for k, v in clusters.items() if len(set(v)) > 1}

protein_kinase_pairs = interaction_data.groupby('Uniprotid')['kinase_sequence'].apply(list).to_dict()

to_remove_interactions = []

for cluster, proteins in filtered_clusters.items():
    unique_proteins = set(proteins)

    kinase_interactions = {}
    for protein in unique_proteins:
        if protein in protein_kinase_pairs:
            kinases = protein_kinase_pairs[protein]
            for kinase in kinases:
                if kinase not in kinase_interactions:
                    kinase_interactions[kinase] = []
                kinase_interactions[kinase].append(protein)

    for kinase, associated_proteins in kinase_interactions.items():
        if len(associated_proteins) > 1:
            to_remove_interactions.extend([(protein, kinase) for protein in associated_proteins[1:]])

to_remove_df = pd.DataFrame(to_remove_interactions, columns=['Uniprotid', 'kinase_sequence'])

to_remove_df.to_csv('valid_interactions_to_remove.csv', index=False)

#
# interaction_data = pd.read_csv('/mnt/data/filtered_unique.csv')
# to_remove_data = pd.read_csv('/mnt/data/interactions_to_remove.csv')
#
# # Step 1: Remove duplicates from both datasets based on 'Uniprotid' and 'kinase'
# interaction_data_cleaned = interaction_data.drop_duplicates(subset=['Uniprotid', 'kinase'])
# to_remove_data_cleaned = to_remove_data.drop_duplicates(subset=['Uniprotid', 'kinase'])
#
# # Step 2: Perform the anti-join to subtract the interactions to remove from the main dataset
# filtered_interaction_data = pd.merge(
#     interaction_data_cleaned,
#     to_remove_data_cleaned,
#     how='left',
#     on=['Uniprotid', 'kinase'],
#     indicator=True
# )
#
# # Step 3: Keep only the rows that are not in the to_remove_data_cleaned
# filtered_interaction_data = filtered_interaction_data[filtered_interaction_data['_merge'] == 'left_only']
#
# # Step 4: Drop the merge indicator column
# filtered_interaction_data = filtered_interaction_data.drop(columns=['_merge'])
#
# # Step 5: Save the cleaned and filtered data or display it
# filtered_interaction_data.to_csv('filtered_interaction_data.csv', index=False)