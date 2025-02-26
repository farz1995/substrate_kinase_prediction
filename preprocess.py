import pandas as pd
import random
import argparse


def load_interaction_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: {file_path} not found.")


def load_cluster_data(file_path):
    try:
        with open(file_path, 'r') as f:
            clusters = {}
            current_cluster = None

            for line in f:
                if line.startswith('>Cluster'):
                    current_cluster = line.strip().split(' ')[1]
                    clusters[current_cluster] = []
                elif '>' in line:
                    uniprot_id = line.split('>')[1].split('...')[0]
                    clusters[current_cluster].append(uniprot_id)
        return clusters
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: {file_path} not found.")


def filter_clusters(clusters):
    return {k: v for k, v in clusters.items() if len(set(v)) > 1}


def get_protein_kinase_pairs(interaction_data):
    return interaction_data.groupby('Uniprotid')['kinase_sequence'].apply(list).to_dict()


def cdhit_remove(filtered_clusters, protein_kinase_pairs):
    to_remove_interactions = []
    for cluster, proteins in filtered_clusters.items():
        unique_proteins = set(proteins)
        kinase_interactions = {}

        for protein in unique_proteins:
            if protein in protein_kinase_pairs:
                kinases = protein_kinase_pairs[protein]
                for kinase in kinases:
                    kinase_interactions.setdefault(kinase, []).append(protein)

        for kinase, associated_proteins in kinase_interactions.items():
            if len(associated_proteins) > 1:
                to_remove_interactions.extend([(protein, kinase) for protein in associated_proteins[1:]])

    return pd.DataFrame(to_remove_interactions, columns=['Uniprotid', 'kinase_sequence'])


def filter_interaction_data(interaction_data, to_remove_df):
    interaction_data_cleaned = interaction_data.drop_duplicates(subset=['Uniprotid', 'kinase_sequence'])
    to_remove_data_cleaned = to_remove_df.drop_duplicates(subset=['Uniprotid', 'kinase_sequence'])

    filtered_interaction_data = pd.merge(
        interaction_data_cleaned,
        to_remove_data_cleaned,
        how='left',
        on=['Uniprotid', 'kinase_sequence'],
        indicator=True
    )

    filtered_interaction_data = filtered_interaction_data[filtered_interaction_data['_merge'] == 'left_only']
    filtered_interaction_data = filtered_interaction_data.drop(columns=['_merge'])
    return filtered_interaction_data


def generate_negative_data(filtered_unique_file, to_remove_file, output_file, num_samples=1700):
    random.seed(1)
    filtered_unique_df = pd.read_csv(filtered_unique_file)
    to_remove_df = pd.read_csv(to_remove_file)

    filtered_unique_df_filtered = filtered_unique_df[
        ~filtered_unique_df[['Uniprotid', 'kinase_sequence']].apply(tuple, axis=1).isin(
            to_remove_df[['Uniprotid', 'kinase_sequence']].apply(tuple, axis=1)
        )
    ]

    unique_uniprotids = filtered_unique_df_filtered['Uniprotid'].unique()
    unique_kinases = filtered_unique_df_filtered['kinase_sequence'].unique()

    negative_samples = []
    while len(negative_samples) < num_samples:
        uniprotid = random.choice(unique_uniprotids)
        kinase = random.choice(unique_kinases)
        if not ((filtered_unique_df_filtered['Uniprotid'] == uniprotid) &
                (filtered_unique_df_filtered['kinase_sequence'] == kinase)).any():
            negative_samples.append((uniprotid, kinase))

    negative_data_df = pd.DataFrame(negative_samples, columns=['Uniprotid', 'kinase_sequence'])
    negative_data_full = []

    for _, row in negative_data_df.iterrows():
        uniprotid = row['Uniprotid']
        kinase = row['kinase_sequence']
        sequence = \
        filtered_unique_df_filtered.loc[filtered_unique_df_filtered['Uniprotid'] == uniprotid, 'Sequence'].values[0]
        kinase_sequence = filtered_unique_df_filtered.loc[
            filtered_unique_df_filtered['kinase_sequence'] == kinase, 'kinase_sequence'].values[0]
        negative_data_full.append([uniprotid, sequence, kinase_sequence])

    negative_data_full_df = pd.DataFrame(negative_data_full, columns=['Uniprotid', 'Sequence', 'kinase_sequence'])
    negative_data_full_df.to_csv(output_file, index=False)


def main():
    parser = argparse.ArgumentParser(description="Process protein interaction data")
    parser.add_argument('--input_file', type=str, required=True, help="Path to input CSV file")
    parser.add_argument('--cluster_file', type=str, help="Path to cdhit output cluster file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to output filtered CSV file")
    args = parser.parse_args()

    interaction_data = load_interaction_data(args.interaction_file)
    clusters = load_cluster_data(args.cluster_file)
    filtered_clusters = filter_clusters(clusters)
    protein_kinase_pairs = get_protein_kinase_pairs(interaction_data)
    to_remove_df = cdhit_remove(filtered_clusters, protein_kinase_pairs)

    filtered_interaction_data = filter_interaction_data(interaction_data, to_remove_df)
    filtered_interaction_data.to_csv(args.output_file, index=False)

    print("Processing completed. Filtered interaction data saved.")


if __name__ == "__main__":
    main()
