import csv
import numpy as np
import pickle
import pandas as pd

# ranges = [(0, 100), (101, 200), (201, 300), (301, 400), (401, 500), (501, 600), (601, 700), (701, 800)] 
ranges = [(201, 300)]

def get_mask(seq,keys=["S","T"]):
    mask = np.zeros(len(seq))
    for index, s in enumerate(seq):
        if s in keys:
            mask[index] = 1

    return mask

def extract_peptides_per_sequence(id, sequence, mask, label, peplen):
    if peplen % 2 == 0:
        assert("peplen is a even value! please change it into a odd value")
    
    peptides = []
    mask_peps = []
    id_peps = []
    label_peps = []

    for p in range(len(sequence)):
        if mask[p] == 1: #effect site
            center = sequence[p]
            left =  sequence[max([p-int(peplen/2),0]):p]
            right = sequence[p+1:p+1+int(peplen/2)]
            peptide = left+center+right
            peptides.append(peptide)
           
            mask_pep = np.zeros(len(peptide), dtype=int)
            mask_pep[len(left)] = 1
            mask_peps.append(mask_pep)

            label_pep = np.zeros(len(peptide), dtype=int)
            label_pep[len(left)] = label
            label_peps.append(label_pep)
            id_peps.append(id+"_"+str(p))
    
    return id_peps,peptides,label_peps,mask_peps

def save_data_to_pickle(ids, peptides, labels, kinases, valid_mask, outputfile):
     with open(outputfile, 'wb') as f:
            pickle.dump({'ids':ids, 'sequences': peptides, \
                         'labels': labels, 'kinases': kinases, 'valid_mask': valid_mask}, f)

def get_kinase_domain(df):
    kinase_group_path = 'Kincat_Hsap.08.02.xls'
    df_kinase  = pd.read_excel(kinase_group_path);
    df_merged = pd.merge(df, df_kinase[['Name', 'Kinase Domain',"Protein"]], 
                         how='left', 
                         left_on='kinase', 
                         right_on='Name')
    
    df_merged = df_merged.drop(columns=['Name'])
    df_cleaned  =  df_merged[df_merged['Kinase Domain'].str.strip() != ''] 
    df_cleaned = df_cleaned.dropna(subset=['Kinase Domain'])
    df_cleaned.rename(columns={'Protein': 'kinase_sequence'}, inplace=True)
    return df_cleaned

def get_kinases():
    df = pd.read_csv('positive_data_train_cdhit_70.csv')
    unique_kinase = df['kinase'].unique()

    df_kinase = pd.DataFrame(unique_kinase, columns=['kinase'])
    df_merged = get_kinase_domain(df_kinase)
    df_merged.to_csv('kinases.csv', index=False)

def read_cdhit_clstrfile(clstrfilename):
    cls_dict = {}
    fp = open(clstrfilename)
    lines = fp.readlines()

    cls_id = ""
    seq = []
    for line in lines:
        if line.startswith('>'):
            if cls_id != "":
                cls_dict[cls_id] = seq
            seq = []
            cls_id = int(line.strip('\n').split()[1]) #  line.split('|')[1] all in > need to be id
        else:
            fasta_id = line.split('\t')[1].split()[1]
            fasta_id = fasta_id[:-3]
            seq.append(fasta_id)
    
    cls_dict[cls_id] = seq 
    return cls_dict

def get_data():
    
    df_kinases = pd.read_csv('kinases.csv')
    cls_dict = read_cdhit_clstrfile('positive_test_cdhit_0.4.clstr')

    df_positive_dataset = pd.read_csv('positive_data_test_cdhit_70.csv')
    df_positive_dataset_dict = {}

    for index, row in df_positive_dataset.iterrows():
        uniprotid = row['Uniprotid']
        sequence = row['Sequence']
        kinase_name = row['kinase']

        if uniprotid in df_positive_dataset_dict:
            df_positive_dataset_dict[uniprotid]['kinase'].append(kinase_name)
        else:
            df_positive_dataset_dict[uniprotid] = {
                'Sequence': sequence,
                'kinase': [kinase_name]
            }   
    
    df_list = []
    for key, sequence_list in cls_dict.items():
        uniprot_id = sequence_list[0].lstrip('>')
        sequence = df_positive_dataset_dict.get(uniprot_id)['Sequence']
        kinase_names = df_positive_dataset_dict.get(uniprot_id)['kinase']

        df = df_kinases.copy() 
        df['label'] = df['kinase'].apply(lambda x: 1 if x in kinase_names else 0)
        df['uniprot_id'] = uniprot_id
        df['Sequence'] = sequence
        # print(len(df))
        df_list.append(df)
        

    df_combined = pd.concat(df_list, axis=0, ignore_index=True)
    print(len(df_combined))
    df_combined.to_csv('top_k_with_label.csv', index=False) # totla 287559 rows 801*359

    # df_combined.drop(columns=['label'], inplace=True)
    # df_combined.drop(columns=['kinase'], inplace=True)
    # df_combined.drop(columns=['uniprot_id'], inplace=True)
    # df_combined.to_csv('top_k_without_label.csv', index=False)

    
def get_split_data():
    # each file contains row 28755
    # column name: kinase,Kinase Domain,kinase_sequence,label,uniprot_id,Sequence
    df = pd.read_csv('top_k_with_label.csv') 

    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    df_split = np.array_split(df_shuffled, 10)
    for i, subset in enumerate(df_split):
        df_t = subset.copy()
        df_t.drop(columns=['kinase_sequence'], inplace=True)
        df_t.to_csv(f'top_k_{i}.csv',index=False)  

    for i, subset in enumerate(df_split):
        # print(len(subset))
        subset.to_csv(f'top_k_{i}_with_label.csv',index=False)     
        subset.drop(columns=['label'], inplace=True)
        subset.drop(columns=['kinase'], inplace=True)
        subset.drop(columns=['uniprot_id'], inplace=True)  
        subset.to_csv(f'top_k_{i}_without_label.csv',index=False)     


def main(): 

    peplen =15
    for i in range(10): 
        print(f'top_k_{i}.csv')
        df = pd.read_csv(f'top_k_{i}.csv')  #kinase	Kinase Domain	label	uniprot_id	Sequence
        
        peptides = []
        masks = []
        labels = []
        ids = []
        kinases = []
        kinase_names = []

        for index, uniprot_id in df['uniprot_id'].items():
            row = df.loc[index]
            sequence = row['Sequence']

            mask = get_mask(sequence)
            label = row['label']

            id_peps, pep_per_protein, label_peps, mask_peps = extract_peptides_per_sequence(
                uniprot_id, sequence, mask, label, peplen
            )
                
            peptides.extend(pep_per_protein)
            masks.extend(mask_peps)
            labels.extend(label_peps)
            ids.extend(id_peps)
            kinase_names.extend([row['kinase']] * len(id_peps))
            kinases.extend([row['Kinase Domain']] * len(id_peps))
        
        df_output = pd.DataFrame({
            'ids': ids,
            'sequences': peptides,
            'labels': labels,
            'kinase_name': kinase_names,
            'kinases': kinases
            # 'masks': masks
            })

        df_output.to_csv(f'top_k_peptide_{i}.csv', index=False)
        save_data_to_pickle(ids, peptides, labels, kinases, masks, f'top_k_peptide_{i}.pkl')    

def get_last10_substrate(output_path):
    df = pd.read_csv('/cluster/pixstor/xudong-lab/yongfang/plm/kinase/top_k/dataset/top_k_with_updated_label.csv')
    
    last_rows_df = df.iloc[0:17950]
    # last_rows_df.to_csv('last_rows.csv', index=False)

    group_counts = last_rows_df.groupby('uniprot_id').size()
    print(group_counts)

    kinase_group_path = 'Kincat_Hsap.08.02.xls'
    df_kinase  = pd.read_excel(kinase_group_path);
    df_merged = pd.merge(last_rows_df, df_kinase[['Name', 'Group', 'Family', 'Subfamily']], 
                         how='left', 
                         left_on='kinase', 
                         right_on='Name')
    
    df_merged = df_merged.copy()
    df_merged['combined'] = df_merged[['Group', 'Family', 'Subfamily','Name']].fillna('').astype(str).agg('-'.join, axis=1)
    sorted_df = df_merged.sort_values(by='combined')

    with open(output_path, 'w') as fasta_file:
        for index, row in sorted_df.iterrows():
            fasta_file.write(f">{row['uniprot_id']} {row['kinase']} {row['combined']} {row['label']}\n{row['Sequence']}\n")
    

if __name__ == '__main__':
    # main()
    get_last10_substrate('last_50_substrates.fasta')

