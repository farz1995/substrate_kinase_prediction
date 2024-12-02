import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, recall_score, roc_auc_score, average_precision_score, f1_score, matthews_corrcoef


def kinase_level_result():
    df_true = pd.read_csv('protein-kinase-pair/duolin-test/combined_with_label.csv')
    df_predict = pd.read_csv('protein-kinase-pair/duolin-test/inference_results_combined_data.csv')
    df = pd.concat([df_true, df_predict], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]    
    print(df.columns)

    result_dict = {}
    kinase_groups = {kinase: group for kinase, group in df.groupby('group')}

    for kinase, group_df in kinase_groups.items():
        print(kinase)

        # print(f"Class distribution in y_true: {dict(pd.Series(df_true['label']).value_counts())}")
        # print(f"Class distribution in y_kinase_predict: {dict(pd.Series(df_predict['kinase_prediction']).value_counts())}")
        # print(f"Class distribution in y_kinase_domain_predict: {dict(pd.Series(df_predict['kinase_domain_prediction']).value_counts())}")

        group_df['v1_kinase_probs'] = group_df['V1_new_model_kinase_connected'] / (group_df['V1_new_model_kinase_disconnected'] + group_df['V1_new_model_kinase_connected'])
        group_df['v1_kinase_domain_probs'] = group_df['V1_domain_new_model_connected'] / (group_df['V1_domain_new_model_disconnected'] + group_df['V1_domain_new_model_connected'])

        cut_off = 0.5
        predicted_kianse_flat = (group_df['v1_kinase_probs'] > cut_off).astype(int) 
        predicted_kianse_domain_flat = (group_df['v1_kinase_domain_probs'] > cut_off).astype(int) 
        
        precision = precision_score(group_df['label'], predicted_kianse_flat,zero_division=0)
        recall = recall_score(group_df['label'], predicted_kianse_flat)
        f1 = f1_score(group_df['label'], predicted_kianse_flat)
        acc = accuracy_score(group_df['label'], predicted_kianse_flat)
        mcc = matthews_corrcoef(group_df['label'], predicted_kianse_flat)
        avg_precision = average_precision_score(group_df['label'], predicted_kianse_flat)

        if len(group_df['label'].unique()) > 1:
            auc = roc_auc_score(group_df['label'], group_df['v1_kinase_probs'])
        else:
            auc = None
        
        metrics = {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "MCC": mcc,
            "acc": acc,
            "AUC": auc,
            "Average Precision": avg_precision
        }
        
        # Initialize dictionary for kinase if it doesn't exist
        if kinase not in result_dict:
            result_dict[kinase] = {}
        else:
            print(f'seen kinase: {kinase}')
        
        result_dict[kinase]['kinase_prediction'] = metrics

        precision = precision_score(group_df['label'], predicted_kianse_domain_flat, zero_division=0)
        recall = recall_score(group_df['label'], predicted_kianse_domain_flat)
        f1 = f1_score(group_df['label'], predicted_kianse_domain_flat)
        acc = accuracy_score(group_df['label'], predicted_kianse_domain_flat)
        mcc = matthews_corrcoef(group_df['label'], predicted_kianse_domain_flat)
        avg_precision = average_precision_score(group_df['label'], predicted_kianse_domain_flat)

        if len(group_df['label'].unique()) > 1:
            auc = roc_auc_score(group_df['label'], group_df['v1_kinase_domain_probs'])
        else:
            auc = None

        metrics = {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ACC": acc,
            "MCC": mcc,
            "AUC": auc,
            "Average Precision": avg_precision
        }

        result_dict[kinase]['kinase_domain_prediction'] = metrics

    with open('group_level_results.txt', 'w') as file:
        for key, value in result_dict.items():
            file.write(f"{key}: {value}\n")
    
    
if __name__ == '__main__':
    kinase_level_result()