import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score, f1_score, matthews_corrcoef


def get_top_k_by_group(df: pd.DataFrame, group_col: str, value_col: str, k: int) -> pd.DataFrame:
    df_sorted = df.sort_values([group_col, value_col], ascending=[True, False])
    top_k_df = df_sorted.groupby(group_col).head(k)
    return top_k_df

def main():
    # Load DataFrame
    df = pd.read_csv('protein-kinase-pair/duolin-test/inference_results_top_K_updated_label.csv')

    # Step 1: Calculate kinase_predicted_probs
    df['kinase_predicted_probs'] = df['V1_kinase_connected'] / (df['V1_kinase_disconnected'] + df['V1_kinase_connected'])

    # Step 2: Get the top k rows within each group of 'uniprot_id' based on 'kinase_predicted_probs'
    k = 5
    top_k_df = get_top_k_by_group(df, 'uniprot_id', 'kinase_predicted_probs', k)

    # Step 3: Apply threshold on kinase_predicted_probs and compare with label
    top_k_df['prob_flat'] = top_k_df['kinase_predicted_probs'].apply(lambda x: 1 if x > 0.5 else 0)
    top_k_df['hit'] = top_k_df.apply(lambda row: 1 if row['label'] == row['prob_flat'] else 0, axis=1)

    # Step 4: Group by 'uniprot_id' and keep rows with max 'hit' value
    df_max_hit = top_k_df.loc[top_k_df.groupby('uniprot_id')['hit'].idxmax()]

    # Step 5: Calculate accuracy
    acc = (df_max_hit['hit'] == 1).sum() / len(df_max_hit)
    print(f"Accuracy: {acc}")

    # Additional Metrics for Imbalanced Data
    y_true = df_max_hit['label']
    y_pred = df_max_hit['prob_flat']

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, df_max_hit['kinase_predicted_probs'])
    average_precision = average_precision_score(y_true, df_max_hit['kinase_predicted_probs'])

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc}")
    print(f"ROC AUC: {auc}")
    print(f"Average Precision Score (PR AUC): {average_precision}")


if __name__ == '__main__':
  main()
