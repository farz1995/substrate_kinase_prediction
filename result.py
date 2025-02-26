import argparse
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, matthews_corrcoef,
    accuracy_score, average_precision_score, roc_auc_score
)


def load_data(true_labels_file, predicted_file):
    """Load true labels and predictions from CSV files."""
    df_true = pd.read_csv(true_labels_file)
    df_predict = pd.read_csv(predicted_file)
    return df_true, df_predict


def calculate_metrics(y_true, y_pred, y_probs):
    """Calculate evaluation metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_probs),
        "Average Precision": average_precision_score(y_true, y_probs)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction metrics")
    parser.add_argument('--true_labels', type=str, required=True, help="Path to true labels CSV file")
    parser.add_argument('--predictions', type=str, required=True, help="Path to predictions CSV file")
    args = parser.parse_args()

    df_true, df_predict = load_data(args.true_labels, args.predictions)

    # Calculate probabilities for predictions
    df_predict['predicted_probs'] = df_predict['kinase_connected'] / (
            df_predict['kinase_disconnected'] + df_predict['kinase_connected']
    )

    y_true = df_true['label']
    y_pred = (df_predict['predicted_probs'] > 0.5).astype(int)  # Default cutoff at 0.5
    y_probs = df_predict['predicted_probs']

    metrics = calculate_metrics(y_true, y_pred, y_probs)
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
