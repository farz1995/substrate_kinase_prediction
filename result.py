import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score, average_precision_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def main():

    df_true = pd.read_csv('protein-kinase-pair/duolin-test/combined_with_label.csv')
    df_predict = pd.read_csv('protein-kinase-pair/duolin-test/inference_results_combined_data.csv')

    print(f"Class distribution in y_true: {dict(pd.Series(df_true['label']).value_counts())}")
    print(f"Class distribution in y_kinase_predict: {dict(pd.Series(df_predict['kinase_prediction']).value_counts())}")
    print(f"Class distribution in y_kinase_domain_predict: {dict(pd.Series(df_predict['kinase_domain_prediction']).value_counts())}")

    # Calculating metrics
    precision = precision_score(df_true['label'], df_predict['kinase_prediction'])
    recall = recall_score(df_true['label'], df_predict['kinase_prediction'])
    f1 = f1_score(df_true['label'], df_predict['kinase_prediction'])
    mcc = matthews_corrcoef(df_true['label'], df_predict['kinase_prediction'])
    avg_precision = average_precision_score(df_true['label'], df_predict['kinase_prediction'])

    df_predict['kinase_predicted_probs'] = df_predict['kinase_connected'] / (df_predict['kinase_disconnected'] + df_predict['kinase_connected'])
    auc = roc_auc_score(df_true['label'], df_predict['kinase_predicted_probs'])

    metrics = {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc,
        "AUC": auc,
        "Average Precision": avg_precision
    }
    print('kinase_prediction')
    print(metrics)

    precision = precision_score(df_true['label'], df_predict['V2_kinase_domain_connected'])
    recall = recall_score(df_true['label'], df_predict['kinase_domain_prediction'])
    f1 = f1_score(df_true['label'], df_predict['kinase_domain_prediction'])
    mcc = matthews_corrcoef(df_true['label'], df_predict['kinase_domain_prediction'])
    avg_precision = average_precision_score(df_true['label'], df_predict['kinase_domain_prediction'])
    acc = accuracy_score(df_true['label'], df_predict['kinase_domain_prediction'])

    df_predict['kinase_domain_predicted_probs'] = df_predict['V2_kinase_domain_connected'] / (df_predict['V2_kinase_domain_disconnected'] + df_predict['V2_kinase_domain_connected'])
    auc = roc_auc_score(df_true['label'], df_predict['kinase_domain_predicted_probs'])

    metrics = {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc,
        "AUC": auc,
        "Average Precision": avg_precision
    }
    print('kinase_domain_prediction')
    print(metrics)

def valid_combined_inference_results():
    df = pd.read_csv('protein-kinase-pair/duolin-test/inference_results_combined_data.csv')
    # df = pd.read_csv('zero_shot/zero_shot_data_random_hidden/random/2024-11-16__17-29-04/inference_results.csv')
    # df = pd.read_csv('zero_shot/zero-shot-evaluation/first_model/2024-11-23__01-25-21/inference_results.csv')
    df_true = pd.read_csv('protein-kinase-pair/duolin-test/combined_with_label.csv')

    # df['kinase_predicted_probs'] = df['V2_kinase_domain_connected'] / (df['V2_kinase_domain_disconnected'] + df['V2_kinase_domain_connected']) #train on domain
    df['kinase_predicted_probs'] = df['V1_domain_new_model_connected'] / (df['V1_domain_new_model_disconnected'] + df['V1_domain_new_model_connected']) # no train on domain, no redundant model
    # df['kinase_predicted_probs'] = df['connected'] / (df['disconnected'] + df['connected']) # zero-shot
    predicted_probs = df['kinase_predicted_probs']

    cut_off = 0.5
    predicted_labels_flat = (predicted_probs > cut_off).astype(int) 

    precision = precision_score(df['label'], predicted_labels_flat)
    recall = recall_score(df['label'], predicted_labels_flat)
    f1 = f1_score(df['label'], predicted_labels_flat)
    mcc = matthews_corrcoef(df['label'], predicted_labels_flat)
    acc = accuracy_score(df['label'], predicted_labels_flat)
    
    print(f"Class distribution in y_true: {dict(pd.Series(df['label']).value_counts())}")
    print(f"Class distribution in y_predict: {dict(pd.Series(predicted_labels_flat).value_counts())}")

    avg_precision = average_precision_score(df['label'], predicted_probs)
    auc = roc_auc_score(df['label'], predicted_probs)
    #
    metrics = {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc,
        "AUC": auc,
        "Average Precision": avg_precision,
        "ACC": acc,
    }

    print(metrics)
    # # Calculate ROC curve
    # fpr, tpr, thresholds = roc_curve(df['label'], predicted_probs)
    #
    # # Plot ROC curve
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
    # plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    # plt.xlabel('False Positive Rate (FPR)')
    # plt.ylabel('True Positive Rate (TPR)')
    # plt.title('ROC Curve')
    # plt.legend(loc='lower right')
    # plt.grid()
    # plt.tight_layout()
    # plt.show()


import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, matthews_corrcoef,
    accuracy_score, average_precision_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt


def calculate_metrics(y_true, predicted_probs, cutoff=0.3):
    """
    Calculate metrics and ROC data for given true labels and predicted probabilities.

    Parameters:
        y_true (pd.Series): True binary labels.
        predicted_probs (pd.Series): Predicted probabilities for the positive class.
        cutoff (float): Threshold to classify probabilities into binary predictions.

    Returns:
        dict: Calculated metrics (Precision, Recall, F1 Score, MCC, AUC, etc.).
        tuple: ROC curve data (fpr, tpr, thresholds).
    """
    # Binary predictions based on cutoff
    predicted_labels_flat = (predicted_probs > cutoff).astype(int)

    # Calculate metrics
    metrics = {
        "Precision": precision_score(y_true, predicted_labels_flat),
        "Recall": recall_score(y_true, predicted_labels_flat),
        "F1 Score": f1_score(y_true, predicted_labels_flat),
        "MCC": matthews_corrcoef(y_true, predicted_labels_flat),
        "AUC": roc_auc_score(y_true, predicted_probs),
        "Average Precision": average_precision_score(y_true, predicted_probs),
        "ACC": accuracy_score(y_true, predicted_labels_flat),
        "Class Distribution (True)": dict(pd.Series(y_true).value_counts()),
        "Class Distribution (Predicted)": dict(pd.Series(predicted_labels_flat).value_counts()),
    }

    # Calculate ROC curve data
    fpr, tpr, thresholds = roc_curve(y_true, predicted_probs)

    return metrics, (fpr, tpr, thresholds)


def plot_roc_curves(roc_data):
    """
    Plot ROC curves for multiple configurations.

    Parameters:
        roc_data (dict): A dictionary where keys are configuration names and values are
                         tuples (fpr, tpr, auc).
    """
    plt.figure(figsize=(10, 8))
    for config_name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f'{config_name} (AUC = {auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
  # main()
  valid_combined_inference_results()
  # Main workflow
  # df_combined = pd.read_csv('protein-kinase-pair/duolin-test/inference_results_combined_data.csv')
  # df_GPS = pd.read_csv('protein-kinase-pair/duolin-test/GPS6.0_combined_data_result.csv')
  # df_phosphormer = pd.read_csv('protein-kinase-pair/duolin-test/PhoformerST_combined_data_result.csv')
  # df_true = pd.read_csv('protein-kinase-pair/duolin-test/combined_with_label.csv')

  # Define different models/configurations and their probability calculation
  # configurations = {
  #     "substrate-domain (train on domain)": df_combined['V2_kinase_domain_connected'] / (
  #             df_combined['V2_kinase_domain_disconnected'] + df_combined['V2_kinase_domain_connected']),
  #     "substrate-domain (No train on domain)": df_combined['V1_domain_new_model_connected'] / (
  #             df_combined['V1_domain_new_model_disconnected'] + df_combined['V1_domain_new_model_connected']),
  #     "substrate-kinase": df_combined['V1_new_model_kinase_connected'] / (df_combined['V1_new_model_kinase_disconnected'] + df_combined['V1_new_model_kinase_connected']),
  #     "GPS 6.0": df_GPS['predicted_prob'] ,
  #     "Phosphormer": df_phosphormer['predicted_prob'],
  # }

  # cut_off = 0.5
  # roc_data = {}

  # Iterate through configurations to calculate metrics and ROC data
  # for config_name, predicted_probs in configurations.items():
  #     if config_name == "GPS 6.0":
  #         y_true = df_GPS['binary_labels']
  #     elif config_name == "Phosphormer":
  #         y_true = df_phosphormer['binary_labels']
  #     else:
  #         y_true = df_combined['label']
  #
  #     metrics, (fpr, tpr, thresholds) = calculate_metrics(y_true, predicted_probs, cutoff=cut_off)
  #     roc_data[config_name] = (fpr, tpr, metrics["AUC"])  # Store ROC data for plotting
  #
  #     print(f"Metrics for {config_name}:")
  #     print(metrics)

  # # Plot ROC curves
  # plot_roc_curves(roc_data)



# combined data
# Class distribution in y_true: {0: 1172, 1: 1047}
# Class distribution in y_kinase_predict: {1: 1329, 0: 890}
# Class distribution in y_kinase_domain_predict: {0: 1271, 1: 948}
# kinase_prediction
# {'Precision': 0.7185854025583145, 'Recall': 0.9121298949379179, 'F1 Score': 0.8038720538720537, 'MCC': 0.6040139817864867, 'AUC': 0.8624992258068723, 'Average Precision': 0.6969033449093331}
# kinase_domain_prediction
# {'Precision': 0.6445147679324894, 'Recall': 0.5835721107927412, 'F1 Score': 0.612531328320802, 'MCC': 0.29874119856143855, 'AUC': 0.7283576348481441, 'Average Precision': 0.5726057466689654}

# valid_combined_inference_result
# Class distribution in y_true: {0: 1700, 1: 1677}
# Class distribution in y_predict: {0: 1852, 1: 1525}
# {'Precision': 0.6845901639344262, 'Recall': 0.6225402504472272, 'F1 Score': 0.6520924422236102, 'MCC': 0.3411945428293445, 'AUC': 0.7253961906766284, 'Average Precision': 0.6979717805794052}