import os
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np

import PCprophet.stats_ as st
import PCprophet.io_ as io


def plot_roc_curve(roc_df, auc_df, output_folder, feature):
    """Generate and save the ROC curve for a specific feature."""
    roc_auc = auc_df.loc[auc_df['obj'] == 'ROC_AUC', 'value'].iloc[0]

    plt.figure(figsize=(8, 6))
    plt.plot(roc_df['fpr'], roc_df['tpr'], color='blue', label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {feature}')
    plt.legend(loc='lower right')
    plt.grid(True)
    save_path = os.path.join(output_folder, f"ROC_{feature}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_precision_recall_curve(pr_df, auc_df, output_folder, feature):
    """Generate and save the Precision-Recall curve for a specific feature."""
    pr_auc = auc_df.loc[auc_df['obj'] == 'PR_AUC', 'value'].iloc[0]
    ground_truth_pos = auc_df.loc[auc_df['obj'] == 'GT_POS', 'value'].iloc[0]

    plt.figure(figsize=(8, 6))
    plt.plot(pr_df['recall'], pr_df['precision'], color='blue', label=f'AUC = {pr_auc:.3f}')
    plt.plot([0, 1], [ground_truth_pos, ground_truth_pos], color='red', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {feature}')
    plt.legend(loc='upper right')
    plt.grid(True)
    save_path = os.path.join(output_folder, f"PR_{feature}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def runner(tmp_fold, out_fold, features):
    """
    Generate plots for all features.

    Parameters:
    - tmp_fold: Path to the temporary folder containing the classifier performance data.
    - out_fold: Path to the base output folder.
    - features: List of features to process.
    """
    # Create output directory for plots
    outf = os.path.join(out_fold, "Plots")
    if not os.path.isdir(outf):
        os.makedirs(outf)

    # Iterate over each feature and generate plots
    for feature in features:
        feature_folder = os.path.join(tmp_fold, "classifier_performance_data", feature)
        if not os.path.isdir(feature_folder):
            print(f"Feature folder not found for {feature}. Skipping...")
            continue

        # Paths for ROC, PR, and AUC data
        roc_path = os.path.join(feature_folder, "roc_df.txt")
        pr_path = os.path.join(feature_folder, "pr_df.txt")
        auc_path = os.path.join(feature_folder, "auc_df.txt")

        if not all(os.path.exists(path) for path in [roc_path, pr_path, auc_path]):
            print(f"Missing data files for {feature}. Skipping...")
            continue

        # Load data
        roc_df = pd.read_csv(roc_path, sep="\t")
        pr_df = pd.read_csv(pr_path, sep="\t")
        auc_df = pd.read_csv(auc_path, sep="\t")

        # Generate and save plots
        plot_roc_curve(roc_df, auc_df, outf, feature)
        plot_precision_recall_curve(pr_df, auc_df, outf, feature)
