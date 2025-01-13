import os
import matplotlib.pyplot as plt
import pandas as pd


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
    plots_folder = os.path.join(out_fold, "Plots")
    os.makedirs(plots_folder, exist_ok=True)

    feature_folder = os.path.join(tmp_fold, "classifier_performance_data")

    # Iterate over individual feature results
    for feature in features:
        roc_path = os.path.join(feature_folder, f"{feature}_roc_df.txt")
        pr_path = os.path.join(feature_folder, f"{feature}_pr_df.txt")
        auc_path = os.path.join(feature_folder, f"{feature}_auc_df.txt")

        if not all(os.path.exists(path) for path in [roc_path, pr_path, auc_path]):
            print(f"Missing data files for {feature}. Skipping...")
            continue

        # Load data
        roc_df = pd.read_csv(roc_path, sep="\t")
        pr_df = pd.read_csv(pr_path, sep="\t")
        auc_df = pd.read_csv(auc_path, sep="\t")

        # Generate plots
        plot_roc_curve(roc_df, auc_df, plots_folder, feature)
        plot_precision_recall_curve(pr_df, auc_df, plots_folder, feature)

    # Process random forest results
    roc_path = os.path.join(feature_folder, "rf_roc_df.txt")
    pr_path = os.path.join(feature_folder, "rf_pr_df.txt")
    auc_path = os.path.join(feature_folder, "rf_auc_df.txt")

    if all(os.path.exists(path) for path in [roc_path, pr_path, auc_path]):
        roc_df = pd.read_csv(roc_path, sep="\t")
        pr_df = pd.read_csv(pr_path, sep="\t")
        auc_df = pd.read_csv(auc_path, sep="\t")

        plot_roc_curve(roc_df, auc_df, plots_folder, "random_forest")
        plot_precision_recall_curve(pr_df, auc_df, plots_folder, "random_forest")

    # Process combined logistic regression results
    roc_path = os.path.join(feature_folder, "combined_logistic_roc_df.txt")
    pr_path = os.path.join(feature_folder, "combined_logistic_pr_df.txt")
    auc_path = os.path.join(feature_folder, "combined_logistic_auc_df.txt")

    if all(os.path.exists(path) for path in [roc_path, pr_path, auc_path]):
        roc_df = pd.read_csv(roc_path, sep="\t")
        pr_df = pd.read_csv(pr_path, sep="\t")
        auc_df = pd.read_csv(auc_path, sep="\t")

        plot_roc_curve(roc_df, auc_df, plots_folder, "combined_logistic")
        plot_precision_recall_curve(pr_df, auc_df, plots_folder, "combined_logistic")

    feature = "random_forest_seperate_split"

    rf_folder = os.path.join(tmp_fold, "random_forest_results")
    roc_path = os.path.join(rf_folder, "roc_df.txt")
    pr_path = os.path.join(rf_folder, "pr_df.txt")
    auc_path = os.path.join(rf_folder, "auc_df.txt")

    roc_df = pd.read_csv(roc_path, sep="\t")
    pr_df = pd.read_csv(pr_path, sep="\t")
    auc_df = pd.read_csv(auc_path, sep="\t")

    # Generate and save plots
    plot_roc_curve(roc_df, auc_df, plots_folder, feature)
    plot_precision_recall_curve(pr_df, auc_df, plots_folder, feature)
