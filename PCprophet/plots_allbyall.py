import os
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np

import PCprophet.stats_ as st
import PCprophet.io_ as io


def smart_makefold(path, folder):
    """ """
    pl_dir = os.path.join(path, folder)
    if not os.path.isdir(pl_dir):
        os.makedirs(pl_dir)
    return pl_dir

def ROC_plot(tmp_fold, output_folder, plot_name, curve_data, auc_data):

    path_roc = os.path.join(tmp_fold, curve_data)
    roc_df = pd.read_csv(path_roc, sep="\t")

    path_auc = os.path.join(tmp_fold, auc_data)
    auc_df = pd.read_csv(path_auc, sep="\t")
    roc_auc = auc_df[auc_df['obj']=='ROC_AUC']['value'].iloc[0]

    plt.figure(figsize=(8, 6))
    plt.plot(roc_df['fpr'], roc_df['tpr'], color='blue', label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{plot_name}')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Save the plot
    save_path = os.path.join(output_folder, f'{plot_name}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

def precision_recall_plot(tmp_fold, output_folder, plot_name, curve_data, auc_data):

    path_pr = os.path.join(tmp_fold, curve_data)
    pr_df = pd.read_csv(path_pr, sep="\t")

    path_auc = os.path.join(tmp_fold, auc_data)
    auc_df = pd.read_csv(path_auc, sep="\t")
    pr_auc = auc_df[auc_df['obj']=='PR_AUC']['value'].iloc[0]
    ground_truth_pos = auc_df[auc_df['obj']=='GT_POS']['value'].iloc[0]

    plt.figure(figsize=(8, 6))
    plt.plot(pr_df['recall'], pr_df['precision'], color='blue', label=f'AUC = {pr_auc:.3f}')
    plt.plot([0, 1], [ground_truth_pos, ground_truth_pos], color='red', linestyle='--')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('precision-recall curve')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Save the plot
    save_path = os.path.join(output_folder, f'{plot_name}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()


def runner(tmp_fold, out_fold, sid):

    outf = os.path.join(out_fold, "Plots")
    if not os.path.isdir(outf):
        os.makedirs(outf)

    ROC_plot(tmp_fold, outf, "ROC_raw", "ROC_curve.txt", "AUCs.txt")
    precision_recall_plot(tmp_fold, outf, "PR_raw", "PR_curve.txt", "AUCs.txt")

    ROC_plot(tmp_fold, outf, "ROC_smooth", "ROC_curve_smooth.txt", "AUCs_smooth.txt")
    precision_recall_plot(tmp_fold, outf, "PR_smooth", "PR_curve_smooth.txt", "AUCs_smooth.txt")

