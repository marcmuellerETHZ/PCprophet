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

def corr_ROC(tmp_fold, output_folder, plot_name="ROC_Curve.png"):

    path_roc = os.path.join(tmp_fold, "ROC_curve.txt")
    roc_df = pd.read_csv(path_roc, sep="\t")

    path_auc = os.path.join(tmp_fold, "AUCs.txt")
    auc_df = pd.read_csv(path_auc, sep="\t")
    roc_auc = auc_df[auc_df['curve']=='ROC']['AUC'].iloc[0]

    plt.figure(figsize=(8, 6))
    plt.plot(roc_df['fpr'], roc_df['tpr'], color='blue', label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Save the plot
    save_path = os.path.join(output_folder, plot_name)
    plt.savefig(save_path, dpi=300)
    plt.close()

def corr_PR(tmp_fold, output_folder, plot_name="PR_Curve.png"):

    path_pr = os.path.join(tmp_fold, "PR_curve.txt")
    pr_df = pd.read_csv(path_pr, sep="\t")

    path_auc = os.path.join(tmp_fold, "AUCs.txt")
    auc_df = pd.read_csv(path_auc, sep="\t")
    pr_auc = auc_df[auc_df['curve']=='PR']['AUC'].iloc[0]

    path_corr = os.path.join(tmp_fold, "pairwise_correlation.txt")
    corr_df = pd.read_csv(path_corr, sep="\t")
    gr_truth_proportion = corr_df['db'].sum()/corr_df.shape[0]

    plt.figure(figsize=(8, 6))
    plt.plot(pr_df['recall'], pr_df['precision'], color='blue', label=f'AUC = {pr_auc:.3f}')
    plt.plot([0, 1], [gr_truth_proportion, gr_truth_proportion], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('precision-recall curve')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Save the plot
    save_path = os.path.join(output_folder, plot_name)
    plt.savefig(save_path, dpi=300)
    plt.close()


def runner(tmp_fold, out_fold, sid):

    outf = os.path.join(out_fold, "Plots")
    if not os.path.isdir(outf):
        os.makedirs(outf)

    corr_ROC(tmp_fold, outf)
    corr_PR(tmp_fold, outf)
