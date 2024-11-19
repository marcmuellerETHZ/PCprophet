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

def corr_ROC(path_roc, output_folder, plot_name="ROC_Curve.png"):

    roc_df = pd.read_csv(path_roc, sep="\t")

    plt.figure(figsize=(8, 6))
    plt.plot(roc_df['fpr'], roc_df['tpr'], color='blue')
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


def runner(tmp_fold, out_fold, sid):

    outf = os.path.join(out_fold, "Plots")
    if not os.path.isdir(outf):
        os.makedirs(outf)

    path_roc = os.path.join(tmp_fold, "ROC_curve.txt")

    corr_ROC(path_roc, outf)
