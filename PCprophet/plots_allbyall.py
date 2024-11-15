import os
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np

import PCprophet.stats_ as st


def smart_makefold(path, folder):
    """ """
    pl_dir = os.path.join(path, folder)
    if not os.path.isdir(pl_dir):
        os.makedirs(pl_dir)
    return pl_dir

def runner(tmp_fold, out_fold, target_fdr, sid):
    """
    performs all plots using matplotlib
    plots
    1) combined fdr
    2) recall from db/positive
    3) Profile plots across all replicates
    4) network combined from all conditions
    """
    outf = os.path.join(out_fold, "Plots")
    if not os.path.isdir(outf):
        os.makedirs(outf)
