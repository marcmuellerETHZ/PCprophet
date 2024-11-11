import re
import sys
import os
import json
import numpy as np
import scipy.signal as signal
import pandas as pd
import random
import matplotlib.pyplot as plt

from dask import dataframe as dd
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve

import PCprophet.io_ as io

np.seterr(all="ignore")
# silence the division by 0 in the correlation calc
mute = np.testing.suppress_warnings()
mute.filter(RuntimeWarning)
mute.filter(module=np.ma.core)

def generate_combinations(prot_dict):
    """
    Generate all unique protein pairs from the provided protein dictionary.
    
    Parameters:
    - prot_dict: Dictionary of proteins and their elution profiles.

    Returns:
    - DataFrame: All unique protein pairs.
    """
    pairs = pd.DataFrame(list(combinations(prot_dict.keys(), 2)), columns=['ProteinA', 'ProteinB'])
    return pairs


def gen_feat(row, prot_dict):
    """
    Compute Pearson correlation for a pair of proteins.
    
    Parameters:
    - row: DataFrame row containing 'ProteinA' and 'ProteinB'.
    - prot_dict: Dictionary with full elution profiles for all proteins.

    Returns:
    - Dict: Protein pair and their correlation.
    """
    prot_a = row['ProteinA']
    prot_b = row['ProteinB']

    if prot_a in prot_dict and prot_b in prot_dict:
        # Extract profiles for the two proteins
        elution_a = prot_dict[prot_a]
        elution_b = prot_dict[prot_b]
        
        # Calculate Pearson correlation
        corr_value = np.corrcoef(elution_a, elution_b)[0, 1]
        
        return {'ProteinA': prot_a, 'ProteinB': prot_b, 'Correlation': corr_value}
    else:
        return {'ProteinA': prot_a, 'ProteinB': prot_b, 'Correlation': np.nan}

def process_slice(df, prot_dict):
    """
    Process a slice of the DataFrame to compute pairwise features.
    
    Parameters:
    - df: DataFrame containing pairs of proteins (ProteinA, ProteinB).
    - prot_dict: Dictionary with full elution profiles for all proteins.

    Returns:
    - DataFrame: Pairwise correlation results for each pair in the slice.
    """
    return df.apply(lambda row: gen_feat(row, prot_dict), axis=1)

def calc_feat_allbyall(prot_dict, npartitions=4):
    """
    Calculate pairwise features across all protein combinations using partitioning.
    
    Parameters:
    - prot_dict: Dictionary of proteins and their elution profiles.
    - npartitions: Number of partitions for parallel processing.
    
    Returns:
    - DataFrame: Pairwise feature results for all combinations.
    """
    # Generate all pairwise combinations
    pairs = generate_combinations(prot_dict)
    
    # Partition the pairwise DataFrame
    ddf = dd.from_pandas(pairs, npartitions=npartitions)
    
    # Apply feature calculation across partitions
    results = ddf.map_partitions(lambda df: process_slice(df, prot_dict)).compute(scheduler='processes')
    
    return pd.DataFrame(results.tolist())


def db_to_dict(db):
    ppi_dict = {}
    for gene_names in db['subunits(Gene name)']:
        if pd.notna(gene_names):
            genes = [gene.upper() for gene in gene_names.split(';')]
            for gene_a in genes:
                if gene_a not in ppi_dict:
                    ppi_dict[gene_a] = set()
                for gene_b in genes:
                    if gene_a != gene_b:
                        ppi_dict[gene_a].add(gene_b)
    return ppi_dict

# Step 2: Check if a pair exists in CORUM
def add_ppi(pairs_df, ppi_dict):
    pairs_df['db'] = pairs_df.apply(
        lambda row: row['ProteinB'].upper() in ppi_dict.get(row['ProteinA'].upper(), set()),
        axis=1
    )
    return pairs_df

def fit_logistic_model(pairwise_corr_df):
    X = pairwise_corr_df[['Correlation']].values
    y = pairwise_corr_df['Label'].values

    model = LogisticRegression()
    model.fit(X, y)

    y_scores = model.predict_proba(X)[:, 1]

    fpr, tpr, _ = roc_curve(y, y_scores)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y, y_scores)
    pr_auc = auc(recall, precision)

    performance_data = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "ROC AUC": roc_auc,
        "PR AUC": pr_auc
    }

    return performance_data

# wrapper
def allbyall_feat(prot_dict, npartitions, db):
    """
    Wrapper to compute all-by-all pairwise features.
    
    Parameters:
    - infile: Path to the input file containing elution profiles.
    - npartitions: Number of partitions for parallel processing.
    
    Returns:
    - pairwise_corr: DataFrame containing pairwise correlation results.
    """
    npartitions = max(1, int(npartitions))
    
    ppi_dict = db_to_dict(db)

    pairwise_corr = calc_feat_allbyall(prot_dict, npartitions=npartitions)

    pairwise_corr_db = add_ppi(pairwise_corr, ppi_dict)

    pairwise_corr_db['Label'] = pairwise_corr_db['db'].astype(int)

    performance_data = fit_logistic_model(pairwise_corr_db)

    return pairwise_corr_db

def runner(infile, output_folder, npartitions, db):
    """
    Runner function to handle file paths, invoke the wrapper, and save results.

    Parameters:
    - infile: Path to the input file containing elution profiles.
    - output_folder: Directory to save the results.
    - npartitions: Number of partitions for parallel processing.
    
    Returns:
    - True: Indicates successful execution.
    """
    os.makedirs(output_folder, exist_ok=True)
    prot_dict = io.read_txt(infile)
    db_file = pd.read_csv(db, sep="\t")

    pairwise_df = allbyall_feat(prot_dict=prot_dict, npartitions=npartitions, db=db_file)

    path_pairwise_df = os.path.join(output_folder, 'pairwise_correlation.txt')
    pairwise_df.to_csv(path_pairwise_df, index=False, sep="\t")
    
    return True