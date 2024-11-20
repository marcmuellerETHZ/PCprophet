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
from scipy.ndimage import uniform_filter1d

import PCprophet.io_ as io

np.seterr(all="ignore")
# silence the division by 0 in the correlation calc
mute = np.testing.suppress_warnings()
mute.filter(RuntimeWarning)
mute.filter(module=np.ma.core)

def clean_profile(chromatogram, impute_NA=True, smooth=True, smooth_width=4, noise_floor=0.001):
    """
    Clean an elution profile by imputing missing values, adding noise, and smoothing.
    
    Parameters:
    - chromatogram: np.ndarray, the elution profile (1D array).
    - impute_NA: bool, if True, impute missing/zero values using neighboring averages.
    - smooth: bool, if True, apply a moving average filter.
    - smooth_width: int, width of the moving average filter.
    - noise_floor: float, the maximum value for near-zero random noise.
    
    Returns:
    - cleaned: np.ndarray, the cleaned elution profile.
    """
    cleaned = np.copy(chromatogram)
    
    # Impute missing or zero values using mean of neighbors
    if impute_NA:
        for i in range(1, len(cleaned) - 1):
            if cleaned[i] == 0 or np.isnan(cleaned[i]):
                cleaned[i] = (cleaned[i - 1] + cleaned[i + 1]) / 2.0
    
    # Replace remaining missing or zero values with near-zero noise
    mask = (cleaned == 0) | np.isnan(cleaned)
    cleaned[mask] = np.random.uniform(0, noise_floor, size=mask.sum())
    
    # Apply moving average smoothing
    if smooth:
        padded = np.pad(cleaned, pad_width=smooth_width, mode='constant', constant_values=0)
        smoothed = uniform_filter1d(padded, size=smooth_width, mode='constant')
        cleaned = smoothed[smooth_width:-smooth_width]
    
    return cleaned

@io.timeit
def clean_prot_dict(prot_dict, impute_NA=True, smooth=True, smooth_width=4, noise_floor=0.001):
    """
    Clean all elution profiles in a protein dictionary.
    
    Parameters:
    - prot_dict: dict, {protein: elution profile}.
    - impute_NA, smooth, smooth_width, noise_floor: cleaning parameters.
    
    Returns:
    - cleaned_dict: dict, {protein: cleaned elution profile}.
    """
    cleaned_dict = {}
    for protein, chromatogram in prot_dict.items():
        cleaned_dict[protein] = clean_profile(
            np.array(chromatogram),
            impute_NA=impute_NA,
            smooth=smooth,
            smooth_width=smooth_width,
            noise_floor=noise_floor
        )
    return cleaned_dict

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

    ground_truth_pos = y.sum()/len(y)

    model = LogisticRegression()
    model.fit(X, y)

    y_scores = model.predict_proba(X)[:, 1]

    fpr, tpr, _ = roc_curve(y, y_scores)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y, y_scores)
    pr_auc = auc(recall, precision)

    roc_df = pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr
    })

    pr_df = pd.DataFrame({
        "precision": precision,
        "recall": recall
    })

    auc_df = pd.DataFrame({
        "obj": ["ROC_AUC", "PR_AUC", "GT_POS"],
        "value": [roc_auc, pr_auc, ground_truth_pos]
    })

    return roc_df, pr_df, auc_df

# wrapper
@io.timeit
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

    roc_df, pr_df, auc_df = fit_logistic_model(pairwise_corr_db)

    return pairwise_corr_db, roc_df, pr_df, auc_df

def runner(infile, tmp_folder, npartitions, db):
    """
    Runner function to handle file paths, invoke the wrapper, and save results.

    Parameters:
    - infile: Path to the input file containing elution profiles.
    - tmp_folder: Directory to save the results.
    - npartitions: Number of partitions for parallel processing.
    
    Returns:
    - True: Indicates successful execution.
    """
    os.makedirs(tmp_folder, exist_ok=True)
    prot_dict = io.read_txt(infile)
    db_file = pd.read_csv(db, sep="\t")

    pairwise_corr, roc_df, pr_df, auc_df = allbyall_feat(prot_dict=prot_dict, npartitions=npartitions, db=db_file)

    prot_dict_smooth = clean_prot_dict(prot_dict)
    pairwise_corr_smooth, roc_df_smooth, pr_df_smooth, auc_df_smooth = allbyall_feat(prot_dict=prot_dict_smooth, npartitions=npartitions, db=db_file)

    # ..\\ --> quick fix to get files from sample-specific subfolder in tmp to tmp folder
    # WONT WORK WITH >1 SAMPLES!!

    parent_folder = os.path.abspath(os.path.join(tmp_folder, os.pardir))
    path_pairwise_corr = os.path.join(parent_folder, 'pairwise_correlation.txt')
    pairwise_corr.to_csv(path_pairwise_corr, index=False, sep="\t")

    path_roc = os.path.join(parent_folder, "ROC_curve.txt")
    path_pr = os.path.join(parent_folder, "PR_curve.txt")
    path_auc = os.path.join(parent_folder, "AUCs.txt")

    roc_df.to_csv(path_roc, sep="\t")
    pr_df.to_csv(path_pr, sep="\t")
    auc_df.to_csv(path_auc, sep="\t")

    path_pairwise_corr_smooth = os.path.join(parent_folder, 'pairwise_correlation_smooth.txt')
    pairwise_corr_smooth.to_csv(path_pairwise_corr_smooth, index=False, sep="\t")

    path_roc_smooth = os.path.join(parent_folder, "ROC_curve_smooth.txt")
    path_pr_smooth = os.path.join(parent_folder, "PR_curve_smooth.txt")
    path_auc_smooth = os.path.join(parent_folder, "AUCs_smooth.txt")

    roc_df_smooth.to_csv(path_roc_smooth, sep="\t")
    pr_df_smooth.to_csv(path_pr_smooth, sep="\t")
    auc_df_smooth.to_csv(path_auc_smooth, sep="\t")

    return True