import re
import sys
import os
import json
import numpy as np
import scipy.signal as signal
import pandas as pd
import random
import time
import matplotlib.pyplot as plt

from dask import dataframe as dd
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import uniform_filter

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
    """
    return pd.DataFrame(list(combinations(prot_dict.keys(), 2)), columns=['ProteinA', 'ProteinB'])

def calc_inv_euclidean_dist(elution_a, elution_b):
    diff = np.array(elution_a) - np.array(elution_b)
    sq_diff = diff**2
    sum_sq_diff = sq_diff.sum()
    eucl_dist = sum_sq_diff**(1/2)
    inv_eucl_dist = 1/eucl_dist
    return inv_eucl_dist

# I think there is a pitfall here: in smoothing, NAs are replaced with near-0 noise 
# Then, two measurements with lots of NAs will have high correlation, because both have lots of 0's
# This is why here, only raw profiles should be used

def sliding_window_correlation(a, b, metric, W=10):
    """
    vectorized correlation between pairs vectors with sliding window
    """
    cor = []

    aa = np.array(a)
    bb = np.array(b)
    
    am = uniform_filter(aa.astype(float), W)
    bm = uniform_filter(bb.astype(float), W)

    amc = am[W // 2 : -W // 2 + 1]
    bmc = bm[W // 2 : -W // 2 + 1]

    da = aa[:, None] - amc
    db = bb[:, None] - bmc

    # Get sliding mask of valid windows
    m, n = da.shape
    mask1 = np.arange(m)[:, None] >= np.arange(n)
    mask2 = np.arange(m)[:, None] < np.arange(n) + W
    mask = mask1 & mask2
    dam = da * mask
    dbm = db * mask

    ssAs = np.einsum("ij,ij->j", dam, dam)
    ssBs = np.einsum("ij,ij->j", dbm, dbm)
    D = np.einsum("ij,ij->j", dam, dbm)
    # add np.nan to reach 72
    cor.append(np.hstack((D / np.sqrt(ssAs * ssBs), np.zeros(9) + np.nan)))

    return_cor = metric(cor)

    return return_cor



def gen_feat(row, prot_dict, prot_dict_smooth, features):
    """
    Compute specified features for a pair of proteins.
    """
    prot_a, prot_b = row['ProteinA'], row['ProteinB']
    raw_a, raw_b = prot_dict.get(prot_a), prot_dict.get(prot_b)
    smooth_a, smooth_b = prot_dict_smooth.get(prot_a), prot_dict_smooth.get(prot_b)

    # Initialize results with protein pair
    results = {'ProteinA': prot_a, 'ProteinB': prot_b}
    if raw_a is not None and raw_b is not None:
        for feature in features:
            if feature == 'correlation_raw':
                results[feature] = np.corrcoef(raw_a, raw_b)[0, 1]
            elif feature == 'correlation_smooth':
                results[feature] = np.corrcoef(smooth_a, smooth_b)[0, 1]
            elif feature == 'euclidean_distance_smooth':
                results[feature] = calc_inv_euclidean_dist(smooth_a, smooth_b)
            elif feature == 'max_sliding_window_correlation_raw':
                results[feature] = sliding_window_correlation(raw_a, raw_b, np.nanmax)
            elif feature == 'mean_sliding_window_correlation_raw':
                results[feature] = sliding_window_correlation(raw_a, raw_b, np.nanmean)
            else:
                results[feature] = np.nan
    else:
        for feature in features:
            results[feature] = np.nan
    return results

def allbyall_feat(prot_dict, features, npartitions):
    """
    Wrapper to compute all-by-all pairwise features, logging time for each feature.
    """
    # Generate smoothed profiles
    prot_dict_smooth = clean_prot_dict(prot_dict)

    # Generate all protein pairs
    pairs = generate_combinations(prot_dict)

    # Partition the dataframe
    ddf = dd.from_pandas(pairs, npartitions=npartitions)

    # Initialize an empty results dataframe
    results_df = pairs.copy()

    # Compute features sequentially
    for feature in features:
        start_time = time.time()

        # Compute the feature in parallel across partitions
        feature_results = ddf.map_partitions(
            lambda df: df.apply(
                lambda row: gen_feat(row, prot_dict, prot_dict_smooth, [feature]),
                axis=1
            )
        ).compute(scheduler="processes")

        # Extract the computed feature values
        feature_values = [result[feature] for result in feature_results.tolist()]
        
        # Add the feature values to the results dataframe
        results_df[feature] = feature_values

        # Log the time taken for this feature
        end_time = time.time()
        print(f"Calculation of {feature} took {end_time - start_time:.2f} seconds.")

    return results_df

def runner(infile, tmp_folder, npartitions, features):


    os.makedirs(tmp_folder, exist_ok=True)

    npartitions = max(1, int(npartitions))

    prot_dict = io.read_txt(infile)

    pairwise_features = allbyall_feat(prot_dict=prot_dict, npartitions=npartitions, features=features)

    # os.pardir --> quick fix to get files from sample-specific subfolder in tmp to tmp folder
    # WONT WORK WITH >1 SAMPLES!!

    parent_folder = os.path.abspath(os.path.join(tmp_folder, os.pardir))
    pairwise_features_path = os.path.join(parent_folder, 'pairwise_features.txt')

    pairwise_features.to_csv(pairwise_features_path, sep="\t", index=False)

    return True