import re
import sys
import os
import numpy as np
import scipy.signal as signal
import pandas as pd

from dask import dataframe as dd
from itertools import combinations

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

# wrapper
def allbyall_feat(infile, npartitions):
    """
    Wrapper to compute all-by-all pairwise features.
    
    Parameters:
    - infile: Path to the input file containing elution profiles.
    - npartitions: Number of partitions for parallel processing.
    
    Returns:
    - pairwise_results: DataFrame containing pairwise correlation results.
    """
    # Step 1: Load elution profiles
    prot_dict = io.read_txt(infile)  # Assume this returns {protein: [elution profile]}

    npartitions = max(1, int(npartitions))
    
    # Step 2: Compute all pairwise correlations
    print(f"Computing pairwise correlations using {npartitions} partitions...")
    pairwise_results = calc_feat_allbyall(prot_dict, npartitions=npartitions)
    
    return pairwise_results

def runner(infile, output_folder, npartitions):
    """
    Runner function to handle file paths, invoke the wrapper, and save results.

    Parameters:
    - infile: Path to the input file containing elution profiles.
    - output_folder: Directory to save the results.
    - npartitions: Number of partitions for parallel processing.
    
    Returns:
    - True: Indicates successful execution.
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Invoke the wrapper to calculate features
    pairwise_results = allbyall_feat(infile, npartitions)
    
    # Define output file path
    output_path = os.path.join(output_folder, 'pairwise_correlation.txt')
    
    # Save the results
    print(f"Saving pairwise correlation results to {output_path}...")
    pairwise_results.to_csv(output_path, index=False, sep="\t")
    
    return True