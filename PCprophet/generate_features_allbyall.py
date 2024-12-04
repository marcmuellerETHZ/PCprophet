import re
import sys
import os
import numpy as np
import scipy.signal as signal
import pandas as pd

from dask import dataframe as dd
from itertools import combinations
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import uniform_filter
from scipy.optimize import curve_fit

import PCprophet.io_ as io

np.seterr(all="ignore")
# silence the division by 0 in the correlation calc
mute = np.testing.suppress_warnings()
mute.filter(RuntimeWarning)
mute.filter(module=np.ma.core)

def generate_combinations(prot_dict):
    """
    Generate all unique protein pairs from the provided protein dictionary.
    """
    return pd.DataFrame(list(combinations(prot_dict.keys(), 2)), columns=['ProteinA', 'ProteinB'])

def min_max_scale_prot_dict(prot_dict):
    """
    Apply min/max scaling to the co-elution data in prot_dict.
    
    Parameters:
    - prot_dict: Dictionary where keys are protein names and values are lists/arrays
                 of co-elution intensities.
    
    Returns:
    - scaled_prot_dict: Dictionary with scaled intensities.
    """
    prot_dict_scaled = {}
    
    for protein, intensities in prot_dict.items():
        # Convert intensities to a NumPy array
        intensities = np.array(intensities)
        
        # Min/max scaling
        min_val = np.min(intensities)
        max_val = np.max(intensities)
        if max_val - min_val > 0:  # Avoid division by zero
            scaled_intensities = (intensities - min_val) / (max_val - min_val)
        else:
            # If all values are the same, set them to 0 (or any consistent value)
            scaled_intensities = np.zeros_like(intensities)
        
        # Add scaled values to the new dictionary
        prot_dict_scaled[protein] = scaled_intensities
    
    return prot_dict_scaled

def remove_outliers(prot_dict, threshold):
    """
    Removes all intensity values from prot_dict where the corresponding z_scores_ms < threshold.

    Parameters:
        prot_dict (dict): A dictionary where keys are gene names and values are lists of intensity values.
        threshold (float): The z-score threshold for identifying outliers.

    Returns:
        dict: A dictionary with outliers removed based on the threshold.
    """
    filtered_dict = {}
    
    for gene, intensities in prot_dict.items():
        intensities = np.array(intensities)
        
        # Padding the first and last values
        pad_start = np.concatenate(([intensities[0]], intensities))
        pad_end = np.concatenate((intensities, [intensities[-1]]))

        # Calculate the difference (pad_end - pad_start)
        diff = pad_end - pad_start

        # Combine squared differences of two neighboring points
        ms = np.array([diff[i] * diff[i+1] for i in range(len(diff) - 1)])

        # Calculate mean, standard deviation, and z-scores
        mean_ms = np.mean(ms)
        std_ms = np.std(ms)
        z_scores_ms = (ms - mean_ms) / std_ms

        # Create a mask where z_scores_ms < threshold
        mask = z_scores_ms < threshold

        # Replace outlier intensities with the average of the preceding and following values
        filtered_intensities = intensities.copy()
        filtered_intensities[mask] = np.nan
        
        # Store the imputed intensities in the new dictionary
        filtered_dict[gene] = filtered_intensities.tolist()
    
    return filtered_dict

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

    prot_dict_scaled = min_max_scale_prot_dict(prot_dict)

    cleaned_dict = {}
    for protein, chromatogram in prot_dict_scaled.items():
        cleaned_dict[protein] = clean_profile(
            np.array(chromatogram),
            impute_NA=impute_NA,
            smooth=smooth,
            smooth_width=smooth_width,
            noise_floor=noise_floor
        )
    return cleaned_dict



def make_initial_conditions(chromatogram, n_gaussians, method="guess", sigma_default=2, sigma_noise=0.5, mu_noise=1.5, A_noise=0.5):
    """
    Generate initial conditions for Gaussian fitting.
    """
    if method == "guess":
        # Identify local maxima
        peaks = (np.diff(np.sign(np.diff(chromatogram))) == -2).nonzero()[0] + 1
        peaks = np.append(peaks, [0, len(chromatogram) - 1])  # Include boundaries
        peaks = peaks[np.argsort(-chromatogram[peaks])]  # Sort by height
        
        # Generate initial conditions
        A = []
        mu = []
        sigma = []
        for i in range(n_gaussians):
            if i < len(peaks):
                A.append(chromatogram[peaks[i]] + np.random.uniform(-A_noise, A_noise))
                mu.append(peaks[i] + np.random.uniform(-mu_noise, mu_noise))
                sigma.append(sigma_default + np.random.uniform(-sigma_noise, sigma_noise))
            else:  # Fill with random values if not enough peaks
                A.append(np.random.uniform(0, max(chromatogram)))
                mu.append(np.random.uniform(0, len(chromatogram)))
                sigma.append(sigma_default + np.random.uniform(-sigma_noise, sigma_noise))
        return {"A": np.array(A), "mu": np.array(mu), "sigma": np.array(sigma)}
    elif method == "random":
        A = np.random.uniform(0, max(chromatogram), n_gaussians)
        mu = np.random.uniform(0, len(chromatogram), n_gaussians)
        sigma = sigma_default + np.random.uniform(-sigma_noise, sigma_noise, n_gaussians)
        return {"A": A, "mu": mu, "sigma": sigma}

def fit_curve(coefs, indices):
    """
    Compute the fitted curve from Gaussian coefficients.
    """
    A = coefs["A"]
    mu = coefs["mu"]
    sigma = coefs["sigma"]
    gaussians = len(A)
    return np.sum([A[i] * np.exp(-((indices - mu[i]) / sigma[i])**2) for i in range(gaussians)], axis=0)

def fit_gaussians(chromatogram, n_gaussians, max_iterations, min_R_squared, method,
                  filter_gaussians_center, filter_gaussians_height,
                  filter_gaussians_variance_min, filter_gaussians_variance_max):
    """
    Fit a mixture of Gaussians to a chromatogram.
    """
    indices = np.arange(len(chromatogram))
    best_R2 = -np.inf
    best_coefs = None
    
    for _ in range(max_iterations):
        # Generate initial conditions
        init = make_initial_conditions(chromatogram, n_gaussians, method)
        A, mu, sigma = init["A"], init["mu"], init["sigma"]
        
        # Define Gaussian mixture model
        def gaussian_model(x, *params):
            gaussians = len(params) // 3
            A, mu, sigma = np.split(np.array(params), 3)
            return np.sum([A[i] * np.exp(-((x - mu[i]) / sigma[i])**2) for i in range(gaussians)], axis=0)
        
        # Flatten initial parameters
        init_params = np.concatenate([A, mu, sigma])
        
        try:
            # Perform curve fitting
            popt, _ = curve_fit(gaussian_model, indices, chromatogram, p0=init_params, maxfev=5000)
            # Extract fitted coefficients
            coefs = {"A": popt[:n_gaussians], "mu": popt[n_gaussians:2*n_gaussians], "sigma": popt[2*n_gaussians:]}
            curve_fit_result = fit_curve(coefs, indices)
            
            # Calculate R-squared
            residual = chromatogram - curve_fit_result
            ss_res = np.sum(residual**2)
            ss_tot = np.sum((chromatogram - np.mean(chromatogram))**2)
            R2 = 1 - (ss_res / ss_tot)
            
            if R2 > best_R2:
                best_R2 = R2
                best_coefs = coefs
        except Exception:
            continue
    
    return {"R2": best_R2, "coefs": best_coefs, "fit_curve": fit_curve(best_coefs, indices) if best_coefs else None}

def gaussian_aicc(coefs, chromatogram):
    """
    Calculate the corrected Akaike Information Criterion (AICc).
    """
    n = len(chromatogram)  # Number of data points
    k = len(coefs["A"]) * 3  # Number of parameters (A, mu, sigma for each Gaussian)
    if k >= n - 1:
        return np.inf  # Avoid division by zero or invalid AICc when parameters exceed data points
    
    rss = np.sum((chromatogram - fit_curve(coefs, np.arange(len(chromatogram))))**2)
    aic = n * np.log(rss / n) + 2 * k
    return aic + (2 * k * (k + 1)) / (n - k - 1)

def gaussian_aic(coefs, chromatogram):
    """
    Calculate the Akaike Information Criterion (AIC).
    """
    n = len(chromatogram)  # Number of data points
    k = len(coefs["A"]) * 3  # Number of parameters (A, mu, sigma for each Gaussian)
    rss = np.sum((chromatogram - fit_curve(coefs, np.arange(len(chromatogram))))**2)
    return n * np.log(rss / n) + 2 * k

def gaussian_bic(coefs, chromatogram):
    """
    Calculate the Bayesian Information Criterion (BIC).
    """
    n = len(chromatogram)  # Number of data points
    k = len(coefs["A"]) * 3  # Number of parameters (A, mu, sigma for each Gaussian)
    rss = np.sum((chromatogram - fit_curve(coefs, np.arange(len(chromatogram))))**2)
    return n * np.log(rss / n) + k * np.log(n)

def choose_gaussians(chromatogram, points=None, max_gaussians=5, criterion="AICc",
                     max_iterations=10, min_R_squared=0.5, method="guess",
                     filter_gaussians_center=True, filter_gaussians_height=0.15,
                     filter_gaussians_variance_min=0.1, filter_gaussians_variance_max=50):
    """
    Fit mixtures of Gaussians to a chromatogram and select the best model using an information criterion.
    """
    # Adjust max_gaussians based on available data points
    if points is not None:
        max_gaussians = min(max_gaussians, points // 3)
    
    # Fit models with increasing numbers of Gaussians
    fits = []
    for n_gaussians in range(1, max_gaussians + 1):
        fit = fit_gaussians(chromatogram, n_gaussians, max_iterations, min_R_squared, method,
                            filter_gaussians_center, filter_gaussians_height,
                            filter_gaussians_variance_min, filter_gaussians_variance_max)
        fits.append(fit)
    
    # Remove models that failed to fit
    valid_fits = [fit for fit in fits if fit["coefs"] is not None]
    if not valid_fits:
        return None  # No valid fits
    
    # Calculate the chosen information criterion for each valid fit
    if criterion == "AICc":
        criteria = [gaussian_aicc(fit["coefs"], chromatogram) for fit in valid_fits]
    elif criterion == "AIC":
        criteria = [gaussian_aic(fit["coefs"], chromatogram) for fit in valid_fits]
    elif criterion == "BIC":
        criteria = [gaussian_bic(fit["coefs"], chromatogram) for fit in valid_fits]
    else:
        raise ValueError("Invalid criterion. Choose 'AICc', 'AIC', or 'BIC'.")
    
    # Select the model with the lowest criterion value
    best_fit_index = np.argmin(criteria)
    return valid_fits[best_fit_index]


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


def co_peak(a, b):
    """
    Calculate the absolute difference
    between the indices of the maximum values of two elution profiles.

    Parameters:
        a: The first elution profile.
        b: The second elution profile.

    Returns:
        int: The absolute difference between the indices of the maximum values.
    """
    # Find the indices of the maximum values for both profiles
    max_a = np.argmax(a)
    max_b = np.argmax(b)
    
    # Calculate the absolute difference
    return abs(max_a - max_b)


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
            elif feature == 'co_peak_smooth':
                results[feature] = co_peak(smooth_a, smooth_b)
            else:
                results[feature] = np.nan
    else:
        for feature in features:
            results[feature] = np.nan
    return results



# wrapper
def allbyall_feat(prot_dict, features, npartitions):
    """
    Wrapper to compute all-by-all pairwise features.
    """
    # Generate smoothened profiles
    prot_dict_filtered = remove_outliers(prot_dict, threshold=-7)
    prot_dict_scaled = clean_prot_dict(prot_dict_filtered, smooth=False)
    prot_dict_smooth_scaled = clean_prot_dict(prot_dict_filtered, smooth=True)
    
    for feature in features:
        print(feature)

    # Generate all protein pairs
    pairs = generate_combinations(prot_dict)

    # Partition and compute features
    ddf = dd.from_pandas(pairs, npartitions=npartitions)
    results = ddf.map_partitions(
        lambda df: df.apply(lambda row: gen_feat(row, prot_dict_scaled, prot_dict_smooth_scaled, features), axis=1)
    ).compute(scheduler='processes')

    results_df = pd.DataFrame(results.tolist())

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