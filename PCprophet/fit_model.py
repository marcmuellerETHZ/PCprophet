import sys
import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

import PCprophet.io_ as io

def validate_inputs(X, y):
    if np.any(np.isnan(X)):
        print("Feature matrix contains NaN values.")
    if np.any(np.isinf(X)):
        print("Feature matrix contains infinity values.")
    if np.any(np.abs(X) > 1e10):
        print("Feature matrix contains excessively large values.")
    if np.any(np.isnan(y)):
        print("Label array contains NaN values.")
    if np.any(np.isinf(y)):
        print("Label array contains infinity values.")

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

def check_correlation(features_df, features):
    """
    Compute and display the correlation matrix for the given features.
    """
    correlation_matrix = features_df[features].corr()
    print("\nFeature Correlation Matrix:")
    print(correlation_matrix)
    correlation_matrix.to_csv("correlation_matrix.csv", sep="\t")
    return correlation_matrix



def check_vif(features_df, features):
    """
    Compute and display Variance Inflation Factor (VIF) for the given features.
    """
    X = features_df[features]
    X = sm.add_constant(X)  # Add constant for VIF computation
    vif_data = pd.DataFrame()
    vif_data["Feature"] = ["Intercept"] + features
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    print("\nVariance Inflation Factor (VIF):")
    print(vif_data)
    vif_data.to_csv("vif_data.csv", sep="\t", index=False)
    return vif_data


def fit_logistic_model(features_df_label, feature):
    X = features_df_label[[feature]].values
    y = features_df_label['Label'].values

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ground_truth_pos = y_test.sum()/len(y_test)

    validate_inputs(X_train, y_train)

    #model = LogisticRegression()
    model = LogisticRegression(penalty='none', class_weight='balanced', fit_intercept=True, solver='newton-cg')
    model.fit(X_train, y_train)

    y_scores = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, y_scores)
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

def fit_multi_logistic_model(features_df_label, features):
    """
    Fit a multiple logistic regression model using all specified features and output model summary.
    """
    X = features_df_label[features].values
    y = features_df_label['Label'].values

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ground_truth_pos = y_test.sum() / len(y_test)

    validate_inputs(X_train, y_train)

    model = LogisticRegression(
        penalty='none', class_weight='balanced', fit_intercept=True, solver='newton-cg'
    )
    model.fit(X_train, y_train)

    # Extract model parameters
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Coefficient': coefficients
    })

    # Predict probabilities and calculate performance metrics
    y_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, y_scores)
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

    # Print and return model summary
    print("\nModel Summary:")
    print(f"Intercept: {intercept}")
    print("Coefficients:")
    print(feature_importance)

    return roc_df, pr_df, auc_df, feature_importance, intercept



def runner(tmp_folder, db, features):
    """
    Main runner for logistic regression, including both individual feature regressions 
    and multi-feature regression.
    """
    features_df_path = os.path.join(tmp_folder, 'pairwise_features.txt')
    features_df = pd.read_csv(features_df_path, sep="\t")

    # Generate labels using database
    ppi_dict = db_to_dict(pd.read_csv(db, sep="\t"))
    features_df_label = add_ppi(features_df, ppi_dict)
    features_df_label['Label'] = features_df_label['db'].astype(int)

    # Ensure classifier performance folder exists
    perf_folder = os.path.join(tmp_folder, "classifier_performance_data")
    os.makedirs(perf_folder, exist_ok=True)

    # Step 1: Perform individual logistic regressions
    for feature in features:
        if feature not in features_df_label.columns:
            print(f"Warning: Feature '{feature}' not found in DataFrame. Skipping...")
            continue

        # Fit individual feature model
        roc_df, pr_df, auc_df = fit_logistic_model(features_df_label, feature)

        # Save results for each feature
        feature_folder = os.path.join(perf_folder, feature)
        os.makedirs(feature_folder, exist_ok=True)
        roc_df.to_csv(os.path.join(feature_folder, "roc_df.txt"), sep="\t", index=False)
        pr_df.to_csv(os.path.join(feature_folder, "pr_df.txt"), sep="\t", index=False)
        auc_df.to_csv(os.path.join(feature_folder, "auc_df.txt"), sep="\t", index=False)

    # Step 2: Perform multi-feature logistic regression
    roc_df, pr_df, auc_df, feature_importance, intercept = fit_multi_logistic_model(features_df_label, features)

    # Save results for multi-feature regression
    all_features_folder = os.path.join(perf_folder, "all_features_combined")
    os.makedirs(all_features_folder, exist_ok=True)

    roc_df.to_csv(os.path.join(all_features_folder, "roc_df.txt"), sep="\t", index=False)
    pr_df.to_csv(os.path.join(all_features_folder, "pr_df.txt"), sep="\t", index=False)
    auc_df.to_csv(os.path.join(all_features_folder, "auc_df.txt"), sep="\t", index=False)
    feature_importance.to_csv(os.path.join(all_features_folder, "feature_importance.txt"), sep="\t", index=False)

    # Save model intercept
    with open(os.path.join(all_features_folder, "intercept.txt"), "w") as f:
        f.write(f"Intercept: {intercept}\n")

    return True