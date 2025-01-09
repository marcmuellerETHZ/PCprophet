import sys
import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split

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

def fit_logistic_model(features_df_label, feature):
    X = features_df_label[[feature]].values
    y = features_df_label['Label'].values

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ground_truth_pos = y_test.sum()/len(y_test)

    validate_inputs(X_train, y_train)

    #model = LogisticRegression()
    model = LogisticRegression(penalty=None, class_weigh='balanced', fit_intercept=True, solver='newton-cholesky')
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

def runner(tmp_folder, db, features):
    """
    Main runner for model fitting and evaluation.

    Parameters:
    - tmp_folder: Path to temporary folder containing pairwise feature data.
    - output_folder: Base output folder.
    - db: Path to the database file.
    - features: List of features to process.
    """

    features_df_path = os.path.join(tmp_folder, 'pairwise_features.txt')
    features_df = pd.read_csv(features_df_path, sep="\t")

    ppi_dict = db_to_dict(pd.read_csv(db, sep="\t"))
    
    features_df_label = add_ppi(features_df, ppi_dict)

    features_df_label['Label'] = features_df_label['db'].astype(int)

    # Ensure classifier performance folder exists
    perf_folder = os.path.join(tmp_folder, "classifier_performance_data")
    os.makedirs(perf_folder, exist_ok=True)

    # Load database and process features
    ppi_dict = db_to_dict(pd.read_csv(db, sep="\t"))
    features_df_label = add_ppi(features_df, ppi_dict)
    features_df_label['Label'] = features_df_label['db'].astype(int)

    for feature in features:
        if feature not in features_df_label.columns:
            print(f"Warning: Feature '{feature}' not found in DataFrame. Skipping...")
            continue

        # Fit model and calculate performance metrics
        roc_df, pr_df, auc_df = fit_logistic_model(features_df_label, feature)

        # Create feature-specific folder
        feature_folder = os.path.join(perf_folder, feature)
        os.makedirs(feature_folder, exist_ok=True)

        # Save results
        roc_df.to_csv(os.path.join(feature_folder, "roc_df.txt"), sep="\t", index=False)
        pr_df.to_csv(os.path.join(feature_folder, "pr_df.txt"), sep="\t", index=False)
        auc_df.to_csv(os.path.join(feature_folder, "auc_df.txt"), sep="\t", index=False)

    return True

