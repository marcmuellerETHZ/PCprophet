import sys
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score
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


def add_ppi(pairs_df, ppi_dict):
    pairs_df['db'] = pairs_df.apply(
        lambda row: row['ProteinB'].upper() in ppi_dict.get(row['ProteinA'].upper(), set()),
        axis=1
    )
    return pairs_df

def train_individual_logistic_regression(X_train, y_train, X_test, y_test, ground_truth_pos, feature, output_folder):
    """
    Train a logistic regression model for a single feature and save results.
    """

    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    model = LogisticRegression(penalty='none', class_weight='balanced', fit_intercept=True, solver='newton-cg')
    model.fit(X_train, y_train)

    
    y_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)

    # Save results
    pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(os.path.join(output_folder, f"{feature}_roc_df.txt"), sep="\t", index=False)
    pd.DataFrame({'precision': precision, 'recall': recall}).to_csv(os.path.join(output_folder, f"{feature}_pr_df.txt"), sep="\t", index=False)
    pd.DataFrame({"obj": ["ROC_AUC", "PR_AUC", "GT_POS"], "value": [roc_auc, pr_auc, ground_truth_pos]}).to_csv(
        os.path.join(output_folder, f"{feature}_auc_df.txt"), sep="\t", index=False
    )


def train_combined_logistic_regression(X_train, y_train, X_test, y_test, ground_truth_pos, features, output_folder):
    """
    Train a logistic regression model using all features and save results.
    """
    model = LogisticRegression(penalty='none', class_weight='balanced', fit_intercept=True, solver='newton-cg')
    model.fit(X_train, y_train)
    
    y_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)

    # Save results
    pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(os.path.join(output_folder, "combined_logistic_roc_df.txt"), sep="\t", index=False)
    pd.DataFrame({'precision': precision, 'recall': recall}).to_csv(os.path.join(output_folder, "combined_logistic_pr_df.txt"), sep="\t", index=False)
    pd.DataFrame({"obj": ["ROC_AUC", "PR_AUC", "GT_POS"], "value": [roc_auc, pr_auc, ground_truth_pos]}).to_csv(
        os.path.join(output_folder, "combined_logistic_auc_df.txt"), sep="\t", index=False
    )


def train_random_forest(X_train, y_train, X_test, y_test, ground_truth_pos, features, output_folder):
    """
    Train a random forest classifier and save results.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(output_folder, 'random_forest_model.pkl'))

    no_gt_pos = y_test.sum()
    no_gt_pos_tot = y_test.sum() + y_train.sum()
    
    y_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = roc_auc_score(y_test, y_scores)
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)

    # Save results
    pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(os.path.join(output_folder, 'rf_roc_df.txt'), sep="\t", index=False)
    pd.DataFrame({'precision': precision, 'recall': recall}).to_csv(os.path.join(output_folder, 'rf_pr_df.txt'), sep="\t", index=False)
    pd.DataFrame({"obj": ["ROC_AUC", "PR_AUC", "GT_POS", "NO_GT_POS", "NO_GT_POS_TOT"], "value": [roc_auc, pr_auc, ground_truth_pos, no_gt_pos, no_gt_pos_tot]}).to_csv(
        os.path.join(output_folder, 'rf_auc_df.txt'), sep="\t", index=False
    )


def wrapper(features_df, db_file, features, output_folder):
    """
    Wrapper for training models with a consistent train/test split.
    """
    ppi_dict = db_to_dict(pd.read_csv(db_file, sep="\t"))
    features_df_label = add_ppi(features_df, ppi_dict)
    features_df_label['Label'] = features_df_label['db'].astype(int)

    # Train/test split
    X = features_df_label[features].values
    y = features_df_label['Label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ground_truth_pos = y_test.sum()/len(y_test)

    # Train individual logistic regressions
    for feature in features:
        train_individual_logistic_regression(
            X_train[:, features.index(feature)],
            y_train,
            X_test[:, features.index(feature)],
            y_test,
            ground_truth_pos,
            feature,
            output_folder
        )
    
    # Train combined logistic regression
    train_combined_logistic_regression(X_train, y_train, X_test, y_test, ground_truth_pos, features, output_folder)
    
    # Train random forest
    train_random_forest(X_train, y_train, X_test, y_test, ground_truth_pos, features, output_folder)


def runner(config, features):
    """
    Runner function to handle input/output operations.
    """
    db_file = config['GLOBAL']['db']
    pairwise_features_file = os.path.join(config['GLOBAL']['temp'], "pairwise_features.txt")
    features_df = pd.read_csv(pairwise_features_file, sep="\t")

    output_folder = os.path.join(config['GLOBAL']['temp'], "classifier_performance_data")
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Call the wrapper
    wrapper(features_df, db_file, features, output_folder)


