import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import joblib

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

def train_random_forest(features, features_df_label):
    # Define input features and labels
    feature_columns = [col for col in features if col in features_df_label.columns]
    X = features_df_label[feature_columns].values
    y = features_df_label['Label'].values

    ground_truth_pos = y.sum()/len(y)

    # Train random forest classifier
    print(f"Training random forest with features: {feature_columns}")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    y_scores = clf.predict_proba(X_test)[:, 1]

    # Calculate metrics
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = roc_auc_score(y_test, y_scores)
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)

    df_pr = pd.DataFrame({'precision': precision, 'recall': recall})
    df_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr})

    df_auc = pd.DataFrame({
        "obj": ["ROC_AUC", "PR_AUC", "GT_POS"],
        "value": [roc_auc, pr_auc, ground_truth_pos]
    })

    return df_pr, df_roc, df_auc


def runner(config, features):
    """
    Runner function to execute the random forest training.
    """
    
    db_file = config['GLOBAL']['db']
    ppi_dict = db_to_dict(pd.read_csv(db_file, sep="\t"))
    
    pairwise_features_file = os.path.join(config['GLOBAL']['temp'], "pairwise_features.txt")
    features_df = pd.read_csv(pairwise_features_file, sep="\t")
    features_df_label = add_ppi(features_df, ppi_dict)

    features_df_label['Label'] = features_df_label['db'].astype(int)

    df_pr, df_roc, df_auc = train_random_forest(features, features_df_label)

    output_folder = os.path.join(config['GLOBAL']['temp'], "random_forest_results")

    os.makedirs(output_folder, exist_ok=True)
    
    df_pr.to_csv(os.path.join(output_folder, 'pr_df.txt'), sep="\t", index=False)
    df_roc.to_csv(os.path.join(output_folder, 'roc_df.txt'), sep="\t", index=False)
    df_auc.to_csv(os.path.join(output_folder, 'auc_df.txt'), sep="\t", index=False)
    

    return True
