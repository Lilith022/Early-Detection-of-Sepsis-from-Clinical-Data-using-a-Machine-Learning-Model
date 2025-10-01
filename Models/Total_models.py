"""
    Comparative analysis of machine learning models for sepsis detection.
    Features group-aware cross-validation, multiple evaluation metrics, and statistical
    confidence intervals for reliable model selection in clinical settings.
    """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from scipy.stats import norm
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier    

# Configuration
INPUT_FILE = "imputation/LightGBM.psv"
OUTPUT_EXCEL = "Results/Results_total_modelsodels.xlsx"
N_FOLDS = 5
RANDOM_STATE = 42
Z = 1.96 

# Data Loading and Preparation
df = pd.read_csv(INPUT_FILE, sep='|')
X = df.drop(columns=["Patient", "SepsisLabel"]).to_numpy()
y = df["SepsisLabel"].to_numpy()
groups = df["Patient"].to_numpy()

# Models definition
models = {
    "CatBoost": CatBoostClassifier(verbose=0, random_state=RANDOM_STATE),
    "ExtraTrees": ExtraTreesClassifier(random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
    "LogisticRegression": LogisticRegression(random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "NaiveBayes": GaussianNB(),
    "XGBoost": XGBClassifier(random_state=RANDOM_STATE),
    "LightGBM": LGBMClassifier(random_state=RANDOM_STATE),
    "MLP": MLPClassifier(random_state=RANDOM_STATE)
}
# Store evaluation results
metrics_results = []
auc_results = []
confusion_matrix = {}

# Cross-Validation Setup
kf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


for name, model in models.items():
    print(f"\nEvaluating model: {name}")
    fold_auc_roc = []
    fold_auc_pr = []
    fold_acc = []
    fold_prec = []
    fold_rec = []
    fold_f1 = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y, groups=groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val)[:, 1]
        else:
            y_proba = model.decision_function(X_val)
        fold_auc_roc.append(roc_auc_score(y_val, y_proba))
        precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_proba)
        fold_auc_pr.append(auc(recall_vals, precision_vals))
        fold_acc.append(accuracy_score(y_val, y_pred))
        fold_prec.append(precision_score(y_val, y_pred))
        fold_rec.append(recall_score(y_val, y_pred))
        fold_f1.append(f1_score(y_val, y_pred))
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)

    mean_auc_roc = np.mean(fold_auc_roc)
    std_auc_roc = np.std(fold_auc_roc)
    margin_error_roc = Z * (std_auc_roc / np.sqrt(N_FOLDS))
    ci_lower_roc = mean_auc_roc - margin_error_roc
    ci_upper_roc = mean_auc_roc + margin_error_roc

    mean_auc_pr = np.mean(fold_auc_pr)
    std_auc_pr = np.std(fold_auc_pr)
    margin_error_pr = Z * (std_auc_pr / np.sqrt(N_FOLDS))
    ci_lower_pr = mean_auc_pr - margin_error_pr
    ci_upper_pr = mean_auc_pr + margin_error_pr

    auc_results.append({
        "model": name,
        "AUC_ROC": f"{mean_auc_roc:.3f} [{ci_lower_roc:.3f}, {ci_upper_roc:.3f}]",
        "AUC_PR": f"{mean_auc_pr:.3f} [{ci_lower_pr:.3f}, {ci_upper_pr:.3f}]"
    })
# Global Metrics
    cm_global = confusion_matrix(all_y_true, all_y_pred)
    confusion_matrix[name] = cm_global
    acc_global = accuracy_score(all_y_true, all_y_pred)
    prec_global = precision_score(all_y_true, all_y_pred)
    rec_global = recall_score(all_y_true, all_y_pred)
    f1_global = f1_score(all_y_true, all_y_pred)
    auc_roc_global = roc_auc_score(all_y_true, all_y_proba)
    precision_vals, recall_vals, _ = precision_recall_curve(all_y_true, all_y_proba)
    auc_pr_global = auc(recall_vals, precision_vals)
        
    metrics_results.append({
        "model": name,
        "Accuracy": acc_global,
        "Precision": prec_global,
        "Recall": rec_global,
        "F1": f1_global,
        "AUC_ROC": auc_roc_global,
        "AUC_PR": auc_pr_global,
        "AUC_ROC_CI95": f"[{ci_lower_roc:.3f}, {ci_upper_roc:.3f}]",
        "AUC_PR_CI95": f"[{ci_lower_pr:.3f}, {ci_upper_pr:.3f}]"
    })
    # Plot ROC Curve
    print(f"  AUC-ROC: {mean_auc_roc:.3f} [{ci_lower_roc:.3f}, {ci_upper_roc:.3f}]")
    print(f"  AUC-PR: {mean_auc_pr:.3f} [{ci_lower_pr:.3f}, {ci_upper_pr:.3f}]")
    fpr, tpr, _ = roc_curve(all_y_true, all_y_proba)
    auc_roc = roc_auc_score(all_y_true, all_y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_roc:.2f})")

with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
    pd.DataFrame(metrics_results).to_excel(writer, sheet_name="Metrics", index=False)
    pd.DataFrame(auc_results).to_excel(writer, sheet_name="AUC_intervals", index=False)
    for name, cm in confusion_matrix.items():
        cm_df = pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["Real 0", "Real 1"])
        cm_df.to_excel(writer, sheet_name=f"CM_{name[:20]}")
# Final ROC Comparison Plot
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curves - Model Comparison")
plt.legend()
plt.grid(True)
plt.show()
# Final metrics DataFrame
resultados_df = pd.DataFrame(metrics_results)
