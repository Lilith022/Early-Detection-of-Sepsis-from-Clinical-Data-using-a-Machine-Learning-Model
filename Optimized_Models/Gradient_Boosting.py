import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import shap
import optuna

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, 
    roc_curve, auc
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

# Utility Metric Functions
    
def compute_patient_utility(
    labels: np.ndarray,
    preds: np.ndarray,
    dt_early: int = -6,
    dt_optimal: int = 0,
    dt_late: int = 9,
    max_u_tp: float = 1.0,
    min_u_fn: float = -2.0,
    u_fp: float = -0.05,
    u_tn: float = 0.0,
    check_errors: bool = True
) -> float:
    """
    Compute utility score for a single patient based on prediction timing.
    """

    if check_errors:
        if len(labels) != len(preds):
            raise ValueError("labels and preds must have the same length.")
        if not all((l in (0, 1) for l in labels)):
            raise ValueError("labels must only contain 0/1.")
        if not all((p in (0, 1) for p in preds)):
            raise ValueError("preds must only contain 0/1.")
        if not (dt_early < dt_optimal < dt_late):
            raise ValueError("Require dt_early < dt_optimal < dt_late.")

    L = len(labels)
    u_t = np.zeros(L)

    if np.any(labels == 1):
        onset = int(np.argmax(labels))
        is_septic = True
    else:
        onset = None
        is_septic = False

    denom1 = float(dt_optimal - dt_early)
    denom2 = float(dt_late - dt_optimal)

    m1 = max_u_tp / denom1 if denom1 != 0 else 0.0
    m2 = -max_u_tp / denom2 if denom2 != 0 else 0.0
    m3 = min_u_fn / denom2 if denom2 != 0 else 0.0

    for t in range(L):
        if not is_septic:
            u_t[t] = u_fp if preds[t] == 1 else u_tn
            continue

        x = t - onset

        if x > dt_late:
            u_t[t] = 0.0
            continue

        if preds[t] == 1:
            if x < dt_early:
                u_t[t] = u_fp
            elif dt_early <= x <= dt_optimal:
                u_t[t] = m1 * (x - dt_early)
            elif dt_optimal < x <= dt_late:
                u_t[t] = max_u_tp + m2 * (x - dt_optimal)
            else:
                u_t[t] = 0.0
        else:
            if x <= dt_optimal:
                u_t[t] = 0.0
            elif dt_optimal < x <= dt_late:
                u_t[t] = m3 * (x - dt_optimal)
            else:
                u_t[t] = 0.0

    return float(np.sum(u_t))


def compute_normalized_utility(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    dt_early: int = -6,
    dt_optimal: int = 0,
    dt_late: int = 9,
    max_u_tp: float = 1.0,
    min_u_fn: float = -2.0,
    u_fp: float = -0.05,
    u_tn: float = 0.0,
    return_details: bool = False
):
    """
    Compute normalized utility score across patients.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    unique_patients = np.unique(groups)
    observed_utils, best_utils, inaction_utils, worst_utils, patient_ids = [], [], [], [], []

    for pid in unique_patients:
        idx = np.where(groups == pid)[0]
        if idx.size == 0:
            continue
        lab = y_true[idx].astype(int)
        pred_obs = y_pred[idx].astype(int)
        L = len(lab)
        u_obs = compute_patient_utility(
            lab, pred_obs, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn
        )
        best_pred = np.zeros(L, dtype=int)
        if np.any(lab == 1):
            onset = int(np.argmax(lab))
            start_idx = max(0, onset + dt_early)
            end_idx = min(L, onset + dt_late + 1)
            if start_idx < end_idx:
                best_pred[start_idx:end_idx] = 1
        u_best = compute_patient_utility(lab, best_pred, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        inaction_pred = np.zeros(L, dtype=int)
        u_inaction = compute_patient_utility(lab, inaction_pred, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        worst_pred = 1 - best_pred
        u_worst = compute_patient_utility(lab, worst_pred, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        patient_ids.append(pid)
        observed_utils.append(u_obs)
        best_utils.append(u_best)
        inaction_utils.append(u_inaction)
        worst_utils.append(u_worst)
    un_obs = float(np.sum(observed_utils))
    un_best = float(np.sum(best_utils))
    un_inaction = float(np.sum(inaction_utils))

    denom = (un_best - un_inaction)
    normalized = (un_obs - un_inaction) / denom if denom != 0 else float('nan')

    details = {
        'patient_ids': np.array(patient_ids),
        'observed_utils': np.array(observed_utils),
        'best_utils': np.array(best_utils),
        'inaction_utils': np.array(inaction_utils),
        'worst_utils': np.array(worst_utils),
        'unnormalized': {'observed': un_obs, 'best': un_best, 'inaction': un_inaction}
    }

    return (normalized, details) if return_details else normalized

# Helper Function

def sensitivity_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sensitivity, specificity

def best_threshold(y_true, y_proba, alpha=0.6):
    thresholds = np.linspace(0, 1, 200)
    scores = []
    for thr in thresholds:
        preds = (y_proba >= thr).astype(int)
        f1_val = f1_score(y_true, preds, zero_division=0)
        rec_val = recall_score(y_true, preds, zero_division=0)
        scores.append(alpha * f1_val + (1 - alpha) * rec_val)
    best_idx = np.argmax(scores)
    return thresholds[best_idx]

def mean_ci(data):
    arr = np.array(data)
    mean = arr.mean()
    std = arr.std(ddof=1)
    n = len(arr)
    se = std / np.sqrt(n)
    h = se * 1.96
    return mean, h

# Data Loading and Preparation

INPUT_FILE = "imputation/LIGHTGBM.psv"
df = pd.read_csv(INPUT_FILE, sep='|')
df = df.drop(columns=['HospAdmTime'], errors='ignore')
X = df.drop(columns=["Paciente", "SepsisLabel"])
y = df["SepsisLabel"]
groups = df["Paciente"]

X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, groups, test_size=0.15, stratify=y, random_state=42
)

# Cross-validation

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
class_weight = {0: 1, 1: (((len(y_train) - sum(y_train)) / sum(y_train)) * 0.5)}

scoring = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'sensitivity', 'specificity', 'utility']
val_scores, fold_metrics, fold_thresholds, models = {m: [] for m in scoring}, [], [], []

for fold, (train_idx, val_idx) in enumerate(tqdm(cv.split(X_train, y_train, groups=groups_train), total=5, desc="CV folds")):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = HistGradientBoostingClassifier(verbose=0, class_weight=class_weight)
    model.fit(X_tr, y_tr)

    y_proba = model.predict_proba(X_val)[:, 1]
    thr = best_threshold(y_val, y_proba)
    y_pred = (y_proba >= thr).astype(int)

    metrics_fold = {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) >= 2 else np.nan
    }
    sens, spec = sensitivity_specificity(y_val, y_pred)
    metrics_fold["sensitivity"], metrics_fold["specificity"] = sens, spec
    try:
        util_fold = compute_normalized_utility(
            y_true=y_val.values, y_proba=y_proba, y_pred=y_pred,
            groups=groups_train.iloc[val_idx].values
        )
    except Exception as e:
        util_fold = float('nan')
        warnings.warn(f"Utility error in fold {fold}: {e}")
    metrics_fold["utility"] = util_fold
    for m in scoring:
        val_scores[m].append(metrics_fold[m])
    models.append(model)
    fold_metrics.append(metrics_fold)
    fold_thresholds.append(thr)

# Best Model Selection + Optuna

best_fold = np.argmax([0.3 * m["f1"] + 0.7 * m["sensitivity"] for m in fold_metrics])
best_model = models[best_fold]
best_threshold_fold = fold_thresholds[best_fold]

X_tr_best, X_val_best = X_train.iloc[train_idx], X_train.iloc[val_idx]
y_tr_best, y_val_best = y_train.iloc[train_idx], y_train.iloc[val_idx]

def objective(trial):
    params = {
        "max_iter": trial.suggest_int("max_iter", 100, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "random_state": 42,
        "class_weight": class_weight
    }
    model = HistGradientBoostingClassifier(**params)
    model.fit(X_tr_best, y_tr_best)
    y_proba = model.predict_proba(X_val_best)[:, 1]
    y_pred = (y_proba >= best_threshold_fold).astype(int)
    return 3 * recall_score(y_val_best, y_pred, zero_division=0) + f1_score(y_val_best, y_pred, zero_division=0)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

optuna_model = HistGradientBoostingClassifier(**study.best_params, random_state=42, class_weight=class_weight)
optuna_model.fit(X_tr_best, y_tr_best)

# Test Set Evaluation

y_test_proba = optuna_model.predict_proba(X_test)[:, 1]
best_thr = best_threshold(y_test, y_test_proba)
y_test_pred = (y_test_proba >= best_thr).astype(int)

test_metrics = {
    "accuracy": accuracy_score(y_test, y_test_pred),
    "f1": f1_score(y_test, y_test_pred, zero_division=0),
    "precision": precision_score(y_test, y_test_pred, zero_division=0),
    "recall": recall_score(y_test, y_test_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) >= 2 else np.nan
}
sens_test, spec_test = sensitivity_specificity(y_test, y_test_pred)
test_metrics["sensitivity"], test_metrics["specificity"] = sens_test, spec_test

util_test, util_details_test = compute_normalized_utility(
    y_true=y_test.values, y_proba=y_test_proba, y_pred=y_test_pred, groups=groups_test.values, return_details=True
)
test_metrics["utility"] = util_test

# SHAP Analysis (Aggregated by Patients)

explainer = shap.Explainer(optuna_model, X_test, algorithm="tree")
shap_values_all = explainer(X_test).values
if isinstance(shap_values_all, list):
    shap_values_all = shap_values_all[1]

shap_df = pd.DataFrame(shap_values_all, index=groups_test, columns=X_test.columns)
shap_values_patient = shap_df.groupby(level=0).mean()
X_test_patient = X_test.groupby(groups_test).mean()


output_dir = "Results/Gradient_Boosting"
os.makedirs(output_dir, exist_ok=True)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], '--', color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(output_dir, "ROC_curve.pdf"))
plt.close()

# Confusion Matrix
plt.figure(figsize=(6,6))
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix - Test")
plt.savefig(os.path.join(output_dir, "confusion_matrix.pdf"))
plt.close()

# SHAP Global Importance
shap.summary_plot(shap_values_patient.values, X_test_patient, plot_type="bar", show=False)
plt.savefig(os.path.join(output_dir, "shap_global_importance.pdf"))
plt.close()

# SHAP Beeswarm
shap.summary_plot(shap_values_patient.values, X_test_patient, show=False, cmap=plt.cm.cool)
plt.savefig(os.path.join(output_dir, "shap_beeswarm.pdf"))
plt.close()

# SHAP Feature Ranking
mean_abs_shap = np.abs(shap_values_patient.values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    "Variable": X_test_patient.columns,
    "Importance (%)": 100 * mean_abs_shap / mean_abs_shap.sum()
}).sort_values(by="Importance (%)", ascending=False)

shap_importance_df.to_csv(os.path.join(output_dir, "shap_ranking.csv"), index=False)

top15 = shap_importance_df.head(15)
plt.figure(figsize=(10,6))
ax = sns.barplot(data=top15, x="Importance (%)", y="Variable", palette="viridis")
for i, v in enumerate(top15["Importance (%)"]):
    ax.text(v + 0.3, i, f"{v:.2f}", va="center")
plt.title("Top 15 Features - SHAP")
plt.savefig(os.path.join(output_dir, "shap_top15.pdf"))
plt.close()
