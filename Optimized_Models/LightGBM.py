"""
Final pipeline for sepsis prediction using LightGBM. Includes patient-level utility metric,
cross-validation, Optuna hyperparameter optimization, and SHAP-based interpretability.

This code implements a comprehensive machine learning pipeline for sepsis prediction that:
1. Computes patient-specific utility metrics considering early/late detection trade-offs
2. Performs stratified group k-fold cross-validation to handle patient data leakage
3. Optimizes LightGBM hyperparameters using Optuna framework
4. Provides model interpretability through SHAP analysis
5. Generates comprehensive evaluation metrics and visualizations 
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import optuna
import shap
import os
import seaborn as sns
import warnings
import lightgbm as lgb

# Patient-level utility calculation
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
    """Compute patient-specific utility considering temporal detection windows."""
    # Input validation
    if check_errors:
        if len(labels) != len(preds):
            raise ValueError("labels and preds must have the same length.")
        if not all((l in (0, 1) for l in labels)):
            raise ValueError("labels must contain only 0/1.")
        if not all((p in (0, 1) for p in preds)):
            raise ValueError("preds must contain only 0/1.")
        if not (dt_early < dt_optimal < dt_late):
            raise ValueError("dt_early < dt_optimal < dt_late is required (offsets relative to onset).")

    L = len(labels) 
    u_t = np.zeros(L)
   # Identify sepsis onset time
    if np.any(labels == 1):
        onset = int(np.argmax(labels)) 
        is_septic = True
    else:
        onset = None
        is_septic = False

 # Slopes for utility calculation
    denom1 = float(dt_optimal - dt_early) 
    denom2 = float(dt_late - dt_optimal) 
    m1 = max_u_tp / denom1 if denom1 != 0 else 0.0         
    m2 = -max_u_tp / denom2 if denom2 != 0 else 0.0        
    m3 = min_u_fn / denom2 if denom2 != 0 else 0.0     
    
    # Compute utility for each time point
    for t in range(L):
        if not is_septic:
            if preds[t] == 1:
                u_t[t] = u_fp
            else:
                u_t[t] = u_tn
            continue
        x = t - onset # Time relative to sepsis onset
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

# Normalized utility calculation across patients
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
    """Compute normalized utility score across all patients."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)

    unique_patients = np.unique(groups)
    observed_utils = []
    best_utils = []
    inaction_utils = []
    worst_utils = []
    patient_ids = []

    # Calculate utilities for each patient
    for pid in unique_patients:
        idx = np.where(groups == pid)[0]
        if idx.size == 0:
            continue
        lab = y_true[idx].astype(int)
        pred_obs = y_pred[idx].astype(int)
        L = len(lab)

        u_obs = compute_patient_utility(
            lab, pred_obs,
            dt_early=dt_early, dt_optimal=dt_optimal, dt_late=dt_late,
            max_u_tp=max_u_tp, min_u_fn=min_u_fn, u_fp=u_fp, u_tn=u_tn
        )

        best_pred = np.zeros(L, dtype=int)
        if np.any(lab == 1):
            onset = int(np.argmax(lab))
            start_idx = max(0, onset + dt_early)
            end_idx = min(L, onset + dt_late + 1)
            if start_idx < end_idx:
                best_pred[start_idx:end_idx] = 1

        u_best = compute_patient_utility( 
            lab, best_pred,
            dt_early=dt_early, dt_optimal=dt_optimal, dt_late=dt_late,
            max_u_tp=max_u_tp, min_u_fn=min_u_fn, u_fp=u_fp, u_tn=u_tn
        )

        inaction_pred = np.zeros(L, dtype=int)
        u_inaction = compute_patient_utility(
            lab, inaction_pred,
            dt_early=dt_early, dt_optimal=dt_optimal, dt_late=dt_late,
            max_u_tp=max_u_tp, min_u_fn=min_u_fn, u_fp=u_fp, u_tn=u_tn
        )

        worst_pred = 1 - best_pred
        u_worst = compute_patient_utility(
            lab, worst_pred,
            dt_early=dt_early, dt_optimal=dt_optimal, dt_late=dt_late,
            max_u_tp=max_u_tp, min_u_fn=min_u_fn, u_fp=u_fp, u_tn=u_tn
        )

        patient_ids.append(pid)
        observed_utils.append(u_obs)
        best_utils.append(u_best)
        inaction_utils.append(u_inaction)
        worst_utils.append(u_worst)

    un_obs = float(np.sum(observed_utils))
    un_best = float(np.sum(best_utils))
    un_inaction = float(np.sum(inaction_utils))
    denom = (un_best - un_inaction)
    if denom == 0:
        normalized = float('nan')
    else:
        normalized = (un_obs - un_inaction) / denom

    details = {
        'patient_ids': np.array(patient_ids),
        'observed_utils': np.array(observed_utils),
        'best_utils': np.array(best_utils),
        'inaction_utils': np.array(inaction_utils),
        'worst_utils': np.array(worst_utils),
        'unnormalized': {'observed': un_obs, 'best': un_best, 'inaction': un_inaction}
    }
    if return_details:
        return normalized, details
    else:
        return normalized

# Data loading and preprocessing
INPUT_FILE = "Imputation_Result/LIGHTGBM.psv"
df = pd.read_csv(INPUT_FILE, sep='|')
df = df.drop(columns=['HospAdmTime'], errors='ignore')
X = df.drop(columns=["Paciente", "SepsisLabel"])
y = df["SepsisLabel"]
groups = df["Paciente"]

# Helper functions
def sensitivity_specificity(y_true, y_pred):
    """Compute sensitivity and specificity from confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp+fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn+fp) > 0 else 0
    return sensitivity, specificity

def best_threshold(y_true, y_proba, alpha):
    """Find optimal classification threshold balancing F1 and recall."""
    thresholds = np.linspace(0, 1, 200)
    f1_scores, recalls, combined_scores = [], [], []
    for thr in thresholds:
        preds = (y_proba >= thr).astype(int)
        f1_val = f1_score(y_true, preds, zero_division=0)
        rec_val = recall_score(y_true, preds, zero_division=0)
        f1_scores.append(f1_val)
        recalls.append(rec_val)
        combined_scores.append(alpha * f1_val + (1 - alpha) * rec_val)
    best_idx = np.argmax(combined_scores)
    return thresholds[best_idx], thresholds, f1_scores, recalls, combined_scores

def mean_ci(data):
    """Calculate mean and 95% confidence interval."""
    arr = np.array(data)
    mean = arr.mean()
    std = arr.std(ddof=1)
    n = len(arr)
    se = std / np.sqrt(n)
    h = se * 1.96
    return mean, h
print(f"X shape: {X.shape} | Class distribution:\n{y.value_counts()}")

# Split 85% train / 15% test (stratified by label)
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, groups, test_size=0.15, stratify=y, random_state=42
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
# Handle class imbalance
class_weight = {0: 1, 1: (((len(y_train) - sum(y_train)) / sum(y_train)) * 0.5)}
print("Class weights:", class_weight)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'sensitivity', 'specificity', 'utility']
val_scores = {metric: [] for metric in scoring}
all_y_true, all_y_proba = [], []
models, fold_metrics, fold_thresholds, all_fold_indices = [], [], [], []

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(tqdm(cv.split(X_train, y_train, groups=groups_train), total=5, desc="Processing folds")):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    all_fold_indices.append((train_idx.copy(), val_idx.copy()))
    model = lgb.LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    verbose=-1,
    class_weight=class_weight,
    n_jobs=-1,
    random_state=42
    )
    model.fit(X_tr, y_tr)
    # Predict and find optimal threshold
    y_proba = model.predict_proba(X_val)[:, 1]
    thr, thresholds, f1s, recalls, combined = best_threshold(y_val, y_proba, alpha=0.6)
    y_pred = (y_proba >= thr).astype(int)
    # Calculate fold metrics
    metrics_fold = {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) >= 2 else np.nan
    }
    sens, spec = sensitivity_specificity(y_val, y_pred)
    metrics_fold["sensitivity"] = sens
    metrics_fold["specificity"] = spec
    # Utility evaluation
    try:
        util_fold = compute_normalized_utility(
            y_true=y_val.values,
            y_proba=y_proba,
            y_pred=y_pred,
            groups=groups_train.iloc[val_idx].values,
            dt_early=-6, dt_optimal=0, dt_late=9,
            max_u_tp=1.0, min_u_fn=-2.0, u_fp=-0.05, u_tn=0.0
        )
    except Exception as e:
        util_fold = float('nan')
        warnings.warn(f"Error computing utility in fold {fold}: {e}")

    metrics_fold["utility"] = util_fold
    # Store results
    for metric in scoring:
        val_scores[metric].append(metrics_fold[metric])

    models.append(model)
    fold_metrics.append(metrics_fold)
    fold_thresholds.append(thr)

    all_y_true.extend(y_val)
    all_y_proba.extend(y_proba)

# Select best fold (weighted combination of F1 and sensitivity)
best_fold = np.argmax([0.3 * m["f1"] + 0.7 * m["sensitivity"] for m in fold_metrics])
best_model = models[best_fold]
best_threshold_fold = fold_thresholds[best_fold]

print(f"\n>>> Best Model: fold {best_fold+1}")
print(f"F1: {fold_metrics[best_fold]['f1']:.4f}")
print(f"Sensitivity: {fold_metrics[best_fold]['sensitivity']:.4f}")
print(f"Threshold: {best_threshold_fold:.3f}")

# Extract best fold data
train_idx_best, val_idx_best = all_fold_indices[best_fold]
X_tr_best, X_val_best = X_train.iloc[train_idx_best], X_train.iloc[val_idx_best]
y_tr_best, y_val_best = y_train.iloc[train_idx_best], y_train.iloc[val_idx_best]

# Optuna Hyperparameter Optimization
def objective(trial):
    params_optuna = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "class_weight": class_weight,
        "random_state": 42,
        "n_jobs": -1
    }

    model = lgb.LGBMClassifier(**params_optuna)
    model.fit(X_tr_best, y_tr_best)

    y_proba = model.predict_proba(X_val_best)[:, 1]
    y_pred = (y_proba >= best_threshold_fold).astype(int)
    sensitivity = recall_score(y_val_best, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_val_best, y_proba)
    f1 = f1_score(y_val_best, y_pred, zero_division=0)
    return 3 * sensitivity + 1 * f1 + 0.5 * auc_score
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("\n>>> Best Optuna Hyperparameters:")
for key, value in study.best_params.items():
    print(f"{key}: {value}")

# Train final Optuna-optimized model
optuna_model = lgb.LGBMClassifier(
    **study.best_params,
    objective='binary',
    boosting_type='gbdt',
    class_weight=class_weight,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
optuna_model.fit(X_tr_best, y_tr_best)
# Evaluate on validation set
y_val_proba_optuna = optuna_model.predict_proba(X_val_best)[:, 1]
best_thr_optuna, _, _, _, _ = best_threshold(y_val_best, y_val_proba_optuna, alpha=0.6)

print(f">>> Best threshold after Optuna: {best_thr_optuna:.3f}")
# Final Evaluation on Test Set
y_test_proba_optuna = optuna_model.predict_proba(X_test)[:, 1]
y_test_pred_optuna = (y_test_proba_optuna >= best_thr_optuna).astype(int)

test_metrics = {
    "accuracy": accuracy_score(y_test, y_test_pred_optuna),
    "f1": f1_score(y_test, y_test_pred_optuna, zero_division=0),
    "precision": precision_score(y_test, y_test_pred_optuna, zero_division=0),
    "recall": recall_score(y_test, y_test_pred_optuna, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_test_proba_optuna) if len(np.unique(y_test)) >= 2 else np.nan
}
sens_test, spec_test = sensitivity_specificity(y_test, y_test_pred_optuna)
test_metrics["sensitivity"] = sens_test
test_metrics["specificity"] = spec_test

# Utility score on test set
normalized_utility_test, util_details_test = compute_normalized_utility(
    y_true=y_test.values,
    y_proba=y_test_proba_optuna,
    y_pred=y_test_pred_optuna,
    groups=groups_test.values,
    dt_early=-6, dt_optimal=0, dt_late=9,
    max_u_tp=1.0, min_u_fn=-2.0, u_fp=-0.05, u_tn=0.0,
    return_details=True
)
test_metrics["utility"] = normalized_utility_test

print("\n" + "="*60)
print("=== FINAL TEST SET METRICS ===")
print("="*60)
for metric, value in test_metrics.items():
    if not np.isnan(value):
        print(f"{metric:15}: {value:.4f}")

print("\n" + "="*60)
print("=== CROSS-VALIDATION RESULTS (5-FOLD) ===")
print("="*60)
for metric in scoring:
    scores = [m[metric] for m in fold_metrics if not np.isnan(m[metric])]
    if scores:
        mean, h = mean_ci(scores)
        print(f"{metric:15}: {mean:.3f} ± {h:.3f} (95% CI)")

cm_test = confusion_matrix(y_test, y_test_pred_optuna)
print(f"\nConfusion Matrix (Test):\n{cm_test}")

# SHAP Analysis
print("\nCalculating SHAP values by PATIENTS...")

explainer = shap.TreeExplainer(optuna_model)
shap_values_all = explainer.shap_values(X_test)

if isinstance(shap_values_all, list):
    shap_values_all = shap_values_all[1]
# Aggregate SHAP values by patient
shap_df = pd.DataFrame(shap_values_all, index=groups_test, columns=X_test.columns)
shap_values_patient = shap_df.groupby(level=0).mean()
X_test_patient = X_test.groupby(groups_test).mean()

print(f"Original SHAP values: {shap_values_all.shape} (per hour)")
print(f"Aggregated SHAP values: {shap_values_patient.shape} (per patient)")

# Save results
output_dir = "Results/LightGBM_Results"
os.makedirs(output_dir, exist_ok=True)

util_patients_df = pd.DataFrame({
    'Paciente': util_details_test['patient_ids'],
    'ObservedUtility': util_details_test['observed_utils'],
    'BestUtility': util_details_test['best_utils'],
    'InactionUtility': util_details_test['inaction_utils'],
    'WorstUtility': util_details_test['worst_utils'],
})
util_patients_df.to_csv(os.path.join(output_dir, "utility_por_paciente_test.csv"), index=False)

# 1. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba_optuna)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0,1],[0,1], color="gray", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ROC_curve_optuna.pdf"), dpi=300)
plt.savefig(os.path.join(output_dir, "ROC_curve_optuna2.pdf"), format="pdf", bbox_inches="tight")
plt.close()

# 2. Confusion Matrix
plt.figure(figsize=(8,6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix - Test set")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix_optuna.pdf"), dpi=300)
plt.savefig(os.path.join(output_dir, "confusion_matrix_optuna2.pdf"), format="pdf", bbox_inches="tight")
plt.close()

# 3. SHAP Summary Plots
plt.figure(figsize=(12,6))
shap.summary_plot(shap_values_patient.values, X_test_patient, plot_type="bar", show=False)
plt.title("Global Feature Importance", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_global_importance.pdf"), dpi=300)
plt.close()

# 4. SHAP Beeswarm Plot
plt.figure(figsize=(12,8))
custom_cmap = plt.cm.get_cmap("cool")
shap.summary_plot(shap_values_patient.values, X_test_patient, show=False, cmap=custom_cmap)
plt.title("Local Explanation Summary ", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_beeswarm_pacientes.pdf"), dpi=300)
plt.close()

# 5. SHAP Feature Importance Ranking
mean_abs_shap_patient = np.abs(shap_values_patient.values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    "Variable": X_test_patient.columns,
    "Importance (%)": 100 * mean_abs_shap_patient / mean_abs_shap_patient.sum()
}).sort_values(by="Importance (%)", ascending=False)
shap_importance_df.to_csv(os.path.join(output_dir, "shap_ranking_pacientes.csv"), index=False)

# 6. SHAP - Top 15 variables
top15 = shap_importance_df.head(15)
plt.figure(figsize=(10,6))
ax = sns.barplot(
    data=top15,
    x="Importance (%)",
    y="Variable",
    hue="Variable",
    dodge=False,
    legend=False,
    palette="viridis"
)
for i, v in enumerate(top15["Importance (%)"]):
    ax.text(v + 0.3, i, f"{v:.2f}", color="black", va="center")
plt.title("Global Importance of Variables SHAP)")
plt.xlabel("Importance (%)")
plt.ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_ranking_top15_pacientes.pdf"), dpi=300)
plt.close()

print(f"\nResults saved in: {output_dir}")
print(f" SHAP computed for {shap_values_patient.shape[0]} patients")
print("\n Process completed successfully!")
