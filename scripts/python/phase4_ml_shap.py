import pandas as pd
import numpy as np
from pathlib import Path
from time import time
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

WORK = Path("C:/Users/Salim/Desktop/makaleler/Sedat Arslan ML Large")

prod = pd.read_csv(WORK / "analysis_data" / "product_dedup.csv", low_memory=False)
print(f"Products: {len(prod):,}")

features = [
    "energy_kcal_100g", "fat_100g", "saturated_fat_100g",
    "carbs_100g", "sugar_100g", "protein_100g", "salt_100g",
    "additives_count",
    "sugar_carb_ratio", "fat_energy_ratio", "sat_fat_ratio",
    "protein_energy_ratio", "salt_energy_ratio",
    "negative_score", "positive_score",
]
target = "paradox_any"

ml = prod[features + [target]].dropna()
ml = ml[np.isfinite(ml[features]).all(axis=1)]
X = ml[features].values
y = ml[target].values

print(f"ML dataset: {X.shape[0]:,} x {X.shape[1]}")
print(f"Paradox: {y.sum():,} ({y.mean()*100:.1f}%)")

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

baselines = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=300, eval_metric="logloss",
        scale_pos_weight=(1-y.mean())/y.mean(),
        tree_method="hist", random_state=42, n_jobs=-1),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=300, is_unbalance=True,
        random_state=42, n_jobs=-1, verbose=-1),
}

print("\nBaseline comparison (default params, 5-fold CV):")
print("-" * 65)

for name, model in baselines.items():
    t0 = time()
    cv_res = cross_validate(model, X, y, cv=cv5,
        scoring=["f1_macro", "roc_auc", "accuracy"], n_jobs=-1)
    t = time() - t0
    f1  = cv_res["test_f1_macro"]
    auc = cv_res["test_roc_auc"]
    acc = cv_res["test_accuracy"]
    print(f"  {name:25s}: F1={f1.mean():.4f}+-{f1.std():.4f}  "
          f"AUC={auc.mean():.4f}  Acc={acc.mean():.4f}  ({t:.0f}s)")


import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def xgb_objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
        "max_depth":        trial.suggest_int("max_depth", 3, 12),
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma":            trial.suggest_float("gamma", 0, 5),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-5, 10, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-5, 10, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 15),
    }
    clf = xgb.XGBClassifier(**params, eval_metric="logloss",
                            tree_method="hist", random_state=42, n_jobs=-1)
    scores = cross_validate(clf, X, y, cv=cv5, scoring="f1_macro", n_jobs=-1)
    return scores["test_score"].mean()

print("XGBoost Optuna (300 trials)...")
t0 = time()
xgb_study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(xgb_objective, n_trials=300, show_progress_bar=True)
print(f"\nDone in {(time()-t0)/60:.1f} min")
print(f"Best F1-macro: {xgb_study.best_value:.4f}")
print(f"Best params: {xgb_study.best_params}")


from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, brier_score_loss, roc_curve
from sklearn.model_selection import cross_val_predict
import shap
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

best_params = xgb_study.best_params
best_clf = xgb.XGBClassifier(**best_params, eval_metric="logloss",
                              tree_method="hist", random_state=42, n_jobs=-1)

# Full CV evaluation
SCORING = ["f1_macro", "f1_weighted", "accuracy", "roc_auc", "precision_macro", "recall_macro"]
cv_final = cross_validate(best_clf, X, y, cv=cv5, scoring=SCORING, n_jobs=-1)

print("Final CV results (tuned XGBoost):")
for s in SCORING:
    vals = cv_final[f"test_{s}"]
    print(f"  {s:20s}: {vals.mean():.4f} +/- {vals.std():.4f}")

# Threshold optimization
print("\nThreshold optimization...")
y_prob = cross_val_predict(best_clf, X, y, cv=cv5, method="predict_proba")[:, 1]
fpr, tpr, thresholds = roc_curve(y, y_prob)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

print(f"Optimal threshold (Youden's J): {best_threshold:.4f}")
print(f"Sensitivity: {tpr[best_idx]:.4f}")
print(f"Specificity: {1-fpr[best_idx]:.4f}")
print(f"AUC: {roc_auc_score(y, y_prob):.4f}")
print(f"Brier: {brier_score_loss(y, y_prob):.4f}")
print(f"MCC: {matthews_corrcoef(y, (y_prob >= 0.5).astype(int)):.4f}")

# SHAP
print("\nFitting full model for SHAP...")
best_clf.fit(X, y)
joblib.dump(best_clf, str(WORK / "analysis_data" / "best_model.pkl"))

rng = np.random.RandomState(42)
idx = rng.choice(len(X), size=50000, replace=False)
X_shap = X[idx]

print("Computing SHAP values (50K)...")
explainer = shap.TreeExplainer(best_clf)
shap_values = explainer.shap_values(X_shap)

shap_df = pd.DataFrame(shap_values, columns=features)
mean_abs = shap_df.abs().mean().sort_values(ascending=False)

print("\nFeature importance (mean |SHAP|):")
for feat, val in mean_abs.items():
    print(f"  {feat:25s}: {val:.4f}")



# SHAP interactions (10K sample)
print("Computing SHAP interaction values (10K)...")
idx_int = rng.choice(len(X), size=10000, replace=False)
shap_interaction = explainer.shap_interaction_values(X[idx_int])

top_interactions = []
n_feat = len(features)
for i in range(n_feat):
    for j in range(i+1, n_feat):
        val = np.abs(shap_interaction[:, i, j]).mean()
        top_interactions.append((features[i], features[j], val))

int_df = pd.DataFrame(top_interactions, columns=["feature_1", "feature_2", "mean_abs_interaction"])
int_df = int_df.sort_values("mean_abs_interaction", ascending=False)
print("\nTop 10 feature interactions:")
print(int_df.head(10).to_string(index=False))

# Subgroup SHAP function
def subgroup_shap(data, label, n_sample=30000):
    sub = data[features + [target]].dropna()
    sub = sub[np.isfinite(sub[features]).all(axis=1)]
    Xs = sub[features].values
    idx_s = rng.choice(len(Xs), size=min(n_sample, len(Xs)), replace=False)
    sv = explainer.shap_values(Xs[idx_s])
    sv_df = pd.DataFrame(sv, columns=features)
    imp = sv_df.abs().mean().sort_values(ascending=False)
    print(f"\n--- {label} (n = {len(sub):,}) ---")
    for feat, val in imp.head(8).items():
        print(f"  {feat:25s}: {val:.4f}")
    return imp

# NOVA subgroups
imp_nova4 = subgroup_shap(prod[prod["nova_group"] == 4], "NOVA 4 (ultra-processed)")
imp_nova1 = subgroup_shap(prod[prod["nova_group"] == 1], "NOVA 1 (unprocessed)")

# Top 3 food categories
imp_composite  = subgroup_shap(prod[prod["food_group_clean"] == "Composite foods"], "Composite foods")
imp_beverages  = subgroup_shap(prod[prod["food_group_clean"] == "Beverages"], "Beverages")
imp_dairy      = subgroup_shap(prod[prod["food_group_clean"] == "Milk and dairy products"], "Dairy")

# Export combined importance
shap_imp = pd.DataFrame({"feature": features})
shap_imp["all"]       = [shap_df[f].abs().mean() for f in features]
shap_imp["nova4"]     = [imp_nova4[f] for f in features]
shap_imp["nova1"]     = [imp_nova1[f] for f in features]
shap_imp["composite"] = [imp_composite[f] for f in features]
shap_imp["beverages"] = [imp_beverages[f] for f in features]
shap_imp["dairy"]     = [imp_dairy[f] for f in features]
shap_imp = shap_imp.sort_values("all", ascending=False)
shap_imp.to_csv(WORK / "analysis_data" / "shap_importance.csv", index=False)
int_df.to_csv(WORK / "analysis_data" / "shap_interactions.csv", index=False)

print("\nExported: shap_importance.csv, shap_interactions.csv")


import joblib
import pickle
import numpy as np
from pathlib import Path

save_dir = WORK / "analysis_data" / "model_objects"
save_dir.mkdir(exist_ok=True)

# Optuna study
with open(save_dir / "optuna_xgb_study.pkl", "wb") as f:
    pickle.dump(xgb_study, f)

# Model + params
joblib.dump(best_clf, save_dir / "xgb_best_model.pkl")
joblib.dump(best_params, save_dir / "best_params.pkl")

# SHAP data
np.savez_compressed(save_dir / "shap_data.npz",
                    shap_values=shap_values, X_shap=X_shap,
                    features=np.array(features))

# CV + threshold
joblib.dump(cv_final, save_dir / "cv_results.pkl")
np.savez_compressed(save_dir / "threshold_data.npz",
                    y_prob=y_prob, fpr=fpr, tpr=tpr,
                    thresholds=thresholds,
                    best_threshold=np.array([best_threshold]))

# Dataset
np.savez_compressed(save_dir / "ml_dataset.npz", X=X, y=y)

# Versions
import sys
print(f"Python: {sys.version}")
for pkg in ["pandas", "numpy", "sklearn", "xgboost", "lightgbm", "optuna", "shap", "joblib"]:
    mod = __import__(pkg)
    print(f"  {pkg}: {mod.__version__}")

print("\nSaved to", save_dir)
for f in sorted(save_dir.iterdir()):
    print(f"  {f.name}: {f.stat().st_size / (1024*1024):.1f} MB")

import pandas as pd

s8 = pd.read_csv(WORK / "analysis_data" / "supp_table_s8_software.csv")

py_versions = {
    "pandas": "2.3.2", "numpy": "2.2.6", "scikit-learn": "1.6.1",
    "xgboost": "3.0.4", "lightgbm": "4.6.0", "optuna": "4.8.0",
    "shap": "0.48.0"
}

for pkg, ver in py_versions.items():
    s8.loc[s8["package"] == pkg, "version"] = ver

s8.to_csv(WORK / "analysis_data" / "supp_table_s8_software.csv", index=False)
print("S Table 8 updated with Python versions")
print(s8.to_string(index=False))

# Baseline comparison CSV (4 algoritma)
baseline_df = pd.DataFrame([
    {"model": "Logistic Regression", "f1_macro": 0.5675, "roc_auc": 0.8056, "accuracy": 0.6721},
    {"model": "Random Forest",       "f1_macro": 0.8341, "roc_auc": 0.9553, "accuracy": 0.9447},
    {"model": "XGBoost (default)",   "f1_macro": 0.7826, "roc_auc": 0.9513, "accuracy": 0.8946},
    {"model": "LightGBM (default)",  "f1_macro": 0.7545, "roc_auc": 0.9485, "accuracy": 0.8718},
    {"model": "XGBoost (tuned)",     "f1_macro": 0.8581, "roc_auc": 0.9595, "accuracy": 0.9496},
])
baseline_df.to_csv(WORK / "analysis_data" / "ml_baseline_comparison.csv", index=False)
print("ml_baseline_comparison.csv saved")

# Final performance CSV
perf = pd.DataFrame({
    "metric": ["f1_macro", "f1_weighted", "accuracy", "roc_auc", "precision_macro", "recall_macro"],
    "mean": [cv_final[f"test_{s}"].mean() for s in ["f1_macro", "f1_weighted", "accuracy", "roc_auc", "precision_macro", "recall_macro"]],
    "std":  [cv_final[f"test_{s}"].std()  for s in ["f1_macro", "f1_weighted", "accuracy", "roc_auc", "precision_macro", "recall_macro"]],
})
perf.to_csv(WORK / "analysis_data" / "ml_performance.csv", index=False)
print("ml_performance.csv saved")

# Threshold optimization CSV
thresh_df = pd.DataFrame([
    {"threshold": 0.5,
     "f1_macro": f1_score(y, (y_prob >= 0.5).astype(int), average="macro"),
     "mcc": matthews_corrcoef(y, (y_prob >= 0.5).astype(int)),
     "sensitivity": None, "specificity": None,
     "brier": brier_score_loss(y, y_prob)},
    {"threshold": round(best_threshold, 4),
     "f1_macro": f1_score(y, (y_prob >= best_threshold).astype(int), average="macro"),
     "mcc": matthews_corrcoef(y, (y_prob >= best_threshold).astype(int)),
     "sensitivity": round(tpr[np.argmax(tpr - fpr)], 4),
     "specificity": round(1 - fpr[np.argmax(tpr - fpr)], 4),
     "brier": brier_score_loss(y, y_prob)},
])
thresh_df.to_csv(WORK / "analysis_data" / "threshold_optimization.csv", index=False)
print("threshold_optimization.csv saved")

# Best params CSV
params_df = pd.DataFrame([{"param": k, "value": v} for k, v in best_params.items()])
params_df.to_csv(WORK / "analysis_data" / "optuna_best_params.csv", index=False)
print("optuna_best_params.csv saved")

# SHAP subgroup importance (already saved but verify)
print("\nVerify existing CSVs:")
for f in ["shap_importance.csv", "shap_interactions.csv", "ml_baseline_comparison.csv",
           "ml_performance.csv", "threshold_optimization.csv", "optuna_best_params.csv"]:
    p = WORK / "analysis_data" / f
    print(f"  {f}: {'OK' if p.exists() else 'MISSING'}")


# Sensitivity analyses
from sklearn.model_selection import StratifiedKFold, cross_validate

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sensitivity = []

# S1: Main model (already done, just record)
sensitivity.append({
    "analysis": "Main model (15 features)",
    "n_samples": len(ml),
    "n_features": 15,
    "f1_macro": cv_final["test_f1_macro"].mean(),
    "f1_std": cv_final["test_f1_macro"].std()
})

# S2: Core features only (no engineered)
features_core = ["energy_kcal_100g", "fat_100g", "saturated_fat_100g",
                  "carbs_100g", "sugar_100g", "protein_100g", "salt_100g",
                  "additives_count"]
ml_core = prod[features_core + [target]].dropna()
ml_core = ml_core[np.isfinite(ml_core[features_core]).all(axis=1)]
clf_core = xgb.XGBClassifier(**best_params, eval_metric="logloss",
                              tree_method="hist", random_state=42, n_jobs=-1)
cv_core = cross_validate(clf_core, ml_core[features_core].values,
                          ml_core[target].values, cv=cv5, scoring="f1_macro", n_jobs=-1)
sensitivity.append({
    "analysis": "Core features only (8)",
    "n_samples": len(ml_core),
    "n_features": 8,
    "f1_macro": cv_core["test_score"].mean(),
    "f1_std": cv_core["test_score"].std()
})
print(f"Core only: F1={cv_core['test_score'].mean():.4f}")

# S3: With fiber (complete cases)
features_fib = features + ["fiber_100g"]
ml_fib = prod[features_fib + [target]].dropna()
ml_fib = ml_fib[np.isfinite(ml_fib[features_fib]).all(axis=1)]
clf_fib = xgb.XGBClassifier(**best_params, eval_metric="logloss",
                             tree_method="hist", random_state=42, n_jobs=-1)
cv_fib = cross_validate(clf_fib, ml_fib[features_fib].values,
                         ml_fib[target].values, cv=cv5, scoring="f1_macro", n_jobs=-1)
sensitivity.append({
    "analysis": "With fiber (16 features)",
    "n_samples": len(ml_fib),
    "n_features": 16,
    "f1_macro": cv_fib["test_score"].mean(),
    "f1_std": cv_fib["test_score"].std()
})
print(f"With fiber: F1={cv_fib['test_score'].mean():.4f} (n={len(ml_fib):,})")

# S4: Upward paradox only
ml_up = prod[features + ["paradox_up"]].dropna()
ml_up = ml_up[np.isfinite(ml_up[features]).all(axis=1)]
clf_up = xgb.XGBClassifier(**best_params, eval_metric="logloss",
                            tree_method="hist", random_state=42, n_jobs=-1)
cv_up = cross_validate(clf_up, ml_up[features].values,
                        ml_up["paradox_up"].values, cv=cv5, scoring="f1_macro", n_jobs=-1)
sensitivity.append({
    "analysis": "Upward paradox only",
    "n_samples": len(ml_up),
    "n_features": 15,
    "f1_macro": cv_up["test_score"].mean(),
    "f1_std": cv_up["test_score"].std()
})
print(f"Upward only: F1={cv_up['test_score'].mean():.4f}")

# Export
sens_df = pd.DataFrame(sensitivity)
sens_df.to_csv(WORK / "analysis_data" / "ml_sensitivity.csv", index=False)
print("\nml_sensitivity.csv saved")
print(sens_df.to_string(index=False))


