import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

WORK    = Path("C:/Users/Salim/Desktop/makaleler/Sedat Arslan ML Large")
FIG_DIR = WORK / "figures"
OUT     = WORK / "analysis_data"
DPI     = 300

rcParams["font.family"] = "Arial"
rcParams["font.size"] = 10

# Load saved objects
model_dir = OUT / "model_objects"
best_clf  = joblib.load(model_dir / "xgb_best_model.pkl")

data = np.load(model_dir / "ml_dataset.npz")
X, y = data["X"], data["y"]

shap_data = np.load(model_dir / "shap_data.npz", allow_pickle=True)
shap_values = shap_data["shap_values"]
X_shap = shap_data["X_shap"]
features = list(shap_data["features"])

thresh_data = np.load(model_dir / "threshold_data.npz")
y_prob = thresh_data["y_prob"]
fpr = thresh_data["fpr"]
tpr = thresh_data["tpr"]
best_threshold = float(thresh_data["best_threshold"][0])

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"Data loaded: {X.shape[0]:,} x {X.shape[1]}")

# S Fig 2 — Learning curve
print("\nGenerating S Fig 2: Learning curve")
train_sizes_frac = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])
train_sizes_abs, train_scores, test_scores = learning_curve(
    best_clf, X, y, train_sizes=train_sizes_frac,
    cv=cv5, scoring="f1_macro", n_jobs=-1, random_state=42)

fig, ax = plt.subplots(figsize=(7, 5))
ax.fill_between(train_sizes_abs,
                train_scores.mean(axis=1) - train_scores.std(axis=1),
                train_scores.mean(axis=1) + train_scores.std(axis=1),
                alpha=0.15, color="#2E75B6")
ax.fill_between(train_sizes_abs,
                test_scores.mean(axis=1) - test_scores.std(axis=1),
                test_scores.mean(axis=1) + test_scores.std(axis=1),
                alpha=0.15, color="#C0392B")
ax.plot(train_sizes_abs, train_scores.mean(axis=1), "o-",
        color="#2E75B6", label="Training", markersize=5)
ax.plot(train_sizes_abs, test_scores.mean(axis=1), "o-",
        color="#C0392B", label="Validation", markersize=5)
ax.set_xlabel("Training set size")
ax.set_ylabel("F1-macro score")
ax.legend(frameon=True, fancybox=False, edgecolor="#CCCCCC")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(str(FIG_DIR / "sfig2_learning_curve.pdf"), dpi=DPI, bbox_inches="tight")
plt.savefig(str(FIG_DIR / "sfig2_learning_curve.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  S Fig 2 saved")

# S Fig 3 — ROC curve + Calibration curve (2 panel)
print("Generating S Fig 3: ROC + Calibration")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

auc_val = roc_auc_score(y, y_prob)
best_idx = np.argmax(tpr - fpr)
ax1.plot(fpr, tpr, color="#C0392B", lw=1.5, label=f"XGBoost (AUC = {auc_val:.3f})")
ax1.plot([0, 1], [0, 1], "--", color="#B3B3B3")
ax1.scatter(fpr[best_idx], tpr[best_idx], s=80, c="black", zorder=5,
            label=f"Optimal threshold = {best_threshold:.3f}")
ax1.set_xlabel("False positive rate (1 - Specificity)")
ax1.set_ylabel("True positive rate (Sensitivity)")
ax1.set_title("a", fontweight="bold", loc="left", fontsize=13)
ax1.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="#CCCCCC")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10, strategy="uniform")
ax2.plot(prob_pred, prob_true, "o-", color="#2E75B6", label="XGBoost")
ax2.plot([0, 1], [0, 1], "--", color="#B3B3B3", label="Perfectly calibrated")
ax2.set_xlabel("Mean predicted probability")
ax2.set_ylabel("Fraction of positives")
ax2.set_title("b", fontweight="bold", loc="left", fontsize=13)
ax2.legend(frameon=True, fancybox=False, edgecolor="#CCCCCC")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(str(FIG_DIR / "sfig3_roc_calibration.pdf"), dpi=DPI, bbox_inches="tight")
plt.savefig(str(FIG_DIR / "sfig3_roc_calibration.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  S Fig 3 saved")

# S Fig 4 — SHAP dependence plots (top 4 features)
print("Generating S Fig 4: SHAP dependence")
shap_df = pd.DataFrame(np.abs(shap_values), columns=features)
top4 = shap_df.mean().sort_values(ascending=False).index[:4].tolist()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, feat in enumerate(top4):
    ax = axes[i // 2, i % 2]
    shap.dependence_plot(feat, shap_values, X_shap,
                          feature_names=features, show=False, ax=ax)
    ax.set_title(chr(ord("a") + i), fontweight="bold", loc="left", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(str(FIG_DIR / "sfig4_shap_dependence.pdf"), dpi=DPI, bbox_inches="tight")
plt.savefig(str(FIG_DIR / "sfig4_shap_dependence.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  S Fig 4 saved")

print("\nSupplementary Python figures complete.")