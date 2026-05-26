import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib import rcParams
import warnings
warnings.filterwarnings("ignore")

WORK    = Path("C:/Users/Salim/Desktop/makaleler/Sedat Arslan ML Large")
FIG_DIR = WORK / "figures"
OUT     = WORK / "analysis_data"
DPI     = 300

rcParams["font.family"] = "Arial"
rcParams["font.size"] = 10

CB_VERMILLON = "#D55E00"
CB_GREEN     = "#009E73"
CB_PURPLE    = "#7B2D8E"
CB_GREY      = "#999999"

FEAT_LABELS = {
    "negative_score":       "Negative score",
    "salt_100g":            "Salt",
    "saturated_fat_100g":   "Saturated fat",
    "additives_count":      "Additives count",
    "positive_score":       "Positive score",
    "sugar_100g":           "Sugar",
    "energy_kcal_100g":     "Energy",
    "carbs_100g":           "Carbohydrates",
    "salt_energy_ratio":    "Salt-energy ratio",
    "sat_fat_ratio":        "Sat. fat ratio",
    "protein_energy_ratio": "Protein-energy ratio",
    "sugar_carb_ratio":     "Sugar-carb ratio",
    "protein_100g":         "Protein",
    "fat_energy_ratio":     "Fat-energy ratio",
    "fat_100g":             "Fat",
}

NS_PENALIZED = {"negative_score", "salt_100g", "saturated_fat_100g",
                "sugar_100g", "energy_kcal_100g", "fat_100g",
                "salt_energy_ratio", "sat_fat_ratio", "sugar_carb_ratio",
                "fat_energy_ratio", "carbs_100g"}
NS_REWARDED  = {"positive_score", "protein_100g", "protein_energy_ratio"}
NS_BLIND     = {"additives_count"}

def feat_color(f):
    if f in NS_BLIND:     return CB_PURPLE
    if f in NS_REWARDED:  return CB_GREEN
    if f in NS_PENALIZED: return CB_VERMILLON
    return CB_GREY

model_dir = OUT / "model_objects"
shap_data = np.load(model_dir / "shap_data.npz", allow_pickle=True)
shap_values = shap_data["shap_values"]
X_shap = shap_data["X_shap"]
features = list(shap_data["features"])
features_beeswarm = [FEAT_LABELS.get(f, f) for f in features]

shap_imp = pd.read_csv(OUT / "shap_importance.csv")
sorted_all = shap_imp.sort_values("all", ascending=False)["feature"].tolist()
print(f"Data loaded: {X_shap.shape[0]:,} samples")

print("\nGenerating Fig 5...")

fig = plt.figure(figsize=(22, 15))
gs = GridSpec(2, 2, width_ratios=[36, 64], height_ratios=[60, 40],
              wspace=0.08, hspace=0.25)

# ============================================================
# Panel a — Beeswarm (top left)
# ============================================================
ax1 = fig.add_subplot(gs[0, 0])
plt.sca(ax1)
shap.summary_plot(shap_values, X_shap,
                  feature_names=features_beeswarm,
                  show=False, max_display=12,
                  cmap="cividis", plot_size=None)

shap_abs = pd.DataFrame(np.abs(shap_values), columns=features)
sorted_features = shap_abs.mean().sort_values(ascending=False).index[:12].tolist()

for label in ax1.get_yticklabels():
    text = label.get_text()
    orig = [f for f in features if FEAT_LABELS.get(f, f) == text]
    if orig:
        label.set_color(feat_color(orig[0]))
        label.set_fontweight("bold")

xlim = ax1.get_xlim()
ax1.text(xlim[0] * 0.6, 12.6, "\u2190  Reduces paradox",
         fontsize=11, fontweight="bold", color="#444444", ha="center")
ax1.text(xlim[1] * 0.5, 12.6, "Increases paradox  \u2192",
         fontsize=11, fontweight="bold", color="#444444", ha="center")

if "additives_count" in sorted_features:
    y_pos = 12 - 1 - sorted_features.index("additives_count")
    ax1.axhspan(y_pos - 0.4, y_pos + 0.4, color=CB_PURPLE, alpha=0.07, zorder=0)

ax1.set_xlabel("SHAP value (impact on model output)", fontsize=10)
ax1.set_title("a", fontweight="bold", loc="left", fontsize=14)

legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=CB_VERMILLON,
           markersize=8, label="NutriScore penalised"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=CB_GREEN,
           markersize=8, label="NutriScore rewarded"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=CB_PURPLE,
           markersize=8, label="NutriScore blind"),
]
ax1.legend(handles=legend_elements, loc="lower right", fontsize=8,
           frameon=True, fancybox=False, edgecolor="#CCCCCC",
           bbox_to_anchor=(1.05, 0.02))

# ============================================================
# Panel c — Subgroup heatmap (bottom left)
# ============================================================
ax3 = fig.add_subplot(gs[1, 0])

top10_feats = sorted_all[:10]
subgroups = ["all", "nova4", "nova1", "composite", "beverages", "dairy"]
subgroup_labels = ["All", "NOVA 4", "NOVA 1", "Composite\nfoods", "Beverages", "Dairy"]

heat_data = shap_imp.set_index("feature").loc[top10_feats, subgroups].values
heat_labels = [FEAT_LABELS.get(f, f) for f in top10_feats]
heat_colors = [feat_color(f) for f in top10_feats]

im = ax3.imshow(heat_data, aspect="auto", cmap="YlOrRd", interpolation="nearest")

for i in range(heat_data.shape[0]):
    for j in range(heat_data.shape[1]):
        val = heat_data[i, j]
        rank_col = shap_imp.sort_values(subgroups[j], ascending=False)["feature"].tolist()
        r = rank_col.index(top10_feats[i]) + 1
        text_color = "white" if val > 0.6 else "black"
        ax3.text(j, i, str(r), ha="center", va="center",
                 fontsize=9, fontweight="bold", color=text_color)

ax3.set_xticks(range(len(subgroups)))
ax3.set_xticklabels(subgroup_labels, fontsize=9, fontweight="bold")
ax3.set_yticks(range(len(top10_feats)))
ax3.set_yticklabels(heat_labels, fontsize=9)

for idx, label in enumerate(ax3.get_yticklabels()):
    label.set_color(heat_colors[idx])
    label.set_fontweight("bold")

cbar = plt.colorbar(im, ax=ax3, fraction=0.03, pad=0.04)
cbar.set_label("Mean |SHAP|", fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax3.set_title("c", fontweight="bold", loc="left", fontsize=14)

# ============================================================
# Panel b — Bump chart full height (right, 15 features, BIG)
# ============================================================
ax2 = fig.add_subplot(gs[:, 1])

ranks = {}
for col in ["all", "nova4", "nova1"]:
    sorted_f = shap_imp.sort_values(col, ascending=False)["feature"].tolist()
    ranks[col] = {f: sorted_f.index(f) + 1 for f in sorted_all}

x_pos = [0.5, 1.5, 2.5]

for feat in sorted_all:
    r = [ranks["all"][feat], ranks["nova4"][feat], ranks["nova1"][feat]]
    color = feat_color(feat)
    label = FEAT_LABELS.get(feat, feat)
    is_add = feat == "additives_count"
    lw = 5.0 if is_add else 2.5
    alpha = 1.0 if is_add else 0.5
    ms = 26 if is_add else 18

    ax2.plot(x_pos, r, "-", color=color, linewidth=lw, alpha=alpha,
             zorder=4 if is_add else 2)

    for xi, ri in zip(x_pos, r):
        ax2.plot(xi, ri, "o", color=color, markersize=ms,
                 markeredgecolor="white", markeredgewidth=2.0,
                 zorder=5 if is_add else 3, alpha=max(alpha, 0.85))
        ax2.text(xi, ri, str(ri), ha="center", va="center",
                 fontsize=12 if is_add else 10, fontweight="bold",
                 color="white", zorder=6)

    ax2.text(0.15, ranks["all"][feat], label, ha="right", va="center",
             fontsize=13 if is_add else 11, color=color,
             fontweight="bold" if is_add else "normal")

    ax2.text(2.85, ranks["nova1"][feat], label, ha="left", va="center",
             fontsize=13 if is_add else 11, color=color,
             fontweight="bold" if is_add else "normal")

if "additives_count" in sorted_all:
    r_all = ranks["all"]["additives_count"]
    r_n1  = ranks["nova1"]["additives_count"]
    ax2.annotate(f"Rank {r_all} \u2192 {r_n1}",
                 xy=(2.5, r_n1 - 0.05), xytext=(2.3, r_n1 - 2.0),
                 fontsize=14, fontweight="bold", color=CB_PURPLE,
                 arrowprops=dict(arrowstyle="->", color=CB_PURPLE, lw=2))

ax2.set_ylim(16, 0.3)
ax2.set_xlim(-0.6, 3.6)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(["All\nproducts", "NOVA 4\n(ultra-processed)", "NOVA 1\n(unprocessed)"],
                     fontsize=13, fontweight="bold")
ax2.set_yticks([])
ax2.grid(axis="x", linewidth=0.3, color="#EEEEEE")
for spine in ["top", "right", "left", "bottom"]:
    ax2.spines[spine].set_visible(False)
ax2.set_title("b", fontweight="bold", loc="left", fontsize=14)

plt.savefig(str(FIG_DIR / "fig5_shap_combined.pdf"), dpi=DPI, bbox_inches="tight")
plt.savefig(str(FIG_DIR / "fig5_shap_combined.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  Fig 5 saved")

print("\nPhase 4 figures complete.")