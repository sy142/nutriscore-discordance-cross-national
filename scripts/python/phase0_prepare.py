import pandas as pd
import numpy as np
import os
from pathlib import Path

WORK = Path("C:/Users/Salim/Desktop/makaleler/Sedat Arslan ML Large")
os.chdir(WORK)
os.makedirs("analysis_data", exist_ok=True)
os.makedirs("figures", exist_ok=True)

raw = pd.read_csv("nutriscore_merged_full.csv", low_memory=False, dtype={"barcode": str})
print(f"Raw: {raw.shape[0]:,} products x {raw.shape[1]} columns")

# Both NutriScore and NOVA available
has_both = raw.dropna(subset=["nutriscore", "nova_group"]).copy()
has_both = has_both[
    has_both["nutriscore"].isin(["a", "b", "c", "d", "e"]) &
    has_both["nova_group"].isin([1, 2, 3, 4])
]
print(f"Both NutriScore + NOVA: {len(has_both):,}")

# Discordance flags
has_both["ns_healthy"]   = has_both["nutriscore"].isin(["a", "b"]).astype(int)
has_both["ns_unhealthy"] = has_both["nutriscore"].isin(["d", "e"]).astype(int)
has_both["nova4"]        = (has_both["nova_group"] == 4).astype(int)
has_both["nova1"]        = (has_both["nova_group"] == 1).astype(int)

has_both["paradox_up"]   = (has_both["nova4"] & has_both["ns_healthy"]).astype(int)
has_both["paradox_down"] = (has_both["nova1"] & has_both["ns_unhealthy"]).astype(int)
has_both["paradox_any"]  = (has_both["paradox_up"] | has_both["paradox_down"]).astype(int)

n_up   = has_both["paradox_up"].sum()
n_down = has_both["paradox_down"].sum()
n_any  = has_both["paradox_any"].sum()
n_tot  = len(has_both)

print(f"\nDiscordance flags:")
print(f"  Upward   (NOVA4 + NS A/B): {n_up:>8,}  ({n_up/n_tot*100:.1f}%)")
print(f"  Downward (NOVA1 + NS D/E): {n_down:>8,}  ({n_down/n_tot*100:.1f}%)")
print(f"  Any paradox:               {n_any:>8,}  ({n_any/n_tot*100:.1f}%)")

# Feature engineering
eps = 0.01
has_both["sugar_carb_ratio"]    = has_both["sugar_100g"] / (has_both["carbs_100g"] + eps)
has_both["fat_energy_ratio"]    = has_both["fat_100g"] * 9 / (has_both["energy_kcal_100g"] + eps)
has_both["sat_fat_ratio"]       = has_both["saturated_fat_100g"] / (has_both["fat_100g"] + eps)
has_both["protein_energy_ratio"]= has_both["protein_100g"] * 4 / (has_both["energy_kcal_100g"] + eps)
has_both["salt_energy_ratio"]   = has_both["salt_100g"] / (has_both["energy_kcal_100g"] + eps)
has_both["negative_score"]      = has_both["sugar_100g"] + has_both["fat_100g"] + has_both["salt_100g"]
has_both["positive_score"]      = has_both["protein_100g"] + has_both["fiber_100g"].fillna(0)
print(f"Engineered features: 7 added")

# Food group cleanup
has_both["food_group_clean"] = (
    has_both["food_group"]
    .fillna("Unknown")
    .str.split(",").str[0]
    .str.strip()
)
top_groups = has_both["food_group_clean"].value_counts()
print(f"\nFood groups: {len(top_groups)} unique")
print(top_groups.head(10).to_string())

# Country-level aggregation
country_agg = (
    has_both
    .groupby("country_code")
    .agg(
        n_products       = ("barcode", "count"),
        n_paradox_up     = ("paradox_up", "sum"),
        n_paradox_down   = ("paradox_down", "sum"),
        n_paradox_any    = ("paradox_any", "sum"),
        pct_nova4        = ("nova4", "mean"),
        pct_ns_healthy   = ("ns_healthy", "mean"),
        pct_ns_unhealthy = ("ns_unhealthy", "mean"),
        mean_ns_score    = ("nutriscore_score", "mean"),
        mean_energy      = ("energy_kcal_100g", "mean"),
        mean_sugar       = ("sugar_100g", "mean"),
        mean_salt        = ("salt_100g", "mean"),
        mean_fat         = ("fat_100g", "mean"),
        mean_sat_fat     = ("saturated_fat_100g", "mean"),
        mean_fiber       = ("fiber_100g", "mean"),
        mean_protein     = ("protein_100g", "mean"),
        mean_additives   = ("additives_count", "mean"),
    )
    .reset_index()
)
country_agg["discordance_up_pct"]   = country_agg["n_paradox_up"]   / country_agg["n_products"] * 100
country_agg["discordance_down_pct"] = country_agg["n_paradox_down"] / country_agg["n_products"] * 100
country_agg["discordance_any_pct"]  = country_agg["n_paradox_any"]  / country_agg["n_products"] * 100
country_agg = country_agg[country_agg["n_products"] >= 50].copy()
print(f"\nCountry aggregates: {len(country_agg)} countries (min 50 products)")

# Merge health indicators
health = pd.read_csv("external_data/country_merged_dataset.csv")
health_cols = [c for c in [
    "country_code", "ncd_mortality_pct", "diabetes_pct",
    "life_expectancy", "overweight_adult_pct",
    "gdp_per_capita_ppp", "population", "smoking_pct",
    "alcohol_per_capita", "health_exp_ppp_pc", "health_exp_pct_gdp",
    "kcal_per_capita_day", "protein_g_per_capita_day", "fat_g_per_capita_day",
] if c in health.columns]

health_sub = health[health_cols].drop_duplicates("country_code")
country_full = country_agg.merge(health_sub, on="country_code", how="left")
print(f"Countries with NCD data: {country_full['ncd_mortality_pct'].notna().sum()} / {len(country_full)}")

# Export
has_both.to_csv("analysis_data/product_level.csv", index=False)
country_full.to_csv("analysis_data/country_level.csv", index=False)

print(f"\nExported:")
print(f"  product_level.csv : {len(has_both):>10,} rows")
print(f"  country_level.csv : {len(country_full):>10,} rows")

# Quick check: cross-tab and top countries
print(f"\nNutriScore x NOVA cross-tab:")
ct = pd.crosstab(has_both["nova_group"], has_both["nutriscore"], margins=True)
print(ct)

print(f"\nTop 10 countries by product count:")
print(country_full.nlargest(10, "n_products")[
    ["country_code", "n_products", "discordance_up_pct", "discordance_down_pct", "discordance_any_pct"]
].to_string(index=False))


# Duplication check
n_unique = has_both["barcode"].nunique()
n_total  = len(has_both)
n_dup    = n_total - n_unique
print(f"\nDuplication check:")
print(f"  Total rows:      {n_total:>10,}")
print(f"  Unique barcodes: {n_unique:>10,}")
print(f"  Duplicate rows:  {n_dup:>10,}  (same product sold in multiple countries)")
print(f"  Duplication rate: {n_dup/n_total*100:.1f}%")

# Country coverage comparison
n_country_full = raw["country_code"].nunique()
n_country_nova = has_both["country_code"].nunique()
print(f"\nCountry coverage:")
print(f"  Full dataset (944K):         {n_country_full} countries")
print(f"  NutriScore+NOVA subset:      {n_country_nova} countries")
print(f"  Lost due to NOVA filter:     {n_country_full - n_country_nova} countries")


# Deduplicated version for RQ1 and RQ4
dedup = has_both.drop_duplicates(subset="barcode", keep="first")
dedup.to_csv("analysis_data/product_dedup.csv", index=False)
print(f"\nDeduplicated export:")
print(f"  product_dedup.csv : {len(dedup):>10,} rows (unique barcodes)")
print(f"  Paradox rate (dedup): {dedup['paradox_any'].mean()*100:.1f}%")

