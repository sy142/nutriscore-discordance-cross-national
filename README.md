# Nutritional quality scores overlook additive-driven ultra-processing globally


Data set:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20379040.svg)](https://doi.org/10.5281/zenodo.20379040)

> Arslan S, Yilmaz S, Gunal AM, Aydemir E. *Nature Food* (2026).

## Overview

We analyse 519,749 food products from the Open Food Facts database across 63 countries to quantify the discordance between NutriScore and the NOVA processing classification. One in ten products receives conflicting classifications. Machine learning (SHAP) identifies food additives count as a key discordance driver invisible to NutriScore.

## Repository structure

```
scripts/python/     Python scripts (data cleaning, ML, SHAP)
scripts/R/          R scripts (statistics, multilevel, figures)
data/raw/           Open Food Facts dump (not included, see below)
data/processed/     Analysis outputs (15 CSV files)
data/external/      World Bank, FAOSTAT indicators
figures/            Main (5), supplementary (4), extended data (1)
docs/               Supplementary tables (6 CSV files)
```

## Reproduction

### Requirements

**Python 3.11.15**
```bash
pip install pandas==2.3.2 numpy==2.2.6 scikit-learn==1.6.1 xgboost==3.0.4 lightgbm==4.6.0 optuna==4.8.0 shap==0.48.0 matplotlib seaborn
```

**R 4.5.2**
```r
install.packages(c("tidyverse","lme4","lmerTest","performance","ppcor","boot",
  "ggplot2","ggalluvial","ggnewscale","patchwork","cowplot","ggrepel",
  "sf","rnaturalearth","scales","showtext","janitor"))
```

### Execution order

| Phase | Script | Output |
|-------|--------|--------|
| 0 | `python scripts/python/process_full_off_dump_v2.py` | Cleaned product data |
| 0 | `python scripts/python/phase0_prepare.py` | Analysis-ready dataset |
| 0 | `python scripts/python/collect_external_data.py` | Country indicators |
| 1 | `Rscript scripts/R/phase1_discordance.R` | RQ1 analysis |
| 1 | `Rscript scripts/R/phase1_figures.R` | Figs 1–2 |
| 2 | `Rscript scripts/R/phase2_multilevel.R` | RQ2 analysis |
| 2 | `Rscript scripts/R/phase2_figures.R` | Fig 3 |
| 3 | `Rscript scripts/R/phase3_health.R` | RQ3 analysis |
| 3 | `Rscript scripts/R/phase3_figures.R` | Fig 4 |
| 4 | `python scripts/python/phase4_ml_shap.py` | RQ4 analysis (~8.5 h) |
| 4 | `python scripts/python/phase4_figures.py` | Fig 5 |
| S | `Rscript scripts/R/supplementary_R.R` | S Tables |
| S | `Rscript scripts/R/supplementary_R_figures.R` | S Fig 1 |
| S | `python scripts/python/supplementary_Python.py` | S Tables 6–7 |
| S | `python scripts/python/supplementary_Python_figures.py` | S Figs 2–4 |

## Data availability

| Dataset | Source | Licence |
|---------|--------|---------|
| Open Food Facts | https://world.openfoodfacts.org/ | ODbL |
| World Bank WDI | https://data.worldbank.org/ | CC-BY 4.0 |
| FAOSTAT | https://www.fao.org/faostat/ | Open |
| USDA FoodData Central | https://fdc.nal.usda.gov/ | Public domain |
| Processed data | This repository | CC-BY 4.0 |

## Software

See `docs/supp_table_s8_software.csv` for full version list.

## Citation

```bibtex
@article{arslan2026nutriscore,
  title={Nutritional quality scores overlook additive-driven
         ultra-processing globally},
  author={Arslan, Sedat and Yilmaz, Salim and Gunal, Ahmet Murat
          and Aydemir, Ecenur},
  journal={Nature Food},
  year={2026},
  doi={10.1038/s43016-026-XXXXX-X}
}
```

## Licence

Code: MIT | Data and figures: CC-BY 4.0
