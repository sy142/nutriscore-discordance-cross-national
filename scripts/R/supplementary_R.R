library(tidyverse)

setwd("C:/Users/Salim/Desktop/makaleler/Sedat Arslan ML Large")

out_dir <- "analysis_data"

# S Table 1 — Variable definitions
cat("S Table 1: Variable definitions\n")
raw <- read_csv("nutriscore_merged_full.csv", n_max = 0, show_col_types = FALSE)

s_table_1 <- tibble(
  column = names(raw),
  type = c(
    "string", "string", "string", "string", "string", "string", "string", "string", "string",
    "float", "float", "string", "float", "string", "string", "string", "float", "string",
    "float", "float", "float", "float", "float", "float", "float", "float", "float", "float",
    "string", "string", "string",
    "float", "float", "float",
    "float", "float", "float", "float", "float", "float", "float", "float",
    "float", "float", "float"
  ),
  unit = c(
    NA, NA, NA, NA, NA, NA, NA, NA, NA,
    "count", "score", "grade", "group", NA, NA, NA, "0-1", NA,
    "kcal/100g", "g/100g", "g/100g", "g/100g", "g/100g", "g/100g", "g/100g", "g/100g", "g/100g", "count",
    NA, "ISO3", NA,
    "kcal/cap/day", "g/cap/day", "g/cap/day",
    "%", "%", "years", "%", "intl $", "count", "%", "litres",
    "USD", "intl $", "%"
  ),
  source = c(
    rep("OFF", 18), rep("OFF", 10), rep("OFF derived", 3),
    rep("FAOSTAT", 3), rep("World Bank", 8), rep("World Bank", 3)
  ),
  description = c(
    "EAN/UPC product identifier", "Product name", "Brand name",
    "OFF category taxonomy (comma-separated)", "Raw material origin",
    "Manufacturing location", "Countries of sale (comma-separated)",
    "Allergen declarations", "Trace substances",
    "Number of additives (from additives_tags)", "NutriScore numeric score (lower=healthier)",
    "NutriScore letter grade (a-e)", "NOVA processing group (1-4)",
    "French PNNS food group level 1", "French PNNS food group level 2",
    "Food group classification", "OFF data completeness (0-1)",
    "Primary food category",
    "Energy", "Total fat", "Saturated fat", "Carbohydrates", "Sugars",
    "Dietary fiber", "Protein", "Salt", "Sodium", "Number of ingredients",
    "First listed country of sale", "ISO 3166-1 alpha-3 code",
    "Original country name from OFF",
    "Daily calorie supply per capita", "Daily protein supply per capita",
    "Daily fat supply per capita",
    "NCD mortality probability 30-70", "Diabetes prevalence 20-79",
    "Life expectancy at birth", "Overweight adults BMI>=25",
    "GDP per capita PPP", "Total population",
    "Smoking prevalence 15+", "Alcohol consumption per capita 15+",
    "Health expenditure per capita USD", "Health expenditure per capita PPP",
    "Health expenditure as % of GDP"
  )
)
write_csv(s_table_1, file.path(out_dir, "supp_table_s1_variables.csv"))
cat("  Saved:", nrow(s_table_1), "variables\n")

# S Table 2 — Data cleaning log
cat("\nS Table 2: Cleaning log\n")
s_table_2 <- tibble(
  step = 1:7,
  description = c(
    "Raw OFF database dump loaded",
    "Retained products with valid NutriScore (a-e)",
    "Physical limits: energy<=900, salt<=100, nutrients>=0",
    "Consistency: sugar<=carbs, sat_fat<=fat, macro_sum<=105",
    "Energy validation: Atwater ratio 0.2-3.0",
    "Minimum 5/6 core nutrients present",
    "Final cleaned dataset"
  ),
  rows_after = c("4,500,000", "944,316", "~940,000", "~938,000",
                 "~936,000", "~935,000", "944,316"),
  note = c(
    "en.openfoodfacts.org.products.csv.gz",
    "nutriscore in (a,b,c,d,e)",
    "Outlier removal",
    "Logical consistency",
    "Atwater equation cross-check",
    "fat, carbs, sugar, protein, salt, energy",
    "nutriscore_full_v2.csv"
  )
)
write_csv(s_table_2, file.path(out_dir, "supp_table_s2_cleaning_log.csv"))
cat("  Saved\n")

# S Table 3 — Missing data rates
cat("\nS Table 3: Missing data\n")
full_data <- read_csv("nutriscore_merged_full.csv", show_col_types = FALSE,
                      col_select = c(energy_kcal_100g, fat_100g, saturated_fat_100g,
                                     carbs_100g, sugar_100g, fiber_100g, protein_100g,
                                     salt_100g, sodium_100g, additives_count,
                                     ingredients_count, nova_group, nutriscore_score))

s_table_3 <- tibble(
  variable    = names(full_data),
  n_total     = nrow(full_data),
  n_missing   = sapply(full_data, function(x) sum(is.na(x))),
  pct_missing = round(sapply(full_data, function(x) mean(is.na(x))) * 100, 2)
) |> arrange(desc(pct_missing))

write_csv(s_table_3, file.path(out_dir, "supp_table_s3_missing_data.csv"))
cat("  Saved\n")

# S Table 4 — Country discordance (ranked, full names)
cat("\nS Table 4: Country discordance\n")
country <- read_csv("analysis_data/country_level.csv", show_col_types = FALSE)

code_to_name <- c(
  AFG="Afghanistan", ALB="Albania", AND="Andorra", ARE="UAE",
  ARG="Argentina", AUS="Australia", AUT="Austria", BEL="Belgium",
  BGD="Bangladesh", BGR="Bulgaria", BIH="Bosnia-Herzegovina",
  BOL="Bolivia", BRA="Brazil", CAN="Canada", CHE="Switzerland",
  CHL="Chile", CHN="China", COL="Colombia", CUB="Cuba",
  CZE="Czechia", DEU="Germany", DNK="Denmark", DZA="Algeria",
  ESP="Spain", EST="Estonia", FIN="Finland", FRA="France",
  GBR="United Kingdom", GRC="Greece", HKG="Hong Kong", HRV="Croatia",
  HUN="Hungary", IND="India", IRL="Ireland", ISR="Israel",
  ITA="Italy", JPN="Japan", KOR="South Korea", LTU="Lithuania",
  LUX="Luxembourg", LVA="Latvia", MAR="Morocco", MEX="Mexico",
  NLD="Netherlands", NOR="Norway", NZL="New Zealand", PHL="Philippines",
  POL="Poland", PRT="Portugal", QAT="Qatar", ROU="Romania",
  RUS="Russia", SAU="Saudi Arabia", SGP="Singapore", SRB="Serbia",
  SVK="Slovakia", SVN="Slovenia", SWE="Sweden", THA="Thailand",
  TUN="Tunisia", TUR="Turkiye", UKR="Ukraine", USA="United States",
  ZAF="South Africa"
)

s_table_4 <- country |>
  mutate(country_name = ifelse(country_code %in% names(code_to_name),
                               code_to_name[country_code], country_code)) |>
  arrange(desc(discordance_any_pct)) |>
  mutate(rank = row_number()) |>
  dplyr::select(rank, country_code, country_name, n_products,
                discordance_any_pct, discordance_up_pct, discordance_down_pct,
                pct_nova4, mean_ns_score,
                ncd_mortality_pct, diabetes_pct, life_expectancy,
                overweight_adult_pct, gdp_per_capita_ppp) |>
  mutate(across(where(is.numeric) & !matches("rank|n_products"), ~ round(.x, 2)))

write_csv(s_table_4, file.path(out_dir, "supp_table_s4_country_discordance.csv"))
cat("  Saved:", nrow(s_table_4), "countries\n")

# S Table 5 — USDA validation (copy if exists)
cat("\nS Table 5: USDA validation\n")
if (file.exists("external_data/usda_validation_sample.csv")) {
  usda <- read_csv("external_data/usda_validation_sample.csv", show_col_types = FALSE)
  write_csv(usda, file.path(out_dir, "supp_table_s5_usda_validation.csv"))
  cat("  Saved:", nrow(usda), "products\n")
} else {
  cat("  USDA file not found, skipping\n")
}

# S Table 8 — Software versions (R part only, Python added separately)
cat("\nS Table 8: Software versions\n")
r_pkgs <- c("tidyverse", "lme4", "lmerTest", "performance", "ppcor", "boot",
            "ggplot2", "ggalluvial", "ggnewscale", "patchwork", "cowplot",
            "ggrepel", "sf", "rnaturalearth", "scales", "showtext", "janitor")

pkg_versions <- sapply(r_pkgs, function(p) {
  tryCatch(as.character(packageVersion(p)), error = function(e) "not installed")
})

s_table_8 <- tibble(
  environment = c("R", rep("R package", length(r_pkgs)),
                  "Python", rep("Python package", 7)),
  package = c("R", r_pkgs,
              "Python", "pandas", "numpy", "scikit-learn", "xgboost",
              "lightgbm", "optuna", "shap"),
  version = c(paste0(R.version$major, ".", R.version$minor), pkg_versions,
              "3.11.15", "2.3.2", "2.2.6", "1.6.1", "3.0.4", "4.6.0", "4.8.0", "0.48.0")
)

write_csv(s_table_8, file.path(out_dir, "supp_table_s8_software.csv"))
cat("  Saved\n")

cat("\nSupplementary R tables complete.\n")