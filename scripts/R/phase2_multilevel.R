library(tidyverse)
library(lme4)
library(lmerTest)
library(performance)

setwd("C:/Users/Salim/Desktop/makaleler/Sedat Arslan ML Large")

prod    <- read_csv("analysis_data/product_level.csv", show_col_types = FALSE)
country <- read_csv("analysis_data/country_level.csv", show_col_types = FALSE)
cat("Product-country pairs:", format(nrow(prod), big.mark = ","), "\n")

# Keep only countries in country_level (min 50 products, valid codes)
valid_countries <- country$country_code

prod <- prod |>
  filter(!is.na(country_code), country_code %in% valid_countries) |>
  mutate(country_code = factor(country_code))

cat("After country filter:", format(nrow(prod), big.mark = ","), "rows,",
    n_distinct(prod$country_code), "countries\n")

# Cluster sizes
cluster_sizes <- prod |> count(country_code) |> arrange(n)
cat("\nSmallest 5 clusters:\n")
print(head(cluster_sizes, 5))
cat("Largest 5 clusters:\n")
print(tail(cluster_sizes, 5))

# ICC from continuous nutriscore_score (fast check)
cat("\n--- ICC: continuous NutriScore score ---\n")
m_lmer <- lmer(nutriscore_score ~ 1 + (1 | country_code), data = prod, REML = TRUE)
icc_cont <- icc(m_lmer)
print(icc_cont)

vc <- as.data.frame(VarCorr(m_lmer))
cat("Between-country:", round(vc$vcov[1], 4), "\n")
cat("Within-country: ", round(vc$vcov[2], 4), "\n")
cat("ICC:            ", round(vc$vcov[1] / sum(vc$vcov), 4), "\n")

# ICC from binary paradox_any (logistic)
cat("\n--- ICC: binary paradox_any ---\n")
cat("Fitting null model...\n")
m0 <- glmer(paradox_any ~ 1 + (1 | country_code),
            data = prod, family = binomial, nAGQ = 0,
            control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))

var_country  <- as.data.frame(VarCorr(m0))$vcov[1]
var_residual <- pi^2 / 3
icc_bin      <- var_country / (var_country + var_residual)

cat("Between-country variance:", round(var_country, 4), "\n")
cat("ICC (binary):            ", round(icc_bin, 4),
    sprintf("(%.1f%%)\n", icc_bin * 100))

# Design effect
mean_cluster <- prod |> count(country_code) |> pull(n) |> mean()
de <- 1 + (mean_cluster - 1) * icc_bin
cat("Mean cluster size:", round(mean_cluster, 1), "\n")
cat("Design effect:    ", round(de, 2), "\n")

# Standardize product-level predictors
prod <- prod |>
  mutate(across(c(sugar_100g, fat_100g, salt_100g, protein_100g,
                  saturated_fat_100g, additives_count, energy_kcal_100g),
                ~ scale(.x)[, 1],
                .names = "z_{.col}"))

# Model 1: product-level predictors
cat("\n--- Model 1: product-level predictors ---\n")
cat("Fitting...\n")
m1 <- glmer(
  paradox_any ~ z_sugar_100g + z_fat_100g + z_salt_100g +
    z_protein_100g + z_saturated_fat_100g + z_additives_count +
    z_energy_kcal_100g + (1 | country_code),
  data = prod, family = binomial, nAGQ = 0,
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))

cat("\nFixed effects:\n")
print(summary(m1)$coefficients)

m1_var <- as.data.frame(VarCorr(m1))$vcov[1]
cat("Residual between-country variance:", round(m1_var, 4), "\n")
cat("Variance reduction from null:     ", sprintf("%.1f%%\n", (1 - m1_var / var_country) * 100))

# Model 2: cross-level interaction (sugar x GDP)
cat("\n--- Model 2: cross-level interaction ---\n")
prod_cross <- prod |>
  filter(!is.na(gdp_per_capita_ppp)) |>
  mutate(z_gdp = scale(log(gdp_per_capita_ppp))[, 1])

cat("Rows with GDP:", format(nrow(prod_cross), big.mark = ","), "\n")
cat("Fitting...\n")

m2 <- glmer(
  paradox_any ~ z_sugar_100g * z_gdp + z_fat_100g + z_salt_100g +
    z_protein_100g + z_saturated_fat_100g + z_additives_count +
    z_energy_kcal_100g + (1 | country_code),
  data = prod_cross, family = binomial, nAGQ = 0,
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))

cat("\nFixed effects:\n")
print(summary(m2)$coefficients)

int_p <- summary(m2)$coefficients["z_sugar_100g:z_gdp", "Pr(>|z|)"]
cat("\nSugar x GDP interaction p =", round(int_p, 4), "\n")
if (int_p < 0.05) {
  cat("GDP significantly moderates the sugar-discordance relationship\n")
} else {
  cat("GDP does not significantly moderate the sugar-discordance relationship\n")
}

# Country random effects
cat("\n--- Country random effects ---\n")
re <- ranef(m0)$country_code
re$country_code <- rownames(re)
names(re)[1] <- "random_intercept"

re <- re |>
  left_join(country |> select(country_code, n_products, discordance_any_pct),
            by = "country_code") |>
  arrange(random_intercept)

cat("\nBottom 5 (lowest discordance tendency):\n")
print(head(re, 5))
cat("\nTop 5 (highest discordance tendency):\n")
print(tail(re, 5))

# Export
write_csv(re, "analysis_data/country_random_effects.csv")

sink("analysis_data/multilevel_results.txt")
cat("Multilevel models ??? 63 countries (min 50 products per country)\n")
cat("Products:", nrow(prod), "\n\n")
cat("=== Null model (continuous) ===\n"); print(summary(m_lmer))
cat("\n\n=== Null model (binary) ===\n"); print(summary(m0))
cat("\nICC (binary):", icc_bin, "\n")
cat("\n\n=== Model 1 (product predictors) ===\n"); print(summary(m1))
cat("\nVariance reduction:", round((1 - m1_var / var_country) * 100, 1), "%\n")
cat("\n\n=== Model 2 (cross-level interaction) ===\n"); print(summary(m2))
sink()

cat("\nExported: multilevel_results.txt, country_random_effects.csv\n")
cat("Phase 2 complete.\n")
