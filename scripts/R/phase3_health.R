library(tidyverse)
library(ppcor)
library(boot)

setwd("C:/Users/Salim/Desktop/makaleler/Sedat Arslan ML Large")

df <- read_csv("analysis_data/country_level.csv", show_col_types = FALSE)
cat("Countries loaded:", nrow(df), "\n")

# Income group classification
df <- df |>
  mutate(income_group = case_when(
    gdp_per_capita_ppp >= 40000 ~ "HIC",
    gdp_per_capita_ppp >= 12000 ~ "UMIC",
    TRUE                        ~ "LMIC"
  ))

cat("\nIncome distribution:\n")
print(table(df$income_group, useNA = "ifany"))

# Outcome and predictor overview
outcomes    <- c("ncd_mortality_pct", "diabetes_pct", "life_expectancy", "overweight_adult_pct")
labels      <- c("NCD mortality (%)", "Diabetes (%)", "Life expectancy (yr)", "Overweight (%)")
confounders <- c("gdp_per_capita_ppp", "smoking_pct", "alcohol_per_capita")

cat("\nMissing data per outcome:\n")
for (v in c(outcomes, confounders, "discordance_any_pct")) {
  n_miss <- sum(is.na(df[[v]]))
  cat(sprintf("  %-25s: %d missing (n = %d)\n", v, n_miss, sum(!is.na(df[[v]]))))
}

# Bivariate correlations (Spearman)
cat("\n--- Bivariate correlations (Spearman) ---\n")
biv_results <- tibble()

for (i in seq_along(outcomes)) {
  sub <- df |> select(discordance_any_pct, all_of(outcomes[i])) |> drop_na()
  r   <- cor.test(sub$discordance_any_pct, sub[[outcomes[i]]], method = "spearman")
  biv_results <- bind_rows(biv_results, tibble(
    outcome   = outcomes[i],
    label     = labels[i],
    rho       = as.numeric(r$estimate),
    p_value   = r$p.value,
    n         = nrow(sub)
  ))
  cat(sprintf("  %-25s: rho = %+.3f, p = %.4f, n = %d\n",
              labels[i], r$estimate, r$p.value, nrow(sub)))
}

# Partial correlations (controlling GDP + smoking + alcohol)
cat("\n--- Partial correlations (controlling GDP, smoking, alcohol) ---\n")
partial_results <- tibble()

for (i in seq_along(outcomes)) {
  sub <- df |>
    select(discordance_any_pct, all_of(outcomes[i]), all_of(confounders)) |>
    drop_na()
  
  if (nrow(sub) < 10) {
    cat(sprintf("  %-25s: insufficient data (n = %d)\n", labels[i], nrow(sub)))
    next
  }
  
  pc <- pcor.test(sub$discordance_any_pct, sub[[outcomes[i]]],
                  sub[confounders], method = "spearman")
  
  partial_results <- bind_rows(partial_results, tibble(
    outcome   = outcomes[i],
    label     = labels[i],
    partial_r = pc$estimate,
    p_value   = pc$p.value,
    n         = nrow(sub)
  ))
  
  cat(sprintf("  %-25s: partial rho = %+.3f, p = %.4f, n = %d\n",
              labels[i], pc$estimate, pc$p.value, nrow(sub)))
}

# Weighted regression: discordance -> NCD mortality
cat("\n--- Weighted regression: discordance -> NCD mortality ---\n")
reg_data <- df |> drop_na(discordance_any_pct, ncd_mortality_pct, n_products)

m_unadj <- lm(ncd_mortality_pct ~ discordance_any_pct,
              data = reg_data, weights = log(n_products))

cat("\nUnadjusted:\n")
print(summary(m_unadj)$coefficients)
cat("R-squared:", round(summary(m_unadj)$r.squared, 4), "\n")

# Adjusted for confounders
reg_adj <- df |>
  drop_na(discordance_any_pct, ncd_mortality_pct, n_products,
          gdp_per_capita_ppp, smoking_pct, alcohol_per_capita)

m_adj <- lm(ncd_mortality_pct ~ discordance_any_pct +
              log(gdp_per_capita_ppp) + smoking_pct + alcohol_per_capita,
            data = reg_adj, weights = log(n_products))

cat("\nAdjusted (+ GDP + smoking + alcohol):\n")
print(summary(m_adj)$coefficients)
cat("R-squared:", round(summary(m_adj)$r.squared, 4), "\n")

# Repeat for all outcomes
cat("\n--- Adjusted regressions for all outcomes ---\n")
reg_all <- tibble()

for (i in seq_along(outcomes)) {
  sub <- df |>
    drop_na(discordance_any_pct, all_of(outcomes[i]), n_products,
            gdp_per_capita_ppp, smoking_pct, alcohol_per_capita)
  
  if (nrow(sub) < 10) next
  
  frm <- as.formula(paste(outcomes[i],
                          "~ discordance_any_pct + log(gdp_per_capita_ppp) + smoking_pct + alcohol_per_capita"))
  
  m <- lm(frm, data = sub, weights = log(n_products))
  s <- summary(m)
  coefs <- s$coefficients["discordance_any_pct", ]
  
  reg_all <- bind_rows(reg_all, tibble(
    outcome  = outcomes[i],
    label    = labels[i],
    beta     = coefs["Estimate"],
    se       = coefs["Std. Error"],
    t_value  = coefs["t value"],
    p_value  = coefs["Pr(>|t|)"],
    r_sq     = s$r.squared,
    n        = nrow(sub)
  ))
  
  cat(sprintf("  %-25s: beta = %+.4f, p = %.4f, R2 = %.3f, n = %d\n",
              labels[i], coefs["Estimate"], coefs["Pr(>|t|)"], s$r.squared, nrow(sub)))
}

# Bootstrap CI for NCD mortality coefficient
cat("\n--- Bootstrap 95% CI (NCD mortality) ---\n")
boot_fn <- function(data, idx) {
  d <- data[idx, ]
  m <- lm(ncd_mortality_pct ~ discordance_any_pct +
            log(gdp_per_capita_ppp) + smoking_pct + alcohol_per_capita,
          data = d, weights = log(n_products))
  coef(m)["discordance_any_pct"]
}

set.seed(42)
boot_out <- boot(reg_adj, boot_fn, R = 5000)
boot_ci  <- boot.ci(boot_out, type = "bca")

cat("Estimate:", round(boot_out$t0, 4), "\n")
if (!is.null(boot_ci$bca)) {
  cat("BCa 95% CI: [", round(boot_ci$bca[4], 4), ",", round(boot_ci$bca[5], 4), "]\n")
} else {
  boot_ci_perc <- boot.ci(boot_out, type = "perc")
  cat("Percentile 95% CI: [", round(boot_ci_perc$percent[4], 4), ",",
      round(boot_ci_perc$percent[5], 4), "]\n")
}

# Full correlation matrix
cat("\n--- Correlation matrix ---\n")
cor_vars <- c("discordance_any_pct", "discordance_up_pct", "mean_ns_score",
              "pct_nova4", outcomes, "gdp_per_capita_ppp",
              "smoking_pct", "alcohol_per_capita", "kcal_per_capita_day")
cor_vars <- cor_vars[cor_vars %in% names(df)]

cor_data <- df |> select(all_of(cor_vars)) |> drop_na()
cor_mat  <- cor(cor_data, method = "spearman")

cat("Correlation matrix dimensions:", nrow(cor_mat), "x", ncol(cor_mat), "\n")
cat("n =", nrow(cor_data), "complete cases\n")

# Export
write_csv(biv_results, "analysis_data/bivariate_correlations.csv")
write_csv(partial_results, "analysis_data/partial_correlations.csv")
write_csv(reg_all, "analysis_data/regression_results.csv")
write_csv(as_tibble(cor_mat, rownames = "variable"), "analysis_data/correlation_matrix.csv")

sink("analysis_data/health_results.txt")
cat("Health outcome associations ??? 63 countries\n\n")
cat("=== Bivariate correlations ===\n"); print(biv_results)
cat("\n\n=== Partial correlations ===\n"); print(partial_results)
cat("\n\n=== Adjusted regressions ===\n"); print(reg_all)
cat("\n\n=== Unadjusted NCD model ===\n"); print(summary(m_unadj))
cat("\n\n=== Adjusted NCD model ===\n"); print(summary(m_adj))
cat("\n\n=== Bootstrap CI ===\n")
cat("Estimate:", boot_out$t0, "\n")
print(boot_ci)
sink()

cat("\nExported: bivariate_correlations.csv, partial_correlations.csv,")
cat(" regression_results.csv, correlation_matrix.csv, health_results.txt\n")
cat("Phase 3 complete.\n")