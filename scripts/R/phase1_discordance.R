library(tidyverse)
library(janitor)
library(scales)
library(ggalluvial)

setwd("C:/Users/Salim/Desktop/makaleler/Sedat Arslan ML Large")

prod <- read_csv("analysis_data/product_dedup.csv", show_col_types = FALSE)
cat("Dedup products loaded:", format(nrow(prod), big.mark = ","), "\n")

# Nature Food theme and palettes
theme_nf <- theme_minimal(base_family = "Arial", base_size = 9) +
  theme(
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_blank(),
    strip.text         = element_text(face = "bold", size = 9),
    plot.title         = element_text(face = "bold", size = 11),
    plot.subtitle      = element_text(size = 9, colour = "grey40"),
    legend.position    = "bottom",
    axis.title         = element_text(size = 9),
    axis.text          = element_text(size = 8)
  )

pal_ns   <- c(a = "#2D8C3C", b = "#97C83C", c = "#FECB02", d = "#EE8100", e = "#E63E11")
pal_nova <- c("1" = "#4A90D9", "2" = "#7BC8A4", "3" = "#F4A460", "4" = "#C0392B")

# Cross-tabulation: counts
ct <- prod |>
  count(nova_group, nutriscore) |>
  group_by(nova_group) |>
  mutate(pct = n / sum(n) * 100) |>
  ungroup()

ct_wide <- ct |>
  select(-pct) |>
  pivot_wider(names_from = nutriscore, values_from = n, values_fill = 0) |>
  arrange(nova_group)

cat("\nCross-tabulation (counts):\n")
print(ct_wide)

# Cross-tabulation: row percentages
ct_pct <- ct |>
  select(-n) |>
  pivot_wider(names_from = nutriscore, values_from = pct, values_fill = 0) |>
  arrange(nova_group) |>
  mutate(across(where(is.numeric), ~ round(.x, 1)))

cat("\nCross-tabulation (row %):\n")
print(ct_pct)

# Chi-square and Cramer's V
tbl <- table(prod$nova_group, prod$nutriscore)
chi <- chisq.test(tbl)
k   <- min(nrow(tbl), ncol(tbl))
cramer_v <- as.numeric(sqrt(chi$statistic / (sum(tbl) * (k - 1))))

cat("\nChi-square test:\n")
cat("  X2 =", format(chi$statistic, big.mark = ","), "\n")
cat("  df =", chi$parameter, "\n")
cat("  p  <", format.pval(chi$p.value, digits = 3), "\n")
cat("  Cramer's V =", round(cramer_v, 4), "\n")

# Discordance summary
disc <- prod |>
  summarise(
    n_total      = n(),
    n_nova4_nsAB = sum(paradox_up),
    n_nova1_nsDE = sum(paradox_down),
    n_any        = sum(paradox_any),
    pct_up       = mean(paradox_up) * 100,
    pct_down     = mean(paradox_down) * 100,
    pct_any      = mean(paradox_any) * 100
  )

cat("\nDiscordance summary (unique products):\n")
cat("  NOVA4 + NutriScore A/B:", format(disc$n_nova4_nsAB, big.mark = ","),
    sprintf("(%.1f%%)\n", disc$pct_up))
cat("  NOVA1 + NutriScore D/E:", format(disc$n_nova1_nsDE, big.mark = ","),
    sprintf("(%.1f%%)\n", disc$pct_down))
cat("  Any paradox:           ", format(disc$n_any, big.mark = ","),
    sprintf("(%.1f%%)\n", disc$pct_any))

# Discordance within NOVA 4 only
nova4 <- prod |> filter(nova_group == 4)
cat("\nWithin NOVA 4 products (n =", format(nrow(nova4), big.mark = ","), "):\n")
cat("  NutriScore A:", sum(nova4$nutriscore == "a"),
    sprintf("(%.1f%%)\n", mean(nova4$nutriscore == "a") * 100))
cat("  NutriScore B:", sum(nova4$nutriscore == "b"),
    sprintf("(%.1f%%)\n", mean(nova4$nutriscore == "b") * 100))
cat("  A+B combined:", sum(nova4$ns_healthy),
    sprintf("(%.1f%%)\n", mean(nova4$ns_healthy) * 100))

# Discordance by food category
cat_disc <- prod |>
  filter(food_group_clean != "Unknown") |>
  group_by(food_group_clean) |>
  summarise(
    n        = n(),
    n_up     = sum(paradox_up),
    n_down   = sum(paradox_down),
    pct_up   = mean(paradox_up) * 100,
    pct_down = mean(paradox_down) * 100,
    pct_any  = mean(paradox_any) * 100,
    .groups  = "drop"
  ) |>
  filter(n >= 500) |>
  arrange(desc(pct_any))

cat("\nDiscordance by food category (n >= 500):\n")
print(cat_disc, n = 20)


write_csv(ct, "analysis_data/crosstab_ns_nova.csv")
write_csv(cat_disc, "analysis_data/category_discordance.csv")

cat("\nPhase 1 complete.\n")