library(tidyverse)
library(ggalluvial)
library(ggnewscale)
library(scales)
library(patchwork)
library(showtext)

font_add("nfont",
         regular = "C:/Windows/Fonts/arial.ttf",
         bold    = "C:/Windows/Fonts/arialbd.ttf",
         italic  = "C:/Windows/Fonts/ariali.ttf")
showtext_auto()
showtext_opts(dpi = 600)

setwd("C:/Users/Salim/Desktop/makaleler/Sedat Arslan ML Large")

prod     <- read_csv("analysis_data/product_dedup.csv", show_col_types = FALSE)
cat_disc <- read_csv("analysis_data/category_discordance.csv", show_col_types = FALSE)
re       <- read_csv("analysis_data/country_random_effects.csv", show_col_types = FALSE)

fig_dir <- "figures"
DPI     <- 600

theme_nf <- theme_minimal(base_family = "nfont", base_size = 10) +
  theme(
    panel.grid.minor   = element_blank(),
    panel.grid.major   = element_line(linewidth = 0.3, colour = "grey90"),
    legend.position    = "bottom",
    axis.title         = element_text(size = 10),
    axis.text          = element_text(size = 9),
    plot.background    = element_rect(fill = "white", colour = NA),
    panel.background   = element_rect(fill = "white", colour = NA),
    plot.margin        = margin(8, 8, 8, 8)
  )

pal_ns   <- c(a = "#2D8C3C", b = "#97C83C", c = "#FECB02", d = "#EE8100", e = "#E63E11")
pal_nova <- c("1" = "#4A90D9", "2" = "#7BC8A4", "3" = "#F4A460", "4" = "#C0392B")
pal_paradox <- c("Non-paradox" = "#95A5A6", "Paradox" = "#C0392B")

# ============================================================
# Fig 1 ??? Nutrient profiles: NutriScore vs NOVA vs Paradox
# ============================================================

nutrients <- c("sugar_100g", "saturated_fat_100g", "salt_100g",
               "protein_100g", "energy_kcal_100g", "additives_count")

nutrient_labels <- c(sugar_100g = "Sugar (g)", saturated_fat_100g = "Saturated fat (g)",
                     salt_100g = "Salt (g)", protein_100g = "Protein (g)",
                     energy_kcal_100g = "Energy (kcal)", additives_count = "Additives (n)")

make_violin <- function(data, group_var, group_label, fill_pal) {
  data |>
    dplyr::select(all_of(group_var), all_of(nutrients)) |>
    pivot_longer(-all_of(group_var), names_to = "nutrient", values_to = "value") |>
    drop_na(value) |>
    mutate(
      nutrient = nutrient_labels[nutrient],
      group = as.factor(.data[[group_var]])
    ) |>
    ggplot(aes(x = group, y = value, fill = group)) +
    geom_violin(alpha = 0.7, linewidth = 0.2, scale = "width") +
    geom_boxplot(width = 0.12, outlier.size = 0.2, linewidth = 0.2) +
    scale_fill_manual(values = fill_pal, guide = "none") +
    facet_wrap(~ nutrient, scales = "free_y", ncol = 6) +
    labs(x = group_label, y = NULL) +
    theme_nf +
    theme(strip.text = element_text(size = 8, face = "bold"),
          axis.text.x = element_text(size = 8))
}

fig1a <- make_violin(prod, "nutriscore", "NutriScore grade", pal_ns)

prod_nova <- prod |> mutate(nova_group = as.character(as.integer(nova_group)))
fig1b <- make_violin(prod_nova, "nova_group", "NOVA group", pal_nova)

prod_par <- prod |>
  mutate(paradox_label = ifelse(paradox_any == 1, "Paradox", "Non-paradox"))
fig1c <- make_violin(prod_par, "paradox_label", "Discordance status", pal_paradox)

fig1 <- fig1a / fig1b / fig1c +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(face = "bold", size = 13, family = "nfont"))

ggsave(file.path(fig_dir, "fig1_nutrient_profiles.pdf"), fig1,
       width = 14, height = 12, dpi = DPI, bg = "white")
ggsave(file.path(fig_dir, "fig1_nutrient_profiles.png"), fig1,
       width = 14, height = 12, dpi = DPI, bg = "white")
cat("Fig 1 (nutrient profiles) saved\n")

# ============================================================
# Fig 2 ??? Combined: Sankey + category bars + forest plot
# ============================================================

# Sankey data
sankey_data <- prod |>
  mutate(
    nova_label = paste0("NOVA ", nova_group),
    ns_label   = paste0("NutriScore ", toupper(nutriscore))
  ) |>
  count(nova_label, ns_label) |>
  mutate(
    flow_type = case_when(
      nova_label == "NOVA 4" & ns_label %in% c("NutriScore A", "NutriScore B") ~ "Upward paradox",
      nova_label == "NOVA 1" & ns_label %in% c("NutriScore D", "NutriScore E") ~ "Downward paradox",
      TRUE ~ "Concordant"
    )
  )

nova_n <- prod |> count(nova_group) |>
  mutate(label = paste0("NOVA ", nova_group, "\nn = ", format(n, big.mark = ",")))
ns_n <- prod |> count(nutriscore) |>
  mutate(label = paste0("NS ", toupper(nutriscore), "\nn = ", format(n, big.mark = ",")))
strata_labels <- c(
  setNames(nova_n$label, paste0("NOVA ", nova_n$nova_group)),
  setNames(ns_n$label, paste0("NutriScore ", toupper(ns_n$nutriscore)))
)

strata_pal <- c(
  "NOVA 1" = "#D6EAF8", "NOVA 2" = "#AED6F1",
  "NOVA 3" = "#F9E79F", "NOVA 4" = "#F5B7B1",
  "NutriScore A" = "#A9DFBF", "NutriScore B" = "#D5F5E3",
  "NutriScore C" = "#FEF9E7", "NutriScore D" = "#FDEBD0",
  "NutriScore E" = "#FADBD8"
)

n_up     <- sum(prod$paradox_up)
n_nova4  <- sum(prod$nova_group == 4)
pct_up   <- round(n_up / n_nova4 * 100, 1)
n_down   <- sum(prod$paradox_down)
n_nova1  <- sum(prod$nova_group == 1)
pct_down <- round(n_down / n_nova1 * 100, 1)

# Panel a ??? Sankey
p_a <- ggplot(sankey_data,
              aes(axis1 = nova_label, axis2 = ns_label, y = n)) +
  geom_alluvium(aes(fill = flow_type), width = 1/3, alpha = 0.7) +
  scale_fill_manual(
    values = c("Upward paradox" = "#C0392B",
               "Downward paradox" = "#E67E22",
               "Concordant" = "#D5D8DC"),
    name = NULL
  ) +
  new_scale_fill() +
  geom_stratum(aes(fill = after_stat(stratum)),
               width = 1/3, colour = "grey50", linewidth = 0.4, alpha = 0.85) +
  scale_fill_manual(values = strata_pal, guide = "none") +
  geom_text(stat = "stratum",
            aes(label = strata_labels[after_stat(stratum)]),
            size = 2.6, family = "nfont", lineheight = 0.85) +
  annotate("label", x = 1.50, y = 400000,
           label = paste0(format(n_up, big.mark = ","), " products (", pct_up, "% of NOVA 4)"),
           size = 2.6, family = "nfont", fontface = "bold",
           colour = "#8B1A1A", fill = "white", label.size = 0.2, angle = 58) +
  annotate("label", x = 1.65, y = 265000,
           label = paste0(format(n_down, big.mark = ","), " products (", pct_down, "% of NOVA 1)"),
           size = 2.4, family = "nfont", fontface = "bold",
           colour = "#B45F06", fill = "white", label.size = 0.2, angle = -58) +
  scale_y_continuous(labels = comma) +
  labs(y = "Number of products") +
  theme_nf +
  theme(axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        panel.grid.major.x = element_blank())

# Panel c ??? Stacked category bars (upward + downward)
cat_long <- cat_disc |>
  dplyr::select(food_group_clean, pct_up, pct_down) |>
  pivot_longer(-food_group_clean, names_to = "type", values_to = "pct") |>
  mutate(
    type = ifelse(type == "pct_up", "Upward", "Downward"),
    food_group_clean = fct_reorder(food_group_clean, pct, .fun = sum)
  )

p_c <- ggplot(cat_long, aes(x = pct, y = food_group_clean, fill = type)) +
  geom_col(width = 0.7) +
  geom_text(
    data = cat_disc |> mutate(food_group_clean = fct_reorder(food_group_clean, pct_any)),
    aes(x = pct_any, y = food_group_clean, label = sprintf("%.1f%%", pct_any)),
    inherit.aes = FALSE, hjust = -0.1, size = 3, family = "nfont"
  ) +
  scale_fill_manual(values = c("Upward" = "#C0392B", "Downward" = "#E67E22"), name = NULL) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(x = "Discordance rate (%)", y = NULL) +
  theme_nf +
  theme(panel.grid.major.x = element_blank(),
        axis.text.y = element_text(size = 9),
        legend.position = "right",
        legend.key.size = unit(0.3, "cm"),
        legend.text = element_text(size = 7))

# Panel b ??? Forest plot with region colors
code_to_name <- c(
  AFG="Afghanistan", ALB="Albania", AND="Andorra", ARE="UAE",
  ARG="Argentina", AUS="Australia", AUT="Austria", BEL="Belgium",
  BGD="Bangladesh", BGR="Bulgaria", BIH="Bosnia-Herz.",
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

code_to_region <- c(
  AFG="Asia", ALB="Europe", AND="Europe", ARE="Middle East",
  ARG="Latin America", AUS="Oceania", AUT="Europe", BEL="Europe",
  BGD="Asia", BGR="Europe", BIH="Europe",
  BOL="Latin America", BRA="Latin America", CAN="North America", CHE="Europe",
  CHL="Latin America", CHN="Asia", COL="Latin America", CUB="Latin America",
  CZE="Europe", DEU="Europe", DNK="Europe", DZA="Africa",
  ESP="Europe", EST="Europe", FIN="Europe", FRA="Europe",
  GBR="Europe", GRC="Europe", HKG="Asia", HRV="Europe",
  HUN="Europe", IND="Asia", IRL="Europe", ISR="Middle East",
  ITA="Europe", JPN="Asia", KOR="Asia", LTU="Europe",
  LUX="Europe", LVA="Europe", MAR="Africa", MEX="Latin America",
  NLD="Europe", NOR="Europe", NZL="Oceania", PHL="Asia",
  POL="Europe", PRT="Europe", QAT="Middle East", ROU="Europe",
  RUS="Europe", SAU="Middle East", SGP="Asia", SRB="Europe",
  SVK="Europe", SVN="Europe", SWE="Europe", THA="Asia",
  TUN="Africa", TUR="Middle East", UKR="Europe", USA="North America",
  ZAF="Africa"
)

pal_region <- c(
  "Europe"        = "#4A90D9",
  "Latin America" = "#E67E22",
  "Asia"          = "#27AE60",
  "North America" = "#8E44AD",
  "Middle East"   = "#C0392B",
  "Africa"        = "#F39C12",
  "Oceania"       = "#16A085"
)

re_plot <- re |>
  drop_na(discordance_any_pct) |>
  mutate(
    country_name = ifelse(country_code %in% names(code_to_name),
                          code_to_name[country_code], country_code),
    region = ifelse(country_code %in% names(code_to_region),
                    code_to_region[country_code], "Other"),
    country_name = fct_reorder(country_name, random_intercept)
  )

p_b <- ggplot(re_plot, aes(x = random_intercept, y = country_name)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50", linewidth = 0.4) +
  geom_point(aes(size = n_products, colour = region), alpha = 0.8) +
  scale_colour_manual(values = pal_region, name = "Region") +
  scale_size_continuous(range = c(1, 5.5), name = "Products", labels = comma,
                        breaks = c(1000, 50000, 200000, 400000)) +
  labs(x = "Country random intercept (log-odds)", y = NULL) +
  theme_nf +
  theme(axis.text.y = element_text(size = 5.5),
        legend.key.size = unit(0.3, "cm"),
        legend.text = element_text(size = 7),
        legend.title = element_text(size = 8, face = "bold"))

# Layout: left 40% (a top, c bottom), right 60% (b full height)
design <- "
AABBB
AABBB
CCBBB
"

fig2 <- p_a + p_b + p_c +
  plot_layout(design = design) +
  plot_annotation(tag_levels = list(c("a", "b", "c"))) &
  theme(plot.tag = element_text(face = "bold", size = 13, family = "nfont"))

ggsave(file.path(fig_dir, "fig2_discordance_combined.pdf"), fig2,
       width = 14, height = 12, dpi = DPI, bg = "white")
ggsave(file.path(fig_dir, "fig2_discordance_combined.png"), fig2,
       width = 14, height = 12, dpi = DPI, bg = "white")
cat("Fig 2 (discordance combined) saved\n")

cat("\nPhase 1 figures done.\n")