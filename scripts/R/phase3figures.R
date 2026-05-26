library(tidyverse)
library(ggrepel)
library(patchwork)
library(cowplot)
library(scales)
library(showtext)

font_add("nfont",
         regular = "C:/Windows/Fonts/arial.ttf",
         bold    = "C:/Windows/Fonts/arialbd.ttf",
         italic  = "C:/Windows/Fonts/ariali.ttf")
showtext_auto()
showtext_opts(dpi = 600)

setwd("C:/Users/Salim/Desktop/makaleler/Sedat Arslan ML Large")

df <- read_csv("analysis_data/country_level.csv", show_col_types = FALSE)

fig_dir <- "figures"
DPI     <- 600

df <- df |>
  mutate(income_group = case_when(
    gdp_per_capita_ppp >= 40000 ~ "HIC",
    gdp_per_capita_ppp >= 12000 ~ "UMIC",
    TRUE                        ~ "LMIC"
  ))

theme_nf <- theme_minimal(base_family = "nfont", base_size = 10) +
  theme(
    panel.grid.minor   = element_blank(),
    panel.grid.major   = element_line(linewidth = 0.3, colour = "grey90"),
    legend.position    = "bottom",
    axis.title         = element_text(size = 10),
    axis.text          = element_text(size = 9),
    plot.background    = element_rect(fill = "white", colour = NA),
    panel.background   = element_rect(fill = "white", colour = NA)
  )

pal_income <- c(HIC = "#2E75B6", UMIC = "#F0AD4E", LMIC = "#D9534F")

scatter_panel <- function(data, yvar, ylab) {
  sub <- data |>
    drop_na(discordance_any_pct, all_of(yvar), population, income_group)
  
  ggplot(sub, aes(x = discordance_any_pct, y = .data[[yvar]])) +
    geom_smooth(method = "lm", se = TRUE, colour = "grey50",
                linewidth = 0.6, fill = "grey88", alpha = 0.5) +
    geom_point(aes(size = population / 1e6, fill = income_group),
               shape = 21, alpha = 0.8, stroke = 0.4, colour = "grey30") +
    geom_text_repel(
      data = sub |> slice_max(n_products, n = 10),
      aes(label = country_code),
      family = "nfont", size = 3.2, fontface = "bold", colour = "grey20",
      segment.colour = "grey40", segment.size = 0.4, segment.linetype = 1,
      min.segment.length = 0, box.padding = 0.6, point.padding = 0.4,
      max.overlaps = 15, seed = 42
    ) +
    scale_size_continuous(range = c(1.5, 11), name = "Population (M)",
                          labels = comma, guide = guide_legend(order = 2)) +
    scale_fill_manual(values = pal_income, name = "Income group",
                      guide = guide_legend(order = 1, override.aes = list(size = 4))) +
    labs(x = "NutriScore-NOVA discordance (%)", y = ylab) +
    theme_nf +
    theme(legend.position = "none")
}

p_a <- scatter_panel(df, "ncd_mortality_pct",    "NCD mortality 30-70 (%)")
p_b <- scatter_panel(df, "diabetes_pct",         "Diabetes prevalence (%)")
p_c <- scatter_panel(df, "life_expectancy",      "Life expectancy (years)")
p_d <- scatter_panel(df, "overweight_adult_pct", "Overweight adults (%)")

legend_data <- df |>
  drop_na(discordance_any_pct, ncd_mortality_pct, population, income_group)

legend_plot <- ggplot(legend_data,
                      aes(x = discordance_any_pct, y = ncd_mortality_pct)) +
  geom_point(aes(size = population / 1e6, fill = income_group),
             shape = 21, alpha = 0.8, stroke = 0.4, colour = "grey30") +
  scale_size_continuous(range = c(1.5, 11), name = "Population (M)",
                        labels = comma, guide = guide_legend(order = 2)) +
  scale_fill_manual(values = pal_income, name = "Income group",
                    guide = guide_legend(order = 1, override.aes = list(size = 4))) +
  theme_nf +
  theme(legend.position = "bottom", legend.box = "horizontal")

shared_legend <- get_legend(legend_plot)

panels <- (p_a + p_b) / (p_c + p_d) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(face = "bold", size = 13, family = "nfont"))

fig5 <- plot_grid(panels, shared_legend, ncol = 1, rel_heights = c(1, 0.07))

ggsave(file.path(fig_dir, "fig4_health_scatter.pdf"), fig5,
       width = 10, height = 10, dpi = DPI, bg = "white")
ggsave(file.path(fig_dir, "fig4_health_scatter.png"), fig5,
       width = 10, height = 10, dpi = DPI, bg = "white")
cat("Fig 4 (health scatter) saved\n")