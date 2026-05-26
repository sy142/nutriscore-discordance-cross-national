library(tidyverse)
library(GGally)
library(showtext)
library(viridis)
library(scales)

font_add("nfont",
         regular = "C:/Windows/Fonts/arial.ttf",
         bold    = "C:/Windows/Fonts/arialbd.ttf",
         italic  = "C:/Windows/Fonts/ariali.ttf")
showtext_auto()
showtext_opts(dpi = 600)

setwd("C:/Users/Salim/Desktop/makaleler/Sedat Arslan ML Large")

country <- read_csv("analysis_data/country_level.csv", show_col_types = FALSE)

fig_dir <- "figures"
DPI     <- 600

# Variables
scatter_vars <- c("discordance_any_pct", "mean_ns_score", "pct_nova4",
                  "ncd_mortality_pct", "diabetes_pct", "life_expectancy",
                  "overweight_adult_pct", "gdp_per_capita_ppp")

scatter_labels <- c("Discordance\n(%)", "Mean NS\nscore", "NOVA 4\n(%)",
                    "NCD\nmortality", "Diabetes\n(%)", "Life\nexpectancy",
                    "Overweight\n(%)", "GDP\n(PPP)")

scatter_data <- country |>
  dplyr::select(all_of(scatter_vars)) |>
  drop_na()

names(scatter_data) <- scatter_labels

# Custom upper: colored tile + BLACK correlation + significance stars
upper_cor <- function(data, mapping, ...) {
  x <- GGally::eval_data_col(data, mapping$x)
  y <- GGally::eval_data_col(data, mapping$y)
  ct <- cor.test(x, y, method = "spearman", exact = FALSE)
  r  <- ct$estimate
  p  <- ct$p.value
  
  stars <- ifelse(p < 0.001, "***", ifelse(p < 0.01, "**", ifelse(p < 0.05, "*", "")))
  
  # Color by correlation strength
  abs_r <- abs(r)
  bg_alpha <- abs_r * 0.5 + 0.05
  bg_col <- ifelse(r > 0, "#E74C3C", "#2E86C1")
  
  ggplot(data.frame(x = 0.5, y = 0.5), aes(x, y)) +
    annotate("rect", xmin = 0, xmax = 1, ymin = 0, ymax = 1,
             fill = bg_col, alpha = bg_alpha) +
    annotate("text", x = 0.5, y = 0.58,
             label = sprintf("%.2f", r),
             size = 5, fontface = "bold", family = "nfont",
             colour = "black") +
    annotate("text", x = 0.5, y = 0.35,
             label = stars,
             size = 6, fontface = "bold", family = "nfont",
             colour = "black") +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    theme_void()
}

# Custom lower: scatter + filled 2D density contour (larger axis text + comma format)
lower_density <- function(data, mapping, ...) {
  ggplot(data, mapping) +
    stat_density_2d(aes(fill = after_stat(level)),
                    geom = "polygon", alpha = 0.4, show.legend = FALSE) +
    scale_fill_viridis_c(option = "mako", direction = -1) +
    geom_point(alpha = 0.4, size = 0.6, colour = "#2C3E50") +
    geom_smooth(method = "lm", se = FALSE, linewidth = 0.5,
                colour = "#E74C3C", linetype = "dashed") +
    scale_x_continuous(labels = scales::label_comma()) +
    scale_y_continuous(labels = scales::label_comma()) +
    theme_minimal(base_family = "nfont", base_size = 8) +
    theme(
      panel.grid = element_line(linewidth = 0.15, colour = "grey92"),
      panel.background = element_rect(fill = "grey99", colour = NA),
      axis.text = element_text(size = 7, face = "bold", family = "nfont")
    )
}

# Custom diagonal: luminous density with gradient fill (comma format on x-axis)
diag_density <- function(data, mapping, ...) {
  ggplot(data, mapping) +
    geom_density(aes(y = after_stat(scaled)),
                 fill = "#1A5276", alpha = 0.15, colour = NA) +
    geom_density(aes(y = after_stat(scaled)),
                 fill = "#2E86C1", alpha = 0.25, colour = NA) +
    geom_density(aes(y = after_stat(scaled)),
                 fill = "#5DADE2", alpha = 0.3, colour = NA) +
    geom_density(aes(y = after_stat(scaled)),
                 colour = "#1A5276", linewidth = 0.8, fill = NA) +
    scale_x_continuous(labels = scales::label_comma()) +
    theme_minimal(base_family = "nfont", base_size = 8) +
    theme(
      panel.grid       = element_blank(),
      panel.background = element_rect(fill = "grey99", colour = NA),
      axis.text.y      = element_blank()
    )
}

# Build the matrix
sfig1 <- ggpairs(
  scatter_data,
  upper = list(continuous = upper_cor),
  lower = list(continuous = lower_density),
  diag  = list(continuous = diag_density)
) +
  theme_minimal(base_family = "nfont", base_size = 9) +
  theme(
    strip.text       = element_text(size = 8, face = "bold", family = "nfont"),
    strip.background = element_rect(fill = "grey95", colour = NA),
    axis.text        = element_text(size = 7, face = "bold", family = "nfont"),
    panel.border     = element_rect(colour = "grey85", fill = NA, linewidth = 0.3),
    plot.background  = element_rect(fill = "white", colour = NA)
  )

ggsave(file.path(fig_dir, "sfig1_scatter_matrix.pdf"), sfig1,
       width = 13, height = 13, dpi = DPI, bg = "white")
ggsave(file.path(fig_dir, "sfig1_scatter_matrix.png"), sfig1,
       width = 13, height = 13, dpi = DPI, bg = "white")
cat("S Fig 1 (scatter matrix) saved\n")
cat("Supplementary R figures complete.\n")