library(tidyverse)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)
library(scales)
library(showtext)

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

# Fig 4 - Choropleth: discordance by country
world <- ne_countries(scale = "medium", returnclass = "sf") |>
  dplyr::select(iso_a3, geometry)

map_data <- world |>
  left_join(country |> dplyr::select(country_code, discordance_any_pct, n_products),
            by = c("iso_a3" = "country_code"))

fig4 <- ggplot(map_data) +
  geom_sf(aes(fill = discordance_any_pct), colour = "grey70", linewidth = 0.1) +
  scale_fill_viridis_c(
    option = "plasma", na.value = "grey90", direction = -1,
    name = "Discordance rate (%)",
    labels = function(x) paste0(round(x, 0), "%")
  ) +
  coord_sf(crs = st_crs("+proj=robin")) +
  theme_void(base_family = "nfont", base_size = 10) +
  theme(
    plot.background    = element_rect(fill = "white", colour = NA),
    panel.background   = element_rect(fill = "white", colour = NA),
    legend.position    = c(0.15, 0.35),
    legend.key.height  = unit(0.8, "cm"),
    legend.key.width   = unit(0.3, "cm"),
    legend.title       = element_text(size = 9, face = "bold"),
    legend.text        = element_text(size = 8)
  )

ggsave(file.path(fig_dir, "fig3_choropleth.pdf"), fig4, width = 10, height = 5,
       dpi = DPI, bg = "white")
ggsave(file.path(fig_dir, "fig3_choropleth.png"), fig4, width = 10, height = 5,
       dpi = DPI, bg = "white")
cat("Fig 3 (choropleth) saved\n")