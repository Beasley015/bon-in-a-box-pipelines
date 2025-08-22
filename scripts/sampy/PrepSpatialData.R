library(rjson)
library(sf)
library(terra)
library(exactextractr)
library(tidyverse)
library(units)

# Load inputs
inputs <- fromJSON(file=file.path(outputFolder, "input.json"))

# K values --------------

# Load hexagonal grid and land cover raster
grid <- st_read(inputs$raw_grid)
land_cover <- rast(inputs$rasters[1])

land_cover <- round(land_cover, digits=0)

# Change crs of raster, if needed
print("Reprojecting land cover raster...")
grid <- st_transform(grid, crs=crs(land_cover))

# Get % land cover in each grid cell
print("Extracting land cover classes...")
perc_land_cover <- exact_extract(x=land_cover$cec_land_cover_2020,
                                 y=grid,fun='frac') %>%
  mutate(id = grid$id)

# Rename columns
colnames(perc_land_cover) <- c("conifer", "deciduous",
                               "mixed_forest", "shrubland",
                               "grassland", "wetland",
                               "cropland", "barren", "urban",
                               "water", "id")

# Define carrying capacities based on Slate et al.
k_per_km <- c(2.9, 3.4, 3.2, 3.4, 4, 3.4, 12.5, 3.4, 18, 0)

# Calculate weighted means
wmeans <- apply(perc_land_cover[,-11], 1, 
                function(x) weighted.mean(k_per_km, w = x))

grid$k <- wmeans

# Special combos adjustment
ag_decid <- which(perc_land_cover$cropland > 0.2 &
                    perc_land_cover$deciduous > 0.2)
grid$k[ag_decid] <- 18

decid_urban <- which(perc_land_cover$deciduous > 0.2 &
                       perc_land_cover$urban > 0.2)

grid$k[decid_urban] <- 21

ag_decid_urban <- which(perc_land_cover$cropland > 0.2 &
                            perc_land_cover$deciduous > 0.2 &
                            perc_land_cover$urban > 0.2)
grid$k[ag_decid_urban] <- 27

# Elevation adjustment
elev <- rast(inputs$rasters[2])
elev <- project(elev, crs(grid))

elev.per.hex <- exact_extract(elev, grid, fun='mean') 
elcell <- which(elev.per.hex >= 500)

grid$k[elcell] <- 1

# Adjust for cell area in km
ar <- st_area(grid[1,])
ar <- set_units(ar, km^2)

grid$k <- grid$k*ar
grid$k <- as.numeric(grid$k)

# Feature to come: Management -------------

# Save as geojson
K_grid <- file.path(outputFolder, "spatial_features.geojson")

st_write(grid, dsn=K_grid, append = F)

biab_output("K_grid", K_grid)
