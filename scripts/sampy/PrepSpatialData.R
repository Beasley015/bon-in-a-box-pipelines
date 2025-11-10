library(rjson)
library(sf)
library(terra)
library(exactextractr)
library(tidyverse)
library(units)

# Load inputs
inputs <- fromJSON(file=file.path(outputFolder, "input.json"))

grid <- st_read(inputs$raw_grid)

# K values --------------

# Load land cover raster
K.grid <- grid
land_cover <- rast(inputs$rasters[1])

land_cover <- round(land_cover, digits=0)

# Change crs of raster, if needed
print("Reprojecting land cover raster...")
K.grid <- st_transform(K.grid, crs=crs(land_cover))

# Get % land cover in each grid cell
print("Extracting land cover classes...")
perc_land_cover <- exact_extract(x=land_cover$cec_land_cover_2020,
                                 y=K.grid,fun='frac') %>%
  mutate(id = K.grid$id)

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

K.grid$k <- wmeans

# Special combos adjustment
ag_decid <- which(perc_land_cover$cropland > 0.2 &
                    perc_land_cover$deciduous > 0.2)
K.grid$k[ag_decid] <- 18

decid_urban <- which(perc_land_cover$deciduous > 0.2 &
                       perc_land_cover$urban > 0.2)

K.grid$k[decid_urban] <- 21

ag_decid_urban <- which(perc_land_cover$cropland > 0.2 &
                            perc_land_cover$deciduous > 0.2 &
                            perc_land_cover$urban > 0.2)
K.grid$k[ag_decid_urban] <- 27

# Elevation adjustment
elev <- rast(inputs$rasters[2])
elev <- project(elev, crs(K.grid))

elev.per.hex <- exact_extract(elev, grid, fun='mean') 
elcell <- which(elev.per.hex >= 500)

K.grid$k[elcell] <- 1

# Adjust for cell area in km
ar <- st_area(K.grid[1,])
ar <- set_units(ar, km^2) 

K.grid$k <- K.grid$k*ar
K.grid$k <- as.numeric(K.grid$k)

# Cases ----------------
if(is.null(inputs$cases) == F){
  # Read in cases and reformat
  cases <- read.csv(file=inputs$cases)
  
  case.sf <- st_as_sf(cases, coords = c(2,1))
  
  st_crs(case.sf) <- 4326
  
  # split cases by year
  cases.list <- split(case.sf, ~year)
  
  inters.list <- lapply(cases.list,st_intersects, grid)
  inters.list <- lapply(inters.list, function(x) do.call(c,x))

  case.grid <- grid
  case.grid[names(inters.list)] <- 0
  
  for(i in 1:length(inters.list)){
    case.grid[inters.list[[i]], 2+i] <- 1
  }
  
} else {
  
  case.grid <- grid
  
}

# Vaccination -------------
if(is.null(inputs$vaccination) == F){
  # Read in cases and reformat
  vax <- read_sf(dsn=inputs$vaccination)
  
  st_crs(vax) <- 4326
  
  # Merge polygons
  sf_use_s2(FALSE)
  vaxp <- st_combine(vax)
  
  # Get intersecting grid cells
  inter.vax.list <- st_intersects(vaxp, grid)
  
  # Add to grid
  vax.grid <- grid
  vax.grid$vax <- 0
  vax.grid$vax[unlist(inter.vax.list)] <- 1
  
  sf_use_s2(TRUE)
  
} else {
  
  vax.grid <- grid
  
}

# Save as geojson ----------------
K_grid <- file.path(outputFolder, "spatial_features.geojson")
st_write(K.grid, dsn=K_grid, append = F)

case_grid <- file.path(outputFolder, "cases.geojson")
st_write(case.grid, dsn=case_grid, append=F)

vax_grid <- file.path(outputFolder, "vax.geojson")
st_write(vax.grid, dsn = vax_grid, append=F)

biab_output("K_grid", K_grid)
biab_output("case_grid", case_grid)
biab_output("vax_grid", vax_grid)
