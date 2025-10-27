library(rjson)
library(tidyverse)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)
library(viridis)

# Test inputs
input <- list()
input$pop_result <- "/home/ebeez/Documents/bon-in-a-box-pipelines/output/sampy/SamPy/5b3fee01fd5780cbedbdab8819404bb0/pop_result.csv"
input$cell_values <- "/home/ebeez/Documents/bon-in-a-box-pipelines/output/sampy/SamPy/5b3fee01fd5780cbedbdab8819404bb0/vertices.csv"
input$raw_grid <- "/home/ebeez/Documents/bon-in-a-box-pipelines/output/sampy/MakeGrid/26823a3cdff2b6139ee6a723e4eb4654/raw_grid.geojson"

# Load in data
input <- biab_inputs()

# Get csv files
pop_data <- read.csv(input$pop_result, header = F) %>%
  rename(Week=V1, rep=V2, pop_size=V3, infected = V4, contagious=V5) %>%
  mutate(Week = Week+1) %>%
  group_by(Week) %>%
  summarise(pop_size=mean(pop_size), infected=mean(infected), 
            contagious=mean(contagious))

spat_data <- read.csv(input$cell_values)

grid <- st_read(dsn = input$raw_grid)

# Cases over time ----------------
pop_fig <- ggplot(data = pop_data, aes(x = Week, y = contagious))+
  geom_line()+
  lims(x = c(53,max(pop_data$Week)))+
  labs(x = "Week", y = "Cases")+
  theme_bw(base_size = 14)+
  theme(panel.grid = element_blank())

# Heat map of cases --------------
# Get political boundaries
poli.bounds <-  ne_download(scale = 50L, type = "states",
                             category = "cultural") %>%
  filter(admin %in% c("Canada", "United States of America"))

# Clip boundaries to simulation extent
poli.bounds <- st_crop(poli.bounds, grid)

# Get major rivers
rivers <- ne_download(scale = 10L, type = "rivers_lake_centerlines", 
                    category = "physical")

rivers <- st_crop(test, grid)

# Get lakes
lakes <- ne_download(scale = 50L, type = "lakes", 
                     category = "physical")

lakes <- st_crop(lakes, grid) #issue here, may need to turn spherical off

# Summarize spatial data and add to the grid
for(i in 1:ncol(spat_data)){
  colnames(spat_data)[i] <- paste("cell", i, sep = "")
}

total_cases <- colSums(spat_data)

grid.cases <- cbind(grid, total_cases)

# Plot cases
spat_plot <- ggplot(data=grid.cases)+
  geom_sf(aes(fill = total_cases))+
  scale_fill_viridis_c(option = "B", name = "Total Cases")+
  geom_sf(data=poli.bounds, linewidth=1.5, color = "white",
          fill = NA)+
  theme_bw(base_size=12)
  
# Save as output----------------
population_plot_path <- file.path(outputFolder, 
                              "population_plot.png")

spat_plot_path <- file.path(outputFolder, 
                            "SpatialCases.png")

map_path <- file.path(outputFolder, "case_map.geojson")

ggsave(population_plot_path, pop_fig, height=5, 
       width = 5, units="in")

ggsave(spat_plot_path, spat_plot, height = 8, width = 8,
       units = "in")

st_write(grid.cases, map_path)

# Save heat map here

biab_output("pop_fig", population_plot_path)
biab_output("disease_figure", spat_plot_path)
biab_output("disease_map", map_path)
