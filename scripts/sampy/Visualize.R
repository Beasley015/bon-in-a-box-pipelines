library(rjson)
library(tidyverse)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)
library(viridis)

# Test inputs
# input <- list()
# input$pop_result <- "/home/ebeez/Documents/bon-in-a-box-pipelines/output/sampy/SamPy/5b3fee01fd5780cbedbdab8819404bb0/pop_result.csv"
# input$cell_values <- "/home/ebeez/Documents/bon-in-a-box-pipelines/output/sampy/SamPy/5b3fee01fd5780cbedbdab8819404bb0/vertices.csv"
# input$raw_grid <- "/home/ebeez/Documents/bon-in-a-box-pipelines/output/sampy/MakeGrid/26823a3cdff2b6139ee6a723e4eb4654/raw_grid.geojson"

# Load in data
input <- biab_inputs()

# Get csv files
pop_data <- read.csv(input$pop_result, header = F) %>%
  rename(Week=V1, rep=V2, pop_size=V3, infected = V4, contagious=V5) %>%
  mutate(Week = Week+1) #%>%
  # group_by(Week) %>%
  # summarise(pop_size=mean(pop_size), infected=mean(infected), 
  #           contagious=mean(contagious))

spat_data <- read.csv(input$cell_values)

grid <- st_read(dsn = input$raw_grid)

# Cases over time ----------------
pop_fig <- ggplot(data = pop_data, aes(x = Week, y = contagious))+
  # geom_line()+
  stat_summary(geom = "line", fun = mean) +
  stat_summary(geom = "ribbon", fun.data = mean_cl_normal, alpha = 0.3)+
  lims(x = c(53,max(pop_data$Week)), y = c(0,NA))+
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

rivers <- st_crop(rivers, grid)

# Get lakes
lakes <- ne_download(scale = 50L, type = "lakes", 
                     category = "physical")

sf_use_s2(FALSE)
lakes <- st_crop(lakes, grid) #issue here, may need to turn spherical off
sf_use_s2(TRUE)

# Summarize spatial data and add to the grid
for(i in 1:ncol(spat_data)){
  colnames(spat_data)[i] <- i
}

spat_data[spat_data > 1] <- 1

spat_data$rep <- rep(c(0,1), each = round(nrow(spat_data)/2), 
                     length.out=nrow(spat_data))
spat_data$week <- rep(c(1:max(pop_data$Week)), length.out=nrow(spat_data))

case.perc <- spat_data %>%
  pivot_longer(cols = -c(week,rep), names_to = 'cell', 
               values_to='rabies.occ') %>%
  group_by(rep, cell) %>%
  summarise(weeks.inf = sum(rabies.occ)) %>%
  mutate(weeks.inf = case_when(weeks.inf >= 1 ~ 1,
                                TRUE ~ 0)) %>%
  ungroup() %>%
  group_by(cell) %>%
  summarise(perc.sims = sum(weeks.inf)/n()) %>%
  mutate(cell = as.numeric(cell)) %>%
  arrange(cell)

grid.cases <- cbind(grid, case.perc[,2])

# Plot cases
spat_plot <- ggplot(data=grid.cases)+
  geom_sf(aes(fill = perc.sims))+
  geom_sf(data=rivers, fill = 'white', color = 'white')+
  geom_sf(data=lakes, fill='white')+
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
