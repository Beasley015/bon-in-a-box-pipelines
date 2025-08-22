## Load required packages

library("rjson")
library("dplyr")
library("gdalcubes")
library("sf")
sf_use_s2(FALSE)

source(paste(Sys.getenv("SCRIPT_LOCATION"), "/data/loadFromStacFun.R", sep = "/"))

input <- fromJSON(file=file.path(outputFolder, "input.json"))

# input$bbox <- c(-74.07131834227772, 42.56134435842028, 
#                 -71.40427444981111, 45.32488403343622)
input$proj <- "EPSG:4326"
input$collections_items <- c("cec_land_cover|cec_land_cover_2020",
                      "earthenv_topography|elevation_median")

gdalcubes_set_gdal_config("VSI_CACHE", "TRUE")
gdalcubes_set_gdal_config("GDAL_CACHEMAX","30%")
gdalcubes_set_gdal_config("VSI_CACHE_SIZE","10000000")
gdalcubes_set_gdal_config("GDAL_HTTP_MULTIPLEX","YES")
gdalcubes_set_gdal_config("GDAL_INGESTED_BYTES_AT_OPEN","32000")
gdalcubes_set_gdal_config("GDAL_DISABLE_READDIR_ON_OPEN","EMPTY_DIR")
gdalcubes_set_gdal_config("GDAL_HTTP_VERSION","2")
gdalcubes_set_gdal_config("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES","YES")
gdalcubes_set_gdal_config("CHECK_WITH_INVERT_PROJ","FALSE")
gdalcubes_set_gdal_config("GDAL_NUM_THREADS", 1)

gdalcubes::gdalcubes_options(parallel = 1)

bbox <- sf::st_bbox(c(xmin = input$bbox[1], ymin = input$bbox[2],
            xmax = input$bbox[3], ymax = input$bbox[4]), crs = sf::st_crs(input$proj))
weight_matrix <- NULL

if("resampling" %in% names(input)){
  resampling=input$resampling
}else{
  resampling="mode"
}

if("aggregation" %in% names(input)){
  aggregation=input$aggregation
}else{
  aggregation="first"
}

collections_items <- input$collections_items

if(!("stac_url" %in% names(input))){
  input$stac_url <- "https://stac.geobon.org"
}

cube_args <- list(stac_path = input$stac_url,
  limit = 5000,
  t0 = NULL,
  t1 = NULL,
  spatial.res = (0.00833*30)/1000, # 0.00833 degrees = 1000 m
  temporal.res = "P1D",
  aggregation = aggregation,
  resampling = resampling)

proj <- input$proj
as_list <- FALSE
raster_layers <- list()
nc_names <- c()
for (coll_it in collections_items){
  ci <- strsplit(coll_it, split = "|", fixed=TRUE)[[1]]
  cube_args_c <- append(cube_args, list(collections=ci[1],
                                        srs.cube = proj,
                                        bbox = bbox,
                                        layers=NULL,
                                        variable = NULL,
                                        ids=ci[2]))
  print(cube_args_c)
  pred <- do.call(load_cube, cube_args_c)
  
  nc_names <- cbind(nc_names,names(pred))
  if(names(pred)=='data'){
    pred <- rename_bands(pred, data=ci[2])
  }
  
  print(pred)
  
  raster_layers[[ci[2]]]=pred
}

output_raster_layers <- file.path(outputFolder)
layer_paths <- c()

for (i in 1:length(raster_layers)) {
  ff <- tempfile(pattern = paste0(names(raster_layers[i][[1]]),'_'))
  out<-gdalcubes::write_tif(raster_layers[i][[1]], dir = output_raster_layers, prefix=basename(ff),creation_options = list("COMPRESS" = "DEFLATE"), COG=TRUE, write_json_descr=TRUE)
  fp <- paste0(out[1])
  layer_paths <- cbind(layer_paths,fp)
}

output <- list("rasters" = layer_paths)
jsonData <- toJSON(output, indent=2)
write(jsonData, file.path(outputFolder,"output.json"))
