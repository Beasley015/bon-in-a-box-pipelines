import sys
import subprocess
# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sampy-abm'])

import json
import numpy as np
import rasterio
import geojson

from sampy.addons.GIS_interface.geographic_grid import HexGrid

# Load inputs
inputs = biab_inputs()

# Load reprojected spatial data
land = rasterio.open(inputs['rasters'][0])

# Reproject bounding box to Albers Equal Area
bds = land.bounds

ul = (bds[3],bds[0])
ll = (bds[1],bds[0])
lr = (bds[1], bds[2])
ur = (bds[3],bds[2])

# Create hex grid
hex_grid = HexGrid.azimuthal_from_corners(ll, lr, ul, ur, cell_area=86.6) # 10km from edge to edge

output_path = output_folder+"/raw_grid.geojson"
hex_grid.create_geojson(output_path) 

# Save grid as geojson to temp folder, needed for land cover extractions
biab_output("raw_grid", output_path)