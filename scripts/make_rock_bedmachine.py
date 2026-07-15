"""Generate a SINGLE, continent-wide exposed-rock control shapefile from
BedMachine (mask==1) that every basin driver — and every era (IS2 / pre-IS2) —
points at, so rock ground control is identical across basins and time periods.

Rock outcrops are time-invariant; a single file removes the per-basin
stale-shapefile gap found in the 2026-07-06 audit
(project_rock_control_audit_2026_07_06). Output geometry-only, EPSG:3031,
features ≥0.25 km², simplified 100 m — matching the validated
beardmore_shelf_rock_bedmachine.shp. align_strip clips this to each strip
footprint, so continent-wide coverage costs only a small per-strip read.

Run:
  PROJ_DATA=$ENVPROJ PROJ_LIB=$ENVPROJ \
  /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python scripts/make_rock_bedmachine.py
"""
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes as rio_shapes
from rasterio.transform import Affine

BM = "/wd2/projects/stereo_melt/data/bedmachine/BedMachineAntarctica-v3.nc"
OUT = "/wd2/projects/stereo_melt/data/shapefiles/rock_bedmachine.shp"
MIN_AREA_M2 = 0.25e6      # ≥0.25 km²
SIMPLIFY_M = 100.0

ds = xr.open_dataset(BM)
X = ds["x"].values.astype("float64")   # ascending, +500
Y = ds["y"].values.astype("float64")   # descending, -500
mask = ds["mask"].values               # (y,x), 1==exposed rock (ice-free land)
res = 500.0
# transform: row0 = max y (Y descending), col0 = min x (X ascending)
transform = Affine(res, 0, X[0] - res / 2, 0, -res, Y[0] + res / 2)

rock = (mask == 1)
print(f"BedMachine mask==1 cells: {int(rock.sum())}  "
      f"(~{int(rock.sum())*res*res/1e6:.0f} km² raw)")

geoms = []
for geom, val in rio_shapes(rock.astype("uint8"), mask=rock, transform=transform):
    g = shape(geom)
    if g.area < MIN_AREA_M2:
        continue
    gs = g.simplify(SIMPLIFY_M, preserve_topology=True)
    if gs.is_empty:
        continue
    geoms.append(gs)

gdf = gpd.GeoDataFrame({"FID": range(len(geoms))}, geometry=geoms, crs="EPSG:3031")
gdf.to_file(OUT)
print(f"wrote {len(gdf)} features, {gdf.area.sum()/1e6:.0f} km² -> {OUT}")
