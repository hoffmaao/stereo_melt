import numpy as np
import config
from geospatial_operations import define_study_area, create_grid, get_expanded_bounds, query_bounds
from visualization import plot_velocity_data, plot_correction_data, plot_tidal_corrections
import os
import rasterio
import pandas as pd
import preprocessing

def main():
    # Define grid and load BedMachine data

    #x, y = create_grid(x_min, x_max, y_min, y_max, config.RES)
    #grid_x, grid_y = np.meshgrid(x, y)
    #corrections = Corrections(x, y)
    #print(len(corrections.x))
    #print(len(corrections.y))

    bm_file = config.BM_DIR + "BedMachineAntarctica-v3.nc"
    dtu22_file = config.MDT_DIR + "dtuuh22mdt.xyz"
    strip_df = pd.read_csv("./strips/beardmore_gz-2019-strips.csv")

    print("🔄 Processing DEM strips with alignment, IceSat-2 data, and tidal corrections...")
    preprocessing.correct_strips(
        strip_df=strip_df,
        bm_file=bm_file,
        mdt_file=dtu22_file,
        strips_dir=config.ASP_DIR+"asp_aligned/",
        output_dir=config.ASP_DIR+"processed/",
        grounded_shapefile=config.SHAPE_DIR+"GroundingLine_Antarctica_v02.shp",
        model="CATS2008"
    )


    plot_tidal_corrections(config.STRIPS_DIR + "processed/", plot_dir="plots")


if __name__ == "__main__":
    main()