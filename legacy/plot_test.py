import numpy as np
from corrections import Corrections
from preprocessing import load_bedmachine, list_rema_v2_tiles, download_tiles, download_rema_strips, list_rema_strips, load_mat_file, load_geotiff, load_dtu10_mdt, create_velocity_stack, create_low_velocity_rock_polygons
import config
from geospatial_operations import define_study_area, create_grid, get_expanded_bounds, query_bounds
from visualization import plot_velocity_data, plot_correction_data, plot_tidal_corrections
import processing
import os
import rasterio
import config
import pandas as pd
import glob
import asp_binder_utils as asp_utils

file_path="/home/hoffmaao/stereo_melt/data/REMA/strips/SETSM_s2s041_WV01_20190130_1020010080035200_102001007E78D500_2m_lsf_seg1.tif"
icesat2_csv="/home/hoffmaao/stereo_melt/data/REMA/strips/ASP/icesat2_data/icesat2_2019-01-20.csv"
rock_csv = "/home/hoffmaao/stereo_melt/data/REMA/strips/ASP/rock_elevations_SETSM_s2s041_WV01_20190130_1020010080035200_102001007E78D500_2m_lsf_seg1.csv"
rock_df = pd.read_csv(rock_csv)
icesat2_df = pd.read_csv(icesat2_csv)
combined_df = pd.concat([rock_df, icesat2_df], ignore_index=True)
tsrs = 'EPSG:3031'

output_dir=os.path.join(config.ASP_DIR, "asp_alignment")
src_dem = file_path
initial_output_prefix = output_dir+'initial'
initial_elevation_difference_fn = glob.glob(initial_output_prefix+'-diff.csv')[0]


final_output_prefix = output_dir+'final'
final_elevation_difference_fn = glob.glob(final_output_prefix+'-diff.csv')[0]



asp_utils.plot_alignment_maps_altimetry(combined_df,src_dem,initial_elevation_difference_fn,final_elevation_difference_fn,plot_crs=tsrs,output_dir=output_dir)
