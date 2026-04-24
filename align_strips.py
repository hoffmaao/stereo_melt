import numpy as np
from corrections import Corrections
from preprocessing import load_bedmachine, list_rema_v2_tiles, download_tiles, download_rema_strips, list_rema_strips, load_mat_file, load_geotiff, load_dtu10_mdt, create_velocity_stack, create_low_velocity_rock_polygons
import config
from geospatial_operations import define_study_area, create_grid, get_expanded_bounds, query_bounds
from visualization import plot_velocity_data, plot_correction_data, plot_tidal_corrections
import processing
import os
import rasterio

def main():
    # Define grid and load BedMachine data
    
    ice_shelf_polygon, ice_shelf_polygon_ll = query_bounds("./data/shapefiles/roosevelt_island_channels/Roosevelt_Island_channels.shp")
    x_min, y_min, x_max, y_max=get_expanded_bounds(ice_shelf_polygon).bounds
    print(ice_shelf_polygon_ll)

    x, y = create_grid(x_min, x_max, y_min, y_max, config.RES)
    grid_x, grid_y = np.meshgrid(x, y)
    corrections = Corrections(x, y)
    print(len(corrections.x))
    print(len(corrections.y))

    bm_file = config.BM_DIR + "BedMachineAntarctica-v3.nc"
    velocity_file= config.MAIN_DIR +"/data/NSIDC-0754/1996.01.01/antarctic_ice_vel_phase_map_v01.nc"


    # Load BedMachine data
    bm_data = load_bedmachine(bm_file)

    # Load MDT data
    dtu22_file = config.MDT_DIR + "dtuuh22mdt.xyz"
    dtu22_data = load_dtu10_mdt(dtu22_file)
    # Initialize Corrections structure
    print(dtu22_data)

    print(f"Lon shape: {dtu22_data['lon'].shape}")
    print(f"Lat shape: {dtu22_data['lat'].shape}")
    print(f"MDT shape: {dtu22_data['mdt'].shape}")

    #apply corrections
    corrections.interpolate_bedmachine(bm_data, grid_x, grid_y)
    corrections.interpolate_mdt(dtu22_data)
    # Process Strips with Tidal Corrections


    print("Listing REMA V2 tiles...")
    tile_paths=list_rema_v2_tiles(ice_shelf_polygon)
    print(f"Found {len(tile_paths)} tiles.")

    download_tiles(tile_paths,config.MOSAIC_DIR)

    print("Listing REMA V2 tiles...")



    strip_df=list_rema_strips(
        ice_shelf_polygon,
        (config.START_TIME,config.END_TIME)
    )
    print(strip_df)
    print(f"Found {len(strip_df)} tiles.")
    strip_df.to_csv("./strips/rossevelt_island_-2019-strips.csv")

    download_rema_strips(strip_df, config.STRIPS_DIR)

    print("🚀 Starting DEM Processing Pipeline...")

    low_velocity_polygons, rock_polygons, extent_polygon = create_low_velocity_rock_polygons(
        velocity_file,
        bm_file,
        ice_shelf_polygon,
        config.SHAPE_DIR,
        velocity_threshold=10)


    # Step 5: Process DEM Strips
    print("🔄 Processing DEM strips with alignment, IceSat-2 data, and tidal corrections...")
    processing.process_strips(
        strip_df=strip_df,
        corrections=corrections,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        strips_dir=config.STRIPS_DIR,
        output_dir=config.ASP_DIR,
        mosaic_dir=config.MOSAIC_DIR,
        low_velocity_shapefile=config.SHAPE_DIR+"low_velocity_polygons.shp",
        rock_shapefile=config.SHAPE_DIR+"rock_polygons.shp",
        grounded_shapefile=config.SHAPE_DIR+"GroundingLine_Antarctica_v02.shp",
        model="CATS2008"
    )

    
    print("✅ Processing pipeline completed successfully!")


    # Visualization
    #plot_velocity_data(velocity_stack)
    #plot_correction_data(
    #    grid_x, grid_y,
    #    geoid=corrections.geoid,
    #    elevation=corrections.z,
    #    mdt=corrections.mdt_correction,
    #    firn=corrections.firn
    #)
    #plot_tidal_corrections(config.STRIPS_DIR + "processed/", plot_dir="plots")

    print("Geoid correction (sample):", np.mean(corrections.geoid))
    print("Full column thickness (sample):", np.mean(corrections.thickness))
    print("Surface height above ellipsoid (sample):", np.mean(corrections.z))


if __name__ == "__main__":
    main()