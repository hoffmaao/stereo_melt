import numpy as np
import matplotlib.pyplot as plt
import rioxarray as rxr
import geopandas as gpd
from shapely.geometry import box
from scipy.interpolate import griddata
import os
import xarray as xr


# Step 1: Create synthetic test dataset and export
def create_synthetic_dems_with_export(
    grid_size=800, resolution=2, velocity=100, melt_gaussian_std=50, dem_dir="./data/dem/",shape_dir="./data/shapefile/", melt_dir="./data/melt/", vel_dir="./data/velocity/"):
    """
    Create two synthetic DEMs with advection and melt, export as GeoTIFF, and save boundaries as Shapefile.

    Parameters:
    - grid_size: size of the grid (m)
    - resolution: grid spacing (m)
    - velocity: ice velocity (m/year)
    - melt_gaussian_std: standard deviation of Gaussian melt perturbation
    - output_dir: directory to save DEMs and boundary shapefile
    """
    #-1584783,-443159
    
    x = np.arange(-1584783, -1584783+ grid_size, resolution)
    y = np.arange(-443159, -443159 + grid_size, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Base DEM
    dem1 = np.zeros_like(X) + 100.0  # Flat surface at 100m elevation

    dem1_da = xr.DataArray(
        dem1,
        coords=[("y", y), ("x", x)],
        attrs={"description": "Synthetic DEM1"}
    )

    dem1_tif_path = os.path.join(dem_dir, "dem1.tif")
    dem1_da.rio.write_crs("EPSG:3031", inplace=True)
    dem1_da.rio.to_raster(dem1_tif_path)

    # Add Gaussian melt bump
    melt_bump = np.exp(-((X + 1584783 - grid_size / 2)**2 + (Y +443159 - grid_size / 2)**2) / (2 * melt_gaussian_std**2)) * -10
    
    melt_da = xr.DataArray(
        melt_bump,
        coords=[("y", y), ("x", x)],
        attrs={"description": "melt rate m/yr"}
    )

    melt_tif_path = os.path.join(melt_dir, "melt.tif")
    melt_da.rio.write_crs("EPSG:3031", inplace=True)
    melt_da.rio.to_raster(melt_tif_path)

    vx=np.zeros_like(X)+velocity
    vy=np.zeros_like(X)

    print(vx)


    vx_da = xr.DataArray(
        vx,
        coords=[("y", y), ("x", x)],
        attrs={"description": "vx m/yr"}
    )
    vy_da = xr.DataArray(
        vy,
        coords=[("y", y), ("x", x)],
        attrs={"description": "vy m/yr"}
    )

    vx_tif_path = os.path.join(vel_dir, "vx.tif")
    vx_da.rio.write_crs("EPSG:3031", inplace=True)
    vx_da.rio.to_raster(vx_tif_path)

    vy_tif_path = os.path.join(vel_dir, "vy.tif")
    vy_da.rio.write_crs("EPSG:3031", inplace=True)
    vy_da.rio.to_raster(vx_tif_path)



    dem1 += melt_bump
    
    # Advection
    x_shift = velocity  # Daily shift in meters
    y_shift = 0
    X2, Y2 = X - x_shift, Y - y_shift


    
    # DEM after advection and melt
    dem2 = griddata((X.ravel(), Y.ravel()), dem1.ravel(), (X2, Y2), method='linear')
    dem2 = np.nan_to_num(dem2, nan=100)  # Fill NaNs with baseline elevation
    
    # Add additional melt pattern
    #dem2 -= melt_bump * 1.1  # Apply slightly different melt

    # Save DEMs as GeoTIFF
    dem2_tif_path = os.path.join(dem_dir, "dem2.tif")

    # Convert to xarray for saving as GeoTIFF

    
    dem2_da = xr.DataArray(
        dem2,
        coords=[("y", y), ("x", x)],
        attrs={"description": "Synthetic DEM2"}
    )
    dem2_da.rio.write_crs("EPSG:3031", inplace=True)
    dem2_da.rio.to_raster(dem2_tif_path)
    
    print(f"DEM1 saved to {dem1_tif_path}")
    print(f"DEM2 saved to {dem2_tif_path}")
    
    # Step 2: Create and export boundary shapefile
    boundary = box(x.min(), y.min(), x.max(), y.max())
    gdf = gpd.GeoDataFrame({"name": ["DEM_Boundary"]}, geometry=[boundary], crs="EPSG:3031")
    boundary_shapefile_path = os.path.join(shape_dir, "dem_boundary.shp")
    gdf.to_file(boundary_shapefile_path)
    
    print(f"Boundary shapefile saved to {boundary_shapefile_path}")
    
    return dem1_tif_path, dem2_tif_path, melt_tif_path, vx_tif_path, boundary_shapefile_path

# Step 2: Visualization
def plot_dems(dem1_path, dem2_path):
    """
    Plot DEM1 and DEM2 from GeoTIFF files.
    """
    dem1 = rxr.open_rasterio(dem1_path).squeeze()
    dem2 = rxr.open_rasterio(dem2_path).squeeze()
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    im1 = axs[0].imshow(dem1, origin='lower', extent=[dem1.x.min(), dem1.x.max(), dem1.y.min(), dem1.y.max()])
    axs[0].set_title('DEM1')
    plt.colorbar(im1, ax=axs[0])
    
    im2 = axs[1].imshow(dem2, origin='lower', extent=[dem2.x.min(), dem2.x.max(), dem2.y.min(), dem2.y.max()])
    axs[1].set_title('DEM2')
    plt.colorbar(im2, ax=axs[1])
    
    plt.tight_layout()
    plt.show()

def plot_vel(melt_path, vel_path):
    """
    Plot DEM1 and DEM2 from GeoTIFF files.
    """
    vel = rxr.open_rasterio(vel_path).squeeze()
    melt = rxr.open_rasterio(melt_path).squeeze()
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    im1 = axs[0].imshow(vel, origin='lower', extent=[vel.x.min(), vel.x.max(), vel.y.min(), vel.y.max()])
    axs[0].set_title('vx')
    plt.colorbar(im1, ax=axs[0])
    
    im2 = axs[1].imshow(melt, origin='lower', extent=[melt.x.min(), melt.x.max(), melt.y.min(), melt.y.max()])
    axs[1].set_title('DEM2')
    plt.colorbar(im2, ax=axs[1])
    
    plt.tight_layout()
    plt.show()
# Main Execution
if __name__ == "__main__":
    dem1_path, dem2_path,  melt_tif_path, vx_tif_path, boundary_path = create_synthetic_dems_with_export()
    plot_dems(dem1_path, dem2_path)
    plot_vel(melt_tif_path, vx_tif_path)
