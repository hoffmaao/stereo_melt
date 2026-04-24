import matplotlib.pyplot as plt
import numpy as np
import os

def plot_velocity_data(velocity_stack, output_dir="plots"):
    """
    Plot velocity data for each year in the stack.

    Parameters:
    - velocity_stack (dict): Velocity data from `create_velocity_stack`.
    - x, y: Grid coordinates.
    - output_dir (str): Directory to save velocity plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    years = velocity_stack["Years"]

    for i, year in enumerate(years):
        vx = velocity_stack["vx"][:, :, i]
        vy = velocity_stack["vy"][:, :, i]
        speed = np.sqrt(vx**2 + vy**2)

        plt.figure(figsize=(10, 8))
        plt.contourf(velocity_stack['x'], velocity_stack['y'], speed, cmap="viridis", levels=50)
        plt.colorbar(label="Velocity (m/a)")
        plt.title(f"Velocity Data for {year}")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.savefig(os.path.join(output_dir, f"velocity_{year}.png"))
        plt.close()


def plot_tidal_corrections(output_dir, plot_dir="plots"):
    """
    Plot tidal corrections for each processed strip.
    
    Parameters:
    - output_dir (str): Directory containing processed strips.
    - plot_dir (str): Directory to save tidal correction plots.
    """
    os.makedirs(plot_dir, exist_ok=True)
    files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]

    for file in files:
        file_path = os.path.join(output_dir, file)
        data = np.load(file_path, allow_pickle=True)
        
        if 'tide_correction' in data:
            x = data['x']
            y = data['y']
            tide_correction = data['tide_correction']
            
            plt.figure(figsize=(12, 9))
            contour = plt.contourf(x, y, tide_correction, cmap='coolwarm', levels=50)
            plt.colorbar(contour, label='Tidal Correction (m)')
            plt.title(f"Tidal Correction for {file}")
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.savefig(os.path.join(plot_dir, f"tide_correction_{os.path.splitext(file)[0]}.png"), dpi=300)
            plt.close()
            
            print(f"[INFO] Tidal correction plot saved for {file}")
        else:
            print(f"[WARNING] No tidal correction data in {file}")



def plot_correction_data(grid_x, grid_y, geoid, elevation, mdt, firn, output_dir="plots"):
    """
    Plot correction data (elevation, geoid, MDT, firn).

    Parameters:
    - grid_x, grid_y: Grid coordinates.
    - geoid: Geoid corrections.
    - elevation: Surface elevation data.
    - mdt: MDT corrections.
    - firn: Firn corrections.
    - output_dir (str): Directory to save correction plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    datasets = {
        "Geoid Correction": geoid,
        "Elevation": elevation,
        "MDT Correction": mdt,
        "Firn Correction": firn
    }

    for name, data in datasets.items():
        plt.figure(figsize=(10, 8))
        plt.contourf(grid_x, grid_y, data, cmap="viridis", levels=50)
        plt.colorbar(label=name)
        plt.title(name)
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.savefig(os.path.join(output_dir, f"{name.replace(' ', '_').lower()}.png"))
        plt.close()