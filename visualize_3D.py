import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist

def load_and_prepare_data(filepath):
    """
    load points from csv and prepare them for visualization
    
    args:
        filepath (str): path to the csv file containing 3d coordinates
        
    returns:
        tuple: x, y, z coordinates as separate arrays and point ids
    """
    df = pd.read_csv(filepath)
    point_ids = range(1, len(df) + 1)
    return df['x'].values, df['y'].values, df['z'].values, point_ids

def create_interpolation_grid(x, y, resolution=100):
    """
    create a regular grid for interpolation
    
    args:
        x, y (np.array): coordinates of the points
        resolution (int): number of points for interpolation grid
        
    returns:
        tuple: xi, yi coordinates of the interpolation grid
    """
    xi = np.linspace(x.min(), x.max(), resolution)
    yi = np.linspace(y.min(), y.max(), resolution)
    return np.meshgrid(xi, yi)

def interpolate_surface(x, y, z, xi, yi):
    """
    interpolate z values on a regular grid
    
    args:
        x, y, z (np.array): coordinates of the points
        xi, yi (np.array): coordinates of the interpolation grid
        
    returns:
        np.array: interpolated z values on the regular grid
    """
    zi = griddata((x, y), z, (xi, yi), method='linear')
    
    return zi

def smooth_surface(zi, sigma=1.0):
    """
    apply gaussian smoothing to the surface
    
    args:
        zi (np.array): z values on the regular grid
        sigma (float): standard deviation for gaussian smoothing
        
    returns:
        np.array: smoothed z values
    """
    mask = np.isnan(zi)
    zi_filled = zi.copy()
    zi_filled[mask] = np.interp(np.flatnonzero(mask), 
                               np.flatnonzero(~mask), 
                               zi_filled[~mask])
    zi_smooth = gaussian_filter(zi_filled, sigma=sigma)
    zi_smooth[mask] = np.nan
    return zi_smooth

def create_surface_plot(xi, yi, zi, x=None, y=None, z=None, point_ids=None, title='', show_points=True):
    """
    create a single surface plot
    
    args:
        xi, yi (np.array): coordinates of the interpolation grid
        zi (np.array): z values on the regular grid
        x, y, z (np.array): coordinates of the original points
        point_ids (array): point identifiers for labeling
        title (str): title of the plot
        show_points (bool): whether to display the original points
        
    returns:
        matplotlib.figure.Figure: the created plot
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # plot the surface
    surf = ax.plot_surface(xi, yi, zi, cmap='terrain', alpha=0.8)
    
    if show_points and x is not None:
        scatter = ax.scatter(x, y, z, c='red', marker='o', s=30, label='Original Points')
        if point_ids is not None:
            for i, (xi, yi, zi, pid) in enumerate(zip(x, y, z, point_ids)):
                ax.text(xi, yi, zi, f' {pid}', size=8, zorder=1)
    
    # calculate ranges considering both surface and points
    x_min = np.nanmin(xi)
    x_max = np.nanmax(xi)
    y_min = np.nanmin(yi)
    y_max = np.nanmax(yi)
    z_min = np.nanmin(zi)
    z_max = np.nanmax(zi)
    
    # include original points in range calculation if they exist
    if show_points and x is not None:
        x_min = min(x_min, np.min(x))
        x_max = max(x_max, np.max(x))
        y_min = min(y_min, np.min(y))
        y_max = max(y_max, np.max(y))
        z_min = min(z_min, np.min(z))
        z_max = max(z_max, np.max(z))
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    # add a minimum range to prevent singular transformation
    min_range = 1e-6
    x_range = max(x_range, min_range)
    y_range = max(y_range, min_range)
    z_range = max(z_range, min_range)
    
    # use the largest range for all axes and add a buffer
    max_range = max(x_range, y_range, z_range) * 1.1  # add 10% buffer
    
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    z_mid = (z_max + z_min) / 2
    
    if np.isfinite(max_range) and np.isfinite(x_mid) and np.isfinite(y_mid) and np.isfinite(z_mid):
        ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
        ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
        ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)
    
    # ensure equal aspect ratio
    ax.set_box_aspect((1, 1, 1))
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate (Elevation)')
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Elevation (m)')
    ax.view_init(elev=30, azim=45)
    
    return fig

def save_mesh_and_points(x, y, z, xi, yi, zi, zi_smooth, point_ids):
    """
    save point cloud and both raw and smoothed meshes
    
    args:
        x, y, z (np.array): original point coordinates
        xi, yi (np.array): interpolation grid coordinates
        zi (np.array): raw surface z values
        zi_smooth (np.array): smoothed surface z values
        point_ids (array): point identifiers
    """
    # save point cloud with ids
    points = np.column_stack((point_ids, x, y, z))
    np.savetxt('point_cloud.xyz', points, fmt='%d %.6f %.6f %.6f', 
               header='id x y z', comments='')
    print(f"saved point cloud with {len(points)} points")
    
    # create grid points for raw mesh
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 100),
                        np.linspace(y.min(), y.max(), 100))
    zz = griddata((x, y), z, (xx, yy), method='linear')
    
    # save raw mesh (only valid points)
    valid_mask = ~np.isnan(zz)
    vertices = np.column_stack((xx[valid_mask], yy[valid_mask], zz[valid_mask]))
    np.savetxt('raw_mesh.xyz', vertices, fmt='%.6f %.6f %.6f',
               header='x y z', comments='')
    print(f"saved raw mesh with {len(vertices)} vertices")
    
    # create smoothed surface
    zz_smooth = smooth_surface(zz)
    
    # save smoothed mesh (only valid points)
    valid_mask = ~np.isnan(zz_smooth)
    vertices_smooth = np.column_stack((xx[valid_mask], 
                                     yy[valid_mask], 
                                     zz_smooth[valid_mask]))
    np.savetxt('smooth_mesh.xyz', vertices_smooth, fmt='%.6f %.6f %.6f',
               header='x y z', comments='')
    print(f"saved smoothed mesh with {len(vertices_smooth)} vertices")
    
    print("\nsaved files:")
    print("- 'point_cloud.xyz' (with point IDs)")
    print("- 'raw_mesh.xyz'")
    print("- 'smooth_mesh.xyz'")

def main():

    # load the data
    x, y, z, point_ids = load_and_prepare_data('3d_coords.csv')
    print(f"loaded {len(x)} points from 3d_coords.csv")
    
    # create interpolation grid and surfaces
    xi, yi = create_interpolation_grid(x, y)
    zi_raw = interpolate_surface(x, y, z, xi, yi)
    zi_smooth = smooth_surface(zi_raw)
    
    # save 3d data
    save_mesh_and_points(x, y, z, xi.ravel(), yi.ravel(), 
                        zi_raw.ravel(), zi_smooth.ravel(), point_ids)
    
    # create and save plots
    plots = [
        (zi_raw, True, 'raw_surface_with_points.png', 'Raw 3D Surface with Points'),
        (zi_smooth, True, 'smooth_surface_with_points.png', 'Smoothed 3D Surface with Points'),
        (zi_raw, False, 'raw_surface.png', 'Raw 3D Surface'),
        (zi_smooth, False, 'smooth_surface.png', 'Smoothed 3D Surface')
    ]
    
    for zi, show_points, filename, title in plots:
        fig = create_surface_plot(
            xi, yi, zi,
            x if show_points else None,
            y if show_points else None,
            z if show_points else None,
            point_ids if show_points else None,
            title=title,
            show_points=show_points
        )
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"saved plot as '{filename}'")
    
    plt.show()

if __name__ == "__main__":
    main()
