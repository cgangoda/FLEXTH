import rasterio
import matplotlib.pyplot as plt
import numpy as np

# File paths for the two rasters
file_path1 = '/../.../output/WL_method_A_Smax_0.05_Nmax_100_a_2_Dmax_10_A12_100.tif'
file_path2 = '/../.../output/WD_method_A_Smax_0.05_Nmax_100_a_2_Dmax_10_A12_100.tif'

# Read the first .tif file (Water Level)
with rasterio.open(file_path1) as src1:
    data1 = src1.read(1)

# Read the second .tif file (Water Depth)
with rasterio.open(file_path2) as src2:
    data2 = src2.read(1)

# Apply a mask for NoData values (set to NaN)
data1 = np.where(data1 > 0, data1, np.nan)
data2 = np.where(data2 > 0, data2, np.nan)

# Plotting with different color schemes
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 2 subplots for 2 files

# Water Level (file_path1) using 'jet' colormap
im1 = axes[0].imshow(data1, cmap='jet', vmin=np.nanmin(data1), vmax=np.nanmax(data1))
axes[0].set_title('Water Level')
axes[0].set_xlabel('Easting (°)')
axes[0].set_ylabel('Northing (°)')
plt.colorbar(im1, ax=axes[0], label='Water Level (m)')  # Add unit 'm'

# Water Depth (file_path2) using 'coolwarm' colormap
im2 = axes[1].imshow(data2, cmap='coolwarm', vmin=np.nanmin(data2), vmax=np.nanmax(data2))
axes[1].set_title('Water Depth')
axes[1].set_xlabel('Easting (°)')
plt.colorbar(im2, ax=axes[1], label='Water Depth (cm)')  # Add unit 'cm'

plt.tight_layout()
plt.show()
