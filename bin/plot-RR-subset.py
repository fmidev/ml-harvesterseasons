import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box, Polygon
import matplotlib.pyplot as plt

# Load the CSV data
data_path = '/home/ubuntu/data/ML/training-data/RR/eobs/eobsStaidInfo.csv'  # Your specified path
df = pd.read_csv(data_path)

# Clean up column names
df.columns = df.columns.str.strip()  # Remove any extra spaces
df = df[['STAID', 'LAT', 'LON']]  # Keep only necessary columns

# Convert the DataFrame to a GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['LON'], df['LAT']), crs="EPSG:4326")

# Define a bounding box for Europe
europe_bounds = gpd.GeoDataFrame(geometry=[box(-30, 25, 50, 75)], crs="EPSG:4326")

# Filter points within Europe
gdf = gdf[gdf.within(europe_bounds.unary_union)]

# Create regions or bins to stratify the data
n_bins = 100  # Number of bins to create; adjust for finer control
gdf['bin_x'] = pd.cut(gdf.geometry.x, bins=n_bins)
gdf['bin_y'] = pd.cut(gdf.geometry.y, bins=n_bins)

# Function to sample points from each bin
def stratified_sample(gdf, min_points_per_bin=50):
    sampled_indices = []
    
    for bin_x in gdf['bin_x'].unique():
        bin_data = gdf[gdf['bin_x'] == bin_x]
        # Ensure that we sample at least min_points_per_bin points if available
        sample_size = min(len(bin_data), min_points_per_bin)
        sampled_indices.extend(bin_data.sample(n=sample_size, random_state=42).index)
        
    return gdf.loc[sampled_indices]

# Perform stratified sampling
reduced_gdf = stratified_sample(gdf, min_points_per_bin=50)

# Ensure we have at least 5000 points, adjust sampling if necessary
if len(reduced_gdf) < 5000:
    additional_needed = 5000 - len(reduced_gdf)
    # Randomly sample additional points from the entire dataset
    additional_samples = gdf.sample(n=additional_needed, random_state=42)
    reduced_gdf = pd.concat([reduced_gdf, additional_samples]).drop_duplicates().reset_index(drop=True)

# Count the number of points
original_count = len(gdf)
reduced_count = len(reduced_gdf)
print(f"Original count: {original_count}, Reduced count: {reduced_count}")

# Save the reduced subset to a new CSV file
reduced_gdf.to_csv('eobsStaidInfo_reduced.csv', index=False)

# Optional: Plot the results to visualize the distribution
fig, ax = plt.subplots(figsize=(10, 10))

# Manually define a polygon for Europe to plot
europe_coords = [
    (-30, 25), (50, 25), (50, 75), (-30, 75), (-30, 25)  # Approximate bounding box
]

europe_polygon = Polygon(europe_coords)
ax.add_patch(plt.Polygon(list(europe_polygon.exterior.coords), color='lightgrey', edgecolor='black'))

# Plot the reduced subset of points
reduced_gdf.plot(ax=ax, color='blue', marker='o', markersize=5, label="Representative Station")

# Customize plot
plt.title("Reduced Representative Stations across Europe")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.xlim(-30, 50)
plt.ylim(25, 75)
plt.legend()
plt.grid(True)

# Save plot as an image
plt.savefig('representative_stations_map.png', dpi=300)
plt.show()
