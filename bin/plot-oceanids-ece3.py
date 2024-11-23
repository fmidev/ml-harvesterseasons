import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
#from Sitia_023330_ece3 import *
#from Horta_002956_ece3 import *
#from Cadiz_000415_ece3 import *
#from Ploumanach_011245_ece3 import *
#from Aktio_023317_cordex_ncc_smhi import *
#from Bremerhaven_004885_ece3 import *
from Vuosaari_151028_cordex_cnrm_cnrm import *

# Load the dataset
df = pd.read_csv('/home/ubuntu/data/ML/training-data/OCEANIDS/'+fname)

# Extract the unique latitude and longitude values
lat_lon_columns = [col for col in df.columns if col.startswith('lat') or col.startswith('lon')]
lat_lon_df = df[lat_lon_columns].iloc[0]
#print(lat_lon_df)

# Extract latitudes and longitudes
lats = [lat_lon_df[f'lat-{i}'] for i in range(1, 5)]
lons = [lat_lon_df[f'lon-{i}'] for i in range(1, 5)]

# Add the main location coordinates
main_lat = lat
main_lon = lon

# Define the extent of the map using lats and lons
extent = [min(lons) - 0.25, max(lons) + 0.25, min(lats) - 0.25, max(lats) + 0.25]

# Create the map
osm_tiles = cimgt.OSM()
ax = plt.axes(projection=osm_tiles.crs)
ax.set_extent(extent)
ax.add_image(osm_tiles, 10) #Zoom level

# Plot the points
colors = ['yellow', 'green', 'blue', 'orange']
for i in range(4):
    plt.scatter(lons[i], lats[i], transform=ccrs.PlateCarree(), color=colors[i], s=15, label=f'{i+1} ({lons[i]}, {lats[i]})', zorder=2)

# Plot the main location
plt.scatter(main_lon, main_lat, transform=ccrs.PlateCarree(), color='red', s=15, label=f'{harbor} Harbor ({main_lon}, {main_lat})', zorder=3)

# Add legend and title
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), facecolor="lightblue")
plt.title(f'Training locations and {harbor} harbor')

# Adjust layout to ensure the legend doesn't get cut off
plt.tight_layout()

# Save the plot
plt.savefig(f'/home/ubuntu/ml-harvesterseasons/{harbor}-{FMISID}-cnrm-cnrm.jpg', dpi=1200, bbox_inches='tight')
plt.show()

