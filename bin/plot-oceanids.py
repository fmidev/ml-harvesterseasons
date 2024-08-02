import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

request = cimgt.OSM()

# Bounds: (lon_min, lon_max, lat_min, lat_max):
#extent = [24.9459,25.4459,59.95867,60.45867]
extent = [24.9459,25.4,59.95867,60.3] 

ax = plt.axes(projection=request.crs)
ax.set_extent(extent)
ax.add_image(request, 13)    # 13 = zoom level

# plot training locations and Helsinki Vuosaari harbor
plt.scatter(25.0000000, 60.2500000, transform=ccrs.PlateCarree(),color='blue',s=5)
plt.scatter(25.2500000, 60.2500000, transform=ccrs.PlateCarree(),color='blue',s=5)
plt.scatter(25.0000000, 60.0000000, transform=ccrs.PlateCarree(),color='blue',s=5)
plt.scatter(25.2500000, 60.0000000, transform=ccrs.PlateCarree(),color='blue',s=5)
plt.scatter(25.1959000, 60.2087600, transform=ccrs.PlateCarree(),color='red',s=5) # harbor

#extra points?
#plt.scatter(25.5000000, 60.0000000, transform=ccrs.PlateCarree(),color='blue',s=5)
#plt.scatter(25.5000000, 60.2500000, transform=ccrs.PlateCarree(),color='blue',s=5)

# save plot 
plt.savefig('HelsinkiVuosaariHarbor151028-2.jpg',dpi=1200)

'''import osmnx as ox
from shapely.geometry import box
import matplotlib.pyplot as plt
plt.savefig('world.jpg')

# Bounding box for given area (Helsinki Vuosaari harbor)
bounds = [24.9459,60.45867,25.4459,59.95867]

# Create a bounding box Polygon
bbox = box(*bounds)

# Retrieve buildings from the given area
buildings = ox.features_from_polygon(bbox, tags={"building": True})
buildings.plot()

plt.savefig('HelsinkiVuosaariHarbor151028.jpg')
'''
