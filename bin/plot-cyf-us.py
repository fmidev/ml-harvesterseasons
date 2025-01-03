import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import osmnx as ox
#from shapely.geometry import box
#import matplotlib.pyplot as plt
#plt.savefig('world.jpg')

# Bounding box for given area (Helsinki Vuosaari harbor)
#bounds = [24.9459,60.45867,25.4459,59.95867]

# Create a bounding box for area
area=ox.geocode_to_gdf('Ternopil Oblast, Ukraine')

request = cimgt.OSM()

# Bounds: (lon_min, lon_max, lat_min, lat_max):
#extent = [24.9459,25.4459,59.95867,60.45867]
extent = [22,41,44,53] 

ax = plt.axes(projection=request.crs)
ax.set_extent(extent)
ax.add_image(request, 13)    # 13 = zoom level

# plot training locations 
plt.scatter(26.1, 48.6, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.2, 48.6, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.3, 48.6, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.7, 48.7, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.8, 48.7, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.9, 48.7, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.0, 48.7, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.1, 48.7, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.2, 48.7, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.5, 48.8, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.6, 48.8, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.7, 48.8, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.8, 48.8, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.9, 48.8, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.0, 48.8, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.1, 48.8, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.2, 48.8, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.2, 48.9, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.3, 48.9, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.4, 48.9, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.5, 48.9, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.6, 48.9, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.7, 48.9, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.8, 48.9, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.9, 48.9, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.0, 48.9, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.1, 48.9, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.2, 48.9, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.0, 49.0, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.1, 49.0, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.2, 49.0, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.3, 49.0, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.4, 49.0, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.5, 49.0, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.6, 49.0, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.7, 49.0, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.8, 49.0, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.9, 49.0, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.0, 49.0, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.1, 49.0, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.2, 49.0, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.0, 49.1, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.1, 49.1, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.2, 49.1, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.3, 49.1, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.4, 49.1, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.5, 49.1, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.6, 49.1, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.7, 49.1, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.8, 49.1, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.9, 49.1, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.0, 49.1, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.1, 49.1, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.2, 49.1, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(24.9, 49.2, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.0, 49.2, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.1, 49.2, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.2, 49.2, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.3, 49.2, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.4, 49.2, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.5, 49.2, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.6, 49.2, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.7, 49.2, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.8, 49.2, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.9, 49.2, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.0, 49.2, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.1, 49.2, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.2, 49.2, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(24.8, 49.3, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(24.9, 49.3, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.0, 49.3, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.1, 49.3, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.2, 49.3, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.3, 49.3, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.4, 49.3, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.5, 49.3, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.6, 49.3, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.7, 49.3, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.8, 49.3, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.9, 49.3, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.0, 49.3, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.1, 49.3, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.2, 49.3, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(24.8, 49.4, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(24.9, 49.4, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.0, 49.4, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.1, 49.4, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.2, 49.4, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.3, 49.4, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.4, 49.4, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.5, 49.4, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.6, 49.4, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.7, 49.4, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.8, 49.4, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.9, 49.4, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.0, 49.4, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.1, 49.4, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.2, 49.4, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(24.8, 49.5, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(24.9, 49.5, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.0, 49.5, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.1, 49.5, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.2, 49.5, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.3, 49.5, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.4, 49.5, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.5, 49.5, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.6, 49.5, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.7, 49.5, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.8, 49.5, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.9, 49.5, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.0, 49.5, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.1, 49.5, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(24.8, 49.6, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(24.9, 49.6, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.0, 49.6, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.1, 49.6, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.2, 49.6, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.3, 49.6, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.4, 49.6, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.5, 49.6, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.6, 49.6, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.7, 49.6, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.8, 49.6, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.9, 49.6, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.0, 49.6, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.1, 49.6, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.2, 49.6, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.0, 49.7, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.1, 49.7, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.2, 49.7, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.3, 49.7, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.4, 49.7, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.5, 49.7, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.6, 49.7, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.7, 49.7, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.8, 49.7, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.9, 49.7, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.0, 49.7, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.1, 49.7, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.2, 49.7, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.2, 49.8, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.3, 49.8, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.4, 49.8, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.5, 49.8, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.6, 49.8, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.7, 49.8, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.8, 49.8, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.9, 49.8, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.0, 49.8, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.1, 49.8, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.2, 49.8, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.5, 49.9, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.6, 49.9, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.7, 49.9, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.8, 49.9, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.9, 49.9, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.0, 49.9, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.1, 49.9, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.4, 50.0, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.5, 50.0, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.6, 50.0, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.7, 50.0, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.8, 50.0, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.9, 50.0, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.0, 50.0, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.1, 50.0, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.2, 50.0, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.5, 50.1, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.6, 50.1, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.7, 50.1, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.8, 50.1, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.9, 50.1, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.0, 50.1, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.1, 50.1, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.2, 50.1, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.3, 50.1, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(25.9, 50.2, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.0, 50.2, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.1, 50.2, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
plt.scatter(26.2, 50.2, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),facecolor="lightblue")
#extra points?
#plt.scatter(25.5, 60., transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")
#plt.scatter(25.5, 60.250000, transform=ccrs.PlateCarree(),color='blue',s=5,marker="+")

# save plot 
#plt.savefig('HelsinkiVuosaariHarbor151028-3.jpg',dpi=1200)


# Retrieve buildings from the given area
#buildings = ox.features_from_place('Ternopil Oblast, Ukraine', tags={"building": True})
#buildings.plot()

plt.savefig('Ternopilska-oblast.jpg',dpi=600)
