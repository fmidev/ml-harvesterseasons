import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import requests,json
#from Raahe_101785 import *
#from Vuosaari_151028 import *
#from Rauma_101061 import *
#from Malaga_000231 import *
from Malaga_ece3 import *

request = cimgt.OSM()

source='desm.harvesterseasons.com:8080' # server for timeseries query
bbox_list = bbox.split(',')
extent = [float(bbox_list[0]),float(bbox_list[2]),float(bbox_list[3]),float(bbox_list[1])] 

print(bbox)
# get grid point lats lons (have to add parameter to query otherwise only one grid point is returned...)
query='http://'+source+'/timeseries?bbox='+bbox+'&param=utctime,latitude,longitude,U10-MS:ERA5:5021:1:0:1:0&starttime='+start+'&endtime='+start+'&hour=0&format=json&precision=full&tz=utc&timeformat=sql'
print(query)
response=requests.get(url=query)
results_json=json.loads(response.content)
rs=results_json[0]
for key,val in rs.items():
    if key=='latitude':   
        lats=val.strip('[]').split()
    if key=='longitude':   
        lons=val.strip('[]').split()
lat1,lon1,lat2,lon2,lat3,lon3,lat4,lon4=float(lats[0]),float(lons[0]),float(lats[1]),float(lons[1]),float(lats[2]),float(lons[2]),float(lats[3]),float(lons[3])
print(lat1,lon1,lat2,lon2,lat3,lon3,lat4,lon4)

ax = plt.axes(projection=request.crs)
ax.set_extent(extent)
ax.add_image(request, 13)    # 13 = zoom level

# plot training locations and harbor
# 1
plt.scatter(lon1, lat1, transform=ccrs.PlateCarree(),color='yellow',s=5,label='1 ('+str(lon1)+', '+str(lat1)+')')
# 2
plt.scatter(lon2, lat2, transform=ccrs.PlateCarree(),color='green',s=5,label='2 ('+str(lon2)+', '+str(lat2)+')')
# 3
plt.scatter(lon3, lat3, transform=ccrs.PlateCarree(),color='blue',s=5,label='3 ('+str(lon3)+', '+str(lat3)+')')
# 4
plt.scatter(lon4, lat4, transform=ccrs.PlateCarree(),color='orange',s=5,label='4 ('+str(lon4)+', '+str(lat4)+')')
# harbor
plt.scatter(lon,lat, transform=ccrs.PlateCarree(),color='red',s=5,label='harbor ('+str(round(lon,2))+', '+str(round(lat,2))+')') # harbor
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),facecolor="lightblue")
plt.title('Training locations and '+harbor+' harbor')
# save plot 
plt.savefig('/home/ubuntu/ml-harvesterseasons/'+harbor+'-'+FMISID+'.jpg',dpi=1200)