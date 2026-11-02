import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import pandas as pd
#import requests,json
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

nuts='/home/ubuntu/data/ML/NUTS_RG_20M_2021_4326.json'
data_dir='/home/ubuntu/data/ML/training-data/RR/'
eobsinfo=data_dir+'eobs/eobsStaidInfo.csv'
subset=data_dir+'stationIDs_4707.txt'
#eobsinfo=data_dir+'eobs/eobsStaidInfo_reduced.csv'
eobs_df_all=pd.read_csv(eobsinfo)

# plot subset
print(eobs_df_all)
with open(subset, "r") as file:
    lines = file.read().splitlines()
ids = [int(line) for line in lines]
eobs_df = eobs_df_all[eobs_df_all['STAID'].isin(ids)]
print(eobs_df)

countries = gpd.read_file(nuts)

locs=str(len(eobs_df))

eobs_df.plot(ax=countries.plot(),x="LON",y="LAT",kind="scatter",color="red",s=0.01)
#plt.xlim(-30, 50)
#plt.ylim(25, 75) 
plt.xlim(-15, 35)
plt.ylim(30, 75) 
plt.grid(color='lightgrey', linestyle='--', linewidth=0.5)
plt.xticks(range(-15, 36, 5))  
plt.yticks(range(30, 76, 5))
plt.title(locs+' Training Locations Across Europe')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.savefig(data_dir+'eobs-'+locs+'-locations.png',dpi=800)
#plt.savefig(data_dir+'TESTI.svg', format='svg') #, bbox_inches='tight'
