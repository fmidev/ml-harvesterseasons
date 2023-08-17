from dataclasses import dataclass
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors

nuts='/home/ubuntu/data/ML/NUTS_RG_20M_2021_4326.json'
data_dir='/home/ubuntu/data/ML/training-data/'
res_dir='/home/ubuntu/data/ML/results/soilwater/figures/'
#with open(r'/home/ubuntu/data/ML/training-data/soilwater/fitdata/236pointIDs.txt', 'r') as file:
#with open(r'/home/ubuntu/data/ML/training-data/soilwater/fitdata/pointIDs.txt', 'r') as file:
#with open(r'/home/ubuntu/data/ML/training-data/soilwater/fitdata/404pointIDs.txt', 'r') as file:
#with open(r'/home/ubuntu/data/ML/training-data/63287pointIDs.txt', 'r') as file:
with open(r'/home/ubuntu/data/ML/training-data/soilwater/workingPointsFrom5000.txt', 'r') as file:
    lines = [line.rstrip() for line in file]
points=[]
for sta in lines:
    points.append(sta)

lucas=data_dir+'LUCAS_2018_Copernicus_attr+additions.csv'
cols_own=['POINT_ID','TH_LAT','TH_LONG']#,'NUTS0','CPRN_LC','LC1_LABEL','DTM_height','DTM_slope','DTM_aspect','TCD','WAW','CorineLC']
lucas_df=pd.read_csv(lucas,usecols=cols_own)

lats=[]
lons=[]
for nro in points:
    lat=lucas_df.query('POINT_ID=='+nro)['TH_LAT'].item()
    lon=lucas_df.query('POINT_ID=='+nro)['TH_LONG'].item()
    lats.append(lat)
    lons.append(lon)

df = pd.DataFrame(list(zip(lats, lons)),
               columns =['lat', 'lon'])

#color_list = ['red','saddlebrown','darkgoldenrod','gold', 'yellowgreen', 'olive','darkolivegreen'] 
#cmap = mpl.cm.plasma
#cmap = mcolors.LinearSegmentedColormap.from_list("my_colormap", color_list)

countries = gpd.read_file(nuts)
df.plot(ax=countries.plot(),x="lon",y="lat",kind="scatter",color="red",s=0.001)
#rr_data.plot(ax=countries.plot(),x="lon", y="lat", kind="scatter", c="slope",colormap=cmap)
#rr_data.plot(ax=countries.plot(),x="lon", y="lat", kind="scatter",color='red')
plt.xlim(-12, 32)
plt.ylim(33, 72) 
#plt.savefig(res_dir+'63287points-on-map.png',dpi=200)
plt.savefig('/home/ubuntu/data/ML/results/soilwater/figures/4108points-on-map.png',dpi=800)
