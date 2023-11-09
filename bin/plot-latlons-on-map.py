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

fname='precipitation-harmonia/5441_training_stations.txt'
with open(r'/home/ubuntu/data/ML/training-data/'+fname, 'r') as file:
    lines = [line.rstrip() for line in file]
points=[]
for sta in lines:
    points.append(sta)
#fname='LUCAS_2018_Copernicus_attr+additions.csv'
fname='EOBS_orography_EU_10766points.csv'
#fname='EOBS_orography_EU_6766points.csv'
lucas=data_dir+fname
#cols_own=['POINT_ID','TH_LAT','TH_LONG']#,'NUTS0','CPRN_LC','LC1_LABEL','DTM_height','DTM_slope','DTM_aspect','TCD','WAW','CorineLC']
cols_own=['STAID','LAT','LON']
df=pd.read_csv(lucas,usecols=cols_own) # lucas_df
'''print(lucas_df)

lats=[]
lons=[]
for nro in points:
    #lat=lucas_df.query('POINT_ID=='+nro)['TH_LAT'].item()
    #lon=lucas_df.query('POINT_ID=='+nro)['TH_LONG'].item()
    lat=lucas_df.query('STAID=='+nro)['LAT'].item()
    lon=lucas_df.query('STAID=='+nro)['LON'].item()
    lats.append(lat)
    lons.append(lon)

df = pd.DataFrame(list(zip(lats, lons)),
               columns =['LAT', 'LON'])
'''
print(df)

#color_list = ['red','saddlebrown','darkgoldenrod','gold', 'yellowgreen', 'olive','darkolivegreen'] 
#cmap = mpl.cm.plasma
#cmap = mcolors.LinearSegmentedColormap.from_list("my_colormap", color_list)

countries = gpd.read_file(nuts)
df.plot(ax=countries.plot(),x="LON",y="LAT",kind="scatter",color="red",s=0.01)
#rr_data.plot(ax=countries.plot(),x="lon", y="lat", kind="scatter", c="slope",colormap=cmap)
#rr_data.plot(ax=countries.plot(),x="lon", y="lat", kind="scatter",color='red')
plt.xlim(-30, 50)
plt.ylim(25, 75) 
#plt.savefig(res_dir+'63287points-on-map.png',dpi=200)
plt.savefig('/home/ubuntu/data/ML/results/precipitation-harmonia/figures/eobs10766-on-map.png',dpi=800)
