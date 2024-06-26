import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

nuts='/home/ubuntu/data/ML/NUTS_RG_20M_2021_4326.json'
data_dir='/home/ubuntu/data/ML/training-data/'
res_dir='/home/ubuntu/data/ML/results/soilwater/figures/'

fname='10000pointIDs-2.txt'
with open(r'/home/ubuntu/data/ML/training-data/'+fname, 'r') as file:
    lines = [line.rstrip() for line in file]
points=[]
for sta in lines:
    points.append(sta)
fname='LUCAS_2018_Copernicus_attr+additions.csv'
lucas=data_dir+fname
cols_own=['POINT_ID','TH_LAT','TH_LONG']#,'NUTS0','CPRN_LC','LC1_LABEL','DTM_height','DTM_slope','DTM_aspect','TCD','WAW','CorineLC']
lucas_df=pd.read_csv(lucas,usecols=cols_own) 
#print(lucas_df)

lats=[]
lons=[]
for nro in points:
    lat=lucas_df.query('POINT_ID=='+nro)['TH_LAT'].item()
    lon=lucas_df.query('POINT_ID=='+nro)['TH_LONG'].item()
    lats.append(lat)
    lons.append(lon)
df = pd.DataFrame(list(zip(lats, lons)),
               columns =['TH_LAT', 'TH_LON'])
#print(df)

countries = gpd.read_file(nuts)
df.plot(ax=countries.plot(),x="TH_LON",y="TH_LAT",kind="scatter",color="red",s=0.01)
plt.xlim(-30, 50)
plt.ylim(25, 75) 

plt.savefig(res_dir+'10000LucasPoints4XGB.png',dpi=800)
