import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

nuts='/home/ubuntu/data/ML/NUTS_RG_20M_2021_4326.json'
data_dir='/home/ubuntu/data/ML/training-data/'
res_dir='/home/ubuntu/data/ML/results/soilwater/figures/'
'''pointsFile='10000pointIDs-2.txt'
figname='10000LucasPoints5XGB'
locnro=10000
'''
pointsFile='63287pointIDs.txt'
figname='63287LucasPoints'
locnro=63287


with open(r'/home/ubuntu/data/ML/training-data/'+pointsFile, 'r') as file:
    lines = [line.rstrip() for line in file]

lucas=data_dir+'LUCAS_2018_Copernicus_attr+additions.csv'
cols_own=['POINT_ID','TH_LAT','TH_LONG']#,'NUTS0','CPRN_LC','LC1_LABEL','DTM_height','DTM_slope','DTM_aspect','TCD','WAW','CorineLC']
lucas_df=pd.read_csv(lucas,usecols=cols_own) 

points_df = pd.DataFrame({'POINT_ID': lines})
points_df['POINT_ID'] = points_df['POINT_ID'].astype(int)

df = points_df.merge(lucas_df, on='POINT_ID', how='left')

countries = gpd.read_file(nuts)

df.plot(ax=countries.plot(),x="TH_LONG",y="TH_LAT",kind="scatter",color="red",s=0.01)
#plt.xlim(-30, 50)
#plt.ylim(25, 75) 
plt.xlim(-15, 35)
plt.ylim(30, 75) 
plt.grid(color='lightgrey', linestyle='--', linewidth=0.5)
plt.xticks(range(-15, 36, 5))  
plt.yticks(range(30, 76, 5))
plt.title(str(locnro)+' LUCAS Locations Across Europe')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

#plt.savefig(res_dir+figname+'.png',dpi=800)
plt.savefig(res_dir+figname+'.svg', format='svg') #, bbox_inches='tight'
