#!/usr/bin/env python3
import requests, os, time, glob, json,sys
import pandas as pd
import functions as fcts
import numpy as np
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
### SmarMet-server timeseries query to fetch static ECC parameters to training data for machine learning
# conda activate xgb

startTime=time.time()

# read in point-ids, latitudes and longitudes from LUCAS to a dictonary
data_dir='/home/ubuntu/data/ML/training-data/'
lucas=data_dir+'LUCAS_2018_Copernicus_attr+additions.csv'
cols_own=['POINT_ID','TH_LAT','TH_LONG']
lucas_df=pd.read_csv(lucas,usecols=cols_own)
lat=lucas_df['TH_LAT'].values.tolist()
lon=lucas_df['TH_LONG'].values.tolist()
points=lucas_df['POINT_ID'].values.tolist()
llpdict={i:[j, k] for i, j, k in zip(points,lat, lon)}
llpdict={k:llpdict[k] for k in list(llpdict.keys())[:5000]} # max 5000 points per ts query [:5000], [5000:10000], ...

'''
# EXAMPLE get subdict based on list of pointids:
pointids=[47982724,48022714,48022732]
llpdict = dict((k, llpdict[k]) for k in pointids
           if k in llpdict)
print(llpdict)
'''
# create empty df for teaching period 2015-2022
dfres=pd.DataFrame()
dfres['utctime']=pd.date_range(start='2015-01-01', end='2022-12-31')    
dfres[['lake_cover','cvh','cvl','lake_depth','land_cover','soiltype','urban_cover','tvh','tvl','latitudet','longitude','pointID']] = np.nan
dfres=dfres.set_index('utctime')

### ecc parameters
pardict = {'lake_cover':'CL-0TO1:ECC:5059:1:0:0', # lake cover
    'cvh':'CVH-N:ECC:5059:1:0:0', # high vegetation cover
    'cvl':'CVL-N:ECC:5059:1:0:0', # low vegetation cover 
    'lake_depth':'DL-M:ECC:5059:1:0:0', # lake total depth
    'land_cover':'LC-0TO1:ECC:5059:1:0:0', # land cover
    'soiltype':'SOILTY-N:ECC:5059:1:0:0', # soil type
    'urban_cover':'CUR-0TO1:ECC:5059:1:0:0', # urban cover fraction    
    'tvh':'TVH-N:ECC:5059:1:0:0', # type of high vegetation
    'tvl':'TVL-N:ECC:5059:1:0:0'} # type of low vegetation

source='desm.harvesterseasons.com:8080'
hour='00'
start='data' # static values have different dates for available data at smartmet-desm
end='data'
df=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
df=df.astype({'pointID': 'int32'})
for point in llpdict.keys():
    dfpoint = df[df['pointID'] == point]
    dfpoint=dfpoint.fillna(method='bfill').drop(columns='utctime').astype('float32')
    dfres.iloc[0]=dfpoint.iloc[0]
    dfres=dfres.fillna(method='ffill')
    dfsave=dfres.drop(columns=['latitudet','longitude','pointID'])
    #print(dfsave)
    dfsave.to_csv(data_dir+'soilwater/eccInfo/ecc_2015-2022_'+str(point)+'.csv',index=False)
    dfres[['lake_cover','cvh','cvl','lake_depth','land_cover','soiltype','urban_cover','tvh','tvl','latitudet','longitude','pointID']] = np.nan

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))