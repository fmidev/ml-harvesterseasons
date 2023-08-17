#!/usr/bin/env python3
import requests, os, time, glob, json,sys
import pandas as pd
import functions as fcts
import numpy as np
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
### SmarMet-server timeseries query to fetch Copernicus DEM (in ERA5-Land grid) training data for machine learning

startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/'
#lucas=data_dir+'soilwater/lucas_allA-Hclass_rep.csv'
lucas=data_dir+'LUCAS_2018_Copernicus_attr+additions.csv'
cols_own=['POINT_ID','TH_LAT','TH_LONG']#,'NUTS0','CPRN_LC','LC1_LABEL','DTM_height','DTM_slope','DTM_aspect','TCD','WAW','CorineLC']
lucas_df=pd.read_csv(lucas,usecols=cols_own)
print(lucas_df)
lst=lucas_df[['TH_LAT','TH_LONG']].values.tolist()
points=lucas_df[['POINT_ID']].values.tolist()
latlonlst=list(itertools.chain.from_iterable(lst))
latlonlst=latlonlst[120000:] # [:10000] [10000:20000] [20000:30000] ... [120000:]
pointslst=list(itertools.chain.from_iterable(points))
pointslst=pointslst[60000:] # [:5000 [5000:10000] [10000:15000] ... [60000:]

### DEM predictors
demERA5L = [
    {'anor':'ANOR-RAD:COPERNICUS:5022:1:0:1'}, # Angle of sub-gridscale orography
    {'hl':'HL-M:COPERNICUS:5022:1:0:1'}, # Height of level in meters
    {'isor':'ISOR:COPERNICUS:5022:1:0:1'}, # Anisotropy of sub-gridscale orography
    {'slor':'SLOR:COPERNICUS:5022:1:0:1'} # Slope of sub-gridscale orography  
]
demERA5 = [
    {'anor':'ANOR-RAD:COPERNICUS:5021:1:0:1'}, # Angle of sub-gridscale orography
    {'hl':'HL-M:COPERNICUS:5021:1:0:1'}, # Height of level in meters
    {'isor':'ISOR:COPERNICUS:5021:1:0:1'}, # Anisotropy of sub-gridscale orography
    {'slor':'SLOR:COPERNICUS:5021:1:0:1'} # Slope of sub-gridscale orography  
]

ymonday='20110701'

source='desm.harvesterseasons.com:8080'
tstep='1440'

for pardict in demERA5L:
    key,value=list(pardict.items())[0]
    start='20110701T000000Z'
    end='20110701T000000Z'
    df=fcts.smartmet_ts_query_multiplePoints(source,start,end,tstep,latlonlst,pardict,pointslst)
    print(key)
    for point in pointslst:
        dfpoint = df[df['pointID'] == point]
        dfpoint.to_csv(data_dir+'soilwater/copernicus/copernicus-dem-era5l_'+key+'_'+ymonday+'_'+str(point)+'.csv',index=False)

for pardict in demERA5:
    key,value=list(pardict.items())[0]
    start='20110701T000000Z'
    end='20110701T000000Z'
    df=fcts.smartmet_ts_query_multiplePoints(source,start,end,tstep,latlonlst,pardict,pointslst)
    print(key)
    for point in pointslst:
        dfpoint = df[df['pointID'] == point]
        dfpoint.to_csv(data_dir+'soilwater/copernicus/copernicus-dem-era5_'+key+'_'+ymonday+'_'+str(point)+'.csv',index=False)


executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))