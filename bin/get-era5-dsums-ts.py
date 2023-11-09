#!/usr/bin/env python3
import requests, os, time, glob, json,sys
import pandas as pd
import functions as fcts
import numpy as np
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
### SmarMet-server timeseries query to fetch ERA5-Land training data for machine learning

startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/'
#eobs=data_dir+'EOBS_orography_EU_6766points.csv'
eobs=data_dir+'EOBS_orography_EU.csv'
cols_own=['STAID','LAT','LON']
eobs_df=pd.read_csv(eobs,usecols=cols_own)
lat=eobs_df['LAT'].values.tolist()
lon=eobs_df['LON'].values.tolist()
points=eobs_df['STAID'].values.tolist()
llpdict={i:[j, k] for i, j, k in zip(points,lat, lon)}
llpdict={k:llpdict[k] for k in list(llpdict.keys())[5000:]} # max 5000 points per ts query [:5000], [5000:10000], ...
'''
# EXAMPLE get subdict based on list of pointids:
pointids=[47982724,48022714,48022732]
llpdict = dict((k, llpdict[k]) for k in pointids
           if k in llpdict)
print(llpdict)
'''

### ERA5 dailysums predictors
staticSL = [
    {'tp':'RR-M:ERA5:5021:1:0:1'}, # total precipitation
    {'evap':'EVAP-M:ERA5:5021:1:0:1'}, # evaporation 
    {'ro':'RO-M:ERA5:5021:1:0:1'}, # runoff
    {'ssr':'RNETSWA-JM2:ERA5:5021:1:0:1'}, # surface net solar radiation
    {'ssrd':'RADGLOA-JM2:ERA5:5021:1:0:1'}, # surface solar radiation downwards
    {'strd':'RADLWA-JM2:ERA5:5021:1:0:1'}, # Surface thermal radiation downwards 
    {'tsr':'TSR-J:ERA5:5021:1:0:1'}, # top net solar radiation
    {'ttr':'RTOPLWA-JM2:ERA5:5021:1:0:1'}, # top net thermal radiation
    {'sf':'SNACC-KGM2:ERA5:5021:1:0:1'}, # snowfall
    {'slhf':'FLLAT-JM2:ERA5:5021:1:0:1'}, # surface latent heat flux
    {'sshf':'FLSEN-JM2:ERA5:5021:1:0:1'} # surface sensible heat flux
    ]

y_start='2000'
y_end='2020'
source='desm.harvesterseasons.com:8080'

for pardict in staticSL:
    time='1130'
    start=y_start+'0101T000000Z'
    end=y_end+'1231T000000Z'
    key,value=list(pardict.items())[0]
    df=fcts.smartmet_ts_query_multiplePointsByID_time(source,start,end,time,pardict,llpdict)
    print(key)
    #df.rename({key:key+'-00'}, axis=1, inplace=True)
    print(df)
    for point in llpdict.keys():
        dfpoint = df[df['pointID'] == point].reset_index(drop=True)
        dfpoint.to_csv(data_dir+'era5/era5-'+key+'_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)


executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))