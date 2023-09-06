#!/usr/bin/env python3
import requests, os, time, glob, json,sys
import pandas as pd
import functions as fcts
import numpy as np
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
### SmarMet-server timeseries query to fetch daily SWI training data for machine learning
# SWI1, SWI2, SWI3, SWI4 for predictand/target in fitting
# conda activate xgb

startTime=time.time()

# read in point-ids, latitudes and longitudes from LUCAS 
data_dir='/home/ubuntu/data/ML/training-data/'
lucas=data_dir+'LUCAS_2018_Copernicus_attr+additions.csv'
cols_own=['POINT_ID','TH_LAT','TH_LONG']#,'NUTS0','CPRN_LC','LC1_LABEL','DTM_height','DTM_slope','DTM_aspect','TCD','WAW','CorineLC']
lucas_df=pd.read_csv(lucas,usecols=cols_own)
lat=lucas_df['TH_LAT'].values.tolist()
lon=lucas_df['TH_LONG'].values.tolist()
points=lucas_df['POINT_ID'].values.tolist()
llpdict={i:[j, k] for i, j, k in zip(points,lat, lon)}

# EXAMPLE get subdict based on list of pointids:
pointids=[47982724,48022714,48022732]
llpdict = dict((k, llpdict[k]) for k in pointids
           if k in llpdict)
print(llpdict)

### SWI predictands
swi = [
    {'swi1':'interpolate_t(SWI1:SWI:5059:1:0:0/2d/2d)'}, # Soil wetness index layer 1
    {'swi2':'interpolate_t(SWI2:SWI:5059:1:0:0/2d/2d)'}#, # Soil wetness index layer 2
    {'swi3':'interpolate_t(SWI3:SWI:5059:1:0:0/2d/2d)'}, # Soil wetness index layer 3
    {'swi4':'interpolate_t(SWI4:SWI:5059:1:0:0/2d/2d)'} # Soil wetness index layer 4 
    ]

y_start='2015'
y_end='2022'
rr_yrs=list(range(int(y_start),int(y_end)+1))
nan=float('nan')
source='desm.harvesterseasons.com:8080'
hour='12'

### timeseries query to server
for pardict in swi:
    key,value=list(pardict.items())[0]
    start=y_start+'0101T120000Z'
    end=y_end+'1231T120000Z'
    df=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,swi,llpdict)
    df=df.astype({'pointID': 'int32'})
    #print(df)
    for point in llpdict.keys():
        dfpoint = df[df['pointID'] == point]
        dfpoint.to_csv(data_dir+'soilwater/swi/'+key+'/'+key+'_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))