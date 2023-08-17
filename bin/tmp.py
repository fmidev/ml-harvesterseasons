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
#lucas=data_dir+'soilwater/lucas_allA-Hclass_rep.csv'
lucas=data_dir+'LUCAS_2018_Copernicus_attr+additions.csv'
cols_own=['POINT_ID','TH_LAT','TH_LONG']#,'NUTS0','CPRN_LC','LC1_LABEL','DTM_height','DTM_slope','DTM_aspect','TCD','WAW','CorineLC']
lucas_df=pd.read_csv(lucas,usecols=cols_own)

lat=lucas_df['TH_LAT'].values.tolist()
lon=lucas_df['TH_LONG'].values.tolist()
points=lucas_df['POINT_ID'].values.tolist()

lat=lat[:3]
lon=lon[:3]
points=points[:3]
llpdict={i:[j, k] for i, j, k in zip(points,lat, lon)}
print(llpdict)

latlonlst=[]
for t in llpdict.values():
    for i in t:
        latlonlst.append(i)
print(latlonlst)

staids=[]
for key in llpdict.keys():
    print(key)