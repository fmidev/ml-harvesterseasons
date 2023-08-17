#!/usr/bin/env python3
import requests, os, time, glob, json,sys
import pandas as pd
import numpy as np
import itertools
import warnings

data_dir='/home/ubuntu/data/ML/training-data/'
#lucas=data_dir+'LUCAS_2018_Copernicus_attr+additions.csv'
lucas=data_dir+'lucas/LUCAS_2018_Copernicus_attr+additions_AT-UK.csv'
cols_own=['POINT_ID','TH_LAT','TH_LONG','DTM_height','DTM_slope','DTM_aspect','TCD','WAW','CorineLC']#,'NUTS0','CPRN_LC','LC1_LABEL','DTM_height','DTM_slope','DTM_aspect','TCD','WAW','CorineLC']
df=pd.read_csv(lucas,usecols=cols_own)
#print(df)

df['POINT_ID']=pd.to_numeric(df['POINT_ID'])

with open(r'/home/ubuntu/data/ML/training-data/63287pointIDs.txt', 'r') as file:
    lines = [line.rstrip() for line in file]

pointslst=[]
for sta in lines:
    pointslst.append(sta)

l=len(pointslst)
#print(pointslst)

for point in pointslst:
        #if df[df['POINT_ID'] == int(point)]
        dfpoint = df[df['POINT_ID'] == int(point)]
        dfpoint=dfpoint[['TH_LAT','TH_LONG','DTM_height','DTM_slope','DTM_aspect','TCD','WAW','CorineLC']].reset_index().drop(columns=['index'])
        dfpoint=dfpoint.loc[dfpoint.index.repeat(2922)].reset_index(drop=True)
        #print(dfpoint)
        print(point)
        dfpoint.to_csv(data_dir+'soilwater/lucasInfo/lucasInfo_'+str(point)+'.csv',index=False)

