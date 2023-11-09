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
llpdict={k:llpdict[k] for k in list(llpdict.keys())[:5000]} # max 5000 points per ts query [:5000], [5000:10000], ...
'''
# EXAMPLE get subdict based on list of pointids:
pointids=[47982724,48022714,48022732]
llpdict = dict((k, llpdict[k]) for k in pointids
           if k in llpdict)
print(llpdict)
'''

### ERA5 pl and sl 00 12 utc predictors
params0012 = [
    {'t850': 'T-K:ERA5:5021:2:850:1:0'}, # temperature in K        
    {'t700': 'T-K:ERA5:5021:2:700:1:0'},  
    {'t500': 'T-K:ERA5:5021:2:500:1:0'},
    {'q850': 'Q-KGKG:ERA5:5021:2:850:1:0'}, # specific humidity in kg/kg
    {'q700': 'Q-KGKG:ERA5:5021:2:700:1:0'},
    {'q500': 'Q-KGKG:ERA5:5021:2:500:1:0'},
    {'u850': 'U-MS:ERA5:5021:2:850:1:0'}, # U comp of wind in m/s
    {'u700': 'U-MS:ERA5:5021:2:700:1:0'},
    {'u500': 'U-MS:ERA5:5021:2:500:1:0'},
    {'v850': 'V-MS:ERA5:5021:2:850:1:0'}, # V comp of wind in m/s
    {'v700': 'V-MS:ERA5:5021:2:700:1:0'},
    {'v500': 'V-MS:ERA5:5021:2:500:1:0'},
    {'z850': 'Z-M2S2:ERA5:5021:2:850:1:0'}, # geopotential in m2 s-2
    {'z700': 'Z-M2S2:ERA5:5021:2:700:1:0'},
    {'z500': 'Z-M2S2:ERA5:5021:2:500:1:0'},    
    {'t2m': 'T2-K:ERA5:5021:1:0:1:0'}, # 2m temperature in K 
    {'td2m': 'TD2-K:ERA5:5021:1:0:1:0'}, # 2m dew point temperature in K
    {'msl': 'PSEA-HPA:ERA5:5021:1:0:1:0'}, # mean sea level pressure in Pa
    {'u10': 'U10-MS:ERA5:5021:1:0:1:0'}, # U comp of wind in m/s
    {'v10': 'V10-MS:ERA5:5021:1:0:1:0'}, # V comp of wind in m/s
    {'tcc': 'N-0TO1:ERA5:5021:1:0:1:0'}, # Cloudiness 0...1
    {'sd': 'SD-M:ERA5:5021:1:0:1:0'}, # Water equivalent of snow cover in mm
    {'skt': 'SKT-K:ERA5:5021:1:0:1:0'}, # Skin temperature
    {'rsn': 'SND-KGM3:ERA5:5021:1:0:1:0'}, # Snow density
    {'mx2t': 'TMAX-K:ERA5:5021:1:0:1:0'}, # Maximum temperature
    {'mn2t': 'TMIN-K:ERA5:5021:1:0:1:0'}, # Minimum temperature
    {'kx': 'KX:ERA5:5021:1:0:0'}, # K index
    {'sst':'TSEA-K:ERA5:5021:1:0:1'} # Sea surface temperature       
]

### ERA5 static predictors
staticSL = [
    {'z': 'Z-M2S2:ERA5:5021:1:0:1:0'}, # geopotential in m2 s-2
    {'lsm': 'LC-0TO1:ERA5:5021:1:0:1:0'}, # Land sea mask: 1=land, 0=sea
    {'sdor': 'SDOR-M:ERA5:5021:1:0:1:0'}, # Standard deviation of orography
    {'slor': 'SLOR:ERA5:5021:1:0:1:0'}, # Slope of sub-gridscale orography
    {'anor': 'ANOR-RAD:ERA5:5021:1:0:1:0'}, # Angle of sub-gridscale orography

]

y_start='2000'
y_end='2020'
source='desm.harvesterseasons.com:8080'
### pl parameters
### 00 and 12 UTC parameters
for pardict in params0012:
    hour='00'
    start=y_start+'0101T000000Z'
    end=y_end+'1231T000000Z'
    key,value=list(pardict.items())[0]
    df=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
    print(key)
    df.rename({key:key+'-00'}, axis=1, inplace=True)
    #print(df)
    for point in llpdict.keys():
        dfpoint = df[df['pointID'] == point].reset_index(drop=True)
        dfpoint.to_csv(data_dir+'era5/era5-'+key+'-00_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
    '''hour='12'
    start=y_start+'0101T000000Z'
    end=y_end+'1231T000000Z' 
    df=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
    df.rename({key:key+'-12'}, axis=1, inplace=True)
    df['utctime']=df['utctime'].dt.date
    #print(df)
    for point in llpdict.keys():
        dfpoint = df[df['pointID'] == point].reset_index(drop=True)
        dfpoint.to_csv(data_dir+'era5/era5-'+key+'-12_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
    '''
### static 
for pardict in staticSL:
    hour='00'
    start=y_start+'0101T000000Z'
    end=y_end+'1231T000000Z'
    key,value=list(pardict.items())[0]
    df=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
    print(key)
    #df.rename({key:key+'-00'}, axis=1, inplace=True)
    #print(df)
    for point in llpdict.keys():
        dfpoint = df[df['pointID'] == point].reset_index(drop=True)
        dfpoint.to_csv(data_dir+'era5/era5-'+key+'_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))