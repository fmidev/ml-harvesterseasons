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
print(lucas_df)
lst=lucas_df[['TH_LAT','TH_LONG']].values.tolist()
points=lucas_df[['POINT_ID']].values.tolist()
latlonlst=list(itertools.chain.from_iterable(lst))
latlonlst=latlonlst[120000:] # [:10000] [10000:20000] [20000:30000] ... [120000:]
pointslst=list(itertools.chain.from_iterable(points))
pointslst=pointslst[60000:] # [:5000 [5000:10000] [10000:15000] ... [60000:]

era5l0012 = [
    {'skt':'SKT-K:ERA5L:5022:1:0:1:0'}, # Skin temperature (K)
    #{'fal':'ALBEDOSLR-0TO1:ERA5L:5022:1:0:1:0'}, # Forecast albedo (0-1)
    #{'asn':'ASN-0TO1:ERA5L:5022:1:0:1:0'}, # Snow albedo (0-1)
    #{'es':'ES-M:ERA5L:5022:1:0:1:0'}, # Snow evaporation (m of water eq.)
    #{'evabs':'EVABS-M:ERA5L:5022:1:0:0'}, # Evaporation from bare soil (m of water eq.)
    #{'evaow':'EVAOW-M:ERA5L:5022:1:0:0'}, # Evaporation from open water surfaces excluding oceans (m of water eq.)
    #{'evatc':'EVATC-M:ERA5L:5022:1:0:0'}, # Evaporation from the top of canopy (m of water eq.)
    #{'evavt':'EVAVT-M:ERA5L:5022:1:0:0'}, # Evaporation from vegetation transpiration (m of water eq)
    {'laihv':'LAI_HV-M2M2:ERA5L:5022:1:0:1:0'}, # Leaf area index, high vegetation (m2 m-2) *DOWNDLOADED
    {'lailv':'LAI_LV-M2M2:ERA5L:5022:1:0:1:0'}, # Leaf area index, low vegetation (m2 m-2) *DOWNDLOADED
    #{'lshf':'LSHF:ERA5L:5022:1:0:1:0'}, # Lake shape factor (dimensionless)
    #{'sp':'PGR-PA:ERA5L:5022:1:0:1:0'}, # Surface pressure/ Pressure on ground (Pa)
    {'sd':'SD-M:ERA5L:5022:1:0:1:0'}, # Snow depth (m of water eq) *DOWNDLOADED
    #{'hsnow':'HSNOW-M:ERA5L:5022:1:0:0'}, # Snow depth (m)
    #{'smlt':'SMLT-M:ERA5L:5022:1:0:1:0'}, # Snowmelt (m of water eq)
    {'rsn':'SND-KGM3:ERA5L:5022:1:0:1:0'}, # Snow density (kg m-3) *DOWNDLOADED
    #{'snowc':'SNOWC:ERA5L:5022:1:0:0'}, # Snow cover (%)
    #{'src':'SRC-M:ERA5L:5022:1:0:1:0'}, # Skin reservoir content (m of water eq)
    {'stl1':'STL1-K:ERA5L:5022:9:7:1:0'}, # Soil temperature level 1 (K) *DOWNDLOADED
    #{'stl2':'TSOIL-K:ERA5L:5022:9:1820:1:0'}, # Soil temperature level 2 (K)
    #{'stl3':'STL3-K:ERA5L:5022:9:7268:1:0'}, # Soil temperature level 3 (K)
    #{'stl4':'STL4-K:ERA5L:5022:9:25855:1:0'}, # Soil temperature level 4 (K)
    {'swvl1':'SOILWET-M3M3:ERA5L:5022:9:7:1:0'}, # Soil wetness layer 1 (m3 m-3) *DOWNDLOADED
    {'swvl2':'SWVL2-M3M3:ERA5L:5022:9:1820:1:0'}, # Soil wetness layer 2 (m3 m-3) *DOWNDLOADED
    {'swvl3':'SWVL3-M3M3:ERA5L:5022:9:7268:1:0'}, # Soil wetness layer 3 (m3 m-3) *DOWNDLOADED
    {'swvl4':'SWVL4-M3M3:ERA5L:5022:9:25855:1:0'}, # Soil wetness layer 4 (m3 m-3) *DOWNDLOADED
    {'t2':'T2-K:ERA5L:5022:1:0:1:0'}, # 2 metre temperature (K) *DOWNDLOADED
    {'td2':'TD2-K:ERA5L:5022:1:0:1:0'}, # 2 metre dewpoint temperature (K) *DOWNDLOADED
    #{'tsn':'TSN-K:ERA5L:5022:1:0:1:0'}, # Temperature of snow layer (K) 
    {'u10':'U10-MS:ERA5L:5022:1:0:1:0'}, # 10 metre U wind component (m s-1) *DOWNDLOADED
    {'v10':'V10-MS:ERA5L:5022:1:0:1:0'}, # 10 metre V wind component (m s-1) *DOWNDLOADED
]

y_start='2015'
y_end='2022'
rr_yrs=list(range(int(y_start),int(y_end)+1))
nan=float('nan')

source='desm.harvesterseasons.com:8080'
start='20150101T000000Z'
end='20221231T000000Z'

### 00 and 12 UTC parameters
for pardict in era5l0012:
    hour='00'
    key,value=list(pardict.items())[0]
    df=fcts.smartmet_ts_query_multiplePoints_hour(source,start,end,hour,latlonlst,pardict,pointslst)
    print(key)
    df.rename({key:key+'-00'}, axis=1, inplace=True)
    for point in pointslst:
        dfpoint = df[df['pointID'] == point]
        dfpoint.to_csv(data_dir+'soilwater/era5l-fixed/era5l-'+key+'-00_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
    hour='12'
    df=fcts.smartmet_ts_query_multiplePoints_hour(source,start,end,hour,latlonlst,pardict,pointslst)
    df.rename({key:key+'-12'}, axis=1, inplace=True)
    df['utctime']=df['utctime'].dt.date
    for point in pointslst:
        dfpoint = df[df['pointID'] == point]
        dfpoint.to_csv(data_dir+'soilwater/era5l-fixed/era5l-'+key+'-12_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))