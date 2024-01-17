#!/usr/bin/env python3
import requests, os, time, glob, json,sys
import pandas as pd
from functools import reduce
import functions as fcts
import numpy as np
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
### SmarMet-server timeseries query to fetch ERA5-Land training data for machine learning (copied from bin and edited )

startTime = time.time()

path_eu_data = "/home/ubuntu/ml-harvesterseasons/soil_temperature/data/Europe_soil_metadata.txt"
source = 'desm.harvesterseasons.com:8080'

eu_data = pd.read_csv(path_eu_data,sep=";")
pd.options.display.max_columns = None

#remove nan values in df

eu_data = eu_data.dropna(subset=['site_lat','site_long'])

# remove duplicate long,lat 

eu_data = eu_data.drop_duplicates(subset=['site_lat','site_long'])

lat = eu_data['site_lat'].tolist()
lon = eu_data['site_long'].tolist()
# points=lucas_df['POINT_ID'].values.tolist()

pointids = list(range(len(eu_data)))
# pointids = list(range(1,6000)) # can only acc 


llpdict = {i:[j, k] for i, j, k in zip(pointids,lat, lon)}


# EXAMPLE get subdict based on list of pointids:
# pointids = list(range(1,10))
# llpdict = dict((k, llpdict[k]) for k in pointids
#            if k in llpdict)
# print(llpdict)

### ERA5-Land predictors
# 24h accumulations
era5l00 = [
    {'slhf':'FLLAT-JM2:ERA5L:5022:1:0:1:0'}, # Surface latent heat flux (J m-2)  
    {'sshf':'FLSEN-JM2:ERA5L:5022:1:0:1:0'}, # Surface sensible heat flux (J m-2)  
    {'ssrd':'RADGLOA-JM2:ERA5L:5022:1:0:1:0'}, # Surface shortwave radiation downwards (J m-2)  
    {'strd':'RADLWA-JM2:ERA5L:5022:1:0:1:0'}, # Surface longwave radiation downwards (J m-2)  
    {'str':'RNETLWA-JM2:ERA5L:5022:1:0:1:0'}, # Net longwave radiation accumulation (J m-2)  
    {'ssr':'RNETSWA-JM2:ERA5L:5022:1:0:1:0'}, # Net shortwave radiation accumulation (J m-2)  
    #{'sf':'SNACC-KGM2:ERA5L:5022:1:0:1:0'}, # Snowfall (m of water eq)  
    {'sktn':'SKT-K:LSASAFC:5064:1:0:0'}, # Skin temperature at night(K)
    {'ro':'RO-M:ERA5L:5022:1:0:1:0'}, # Total precipiation in meters (m), 24h sum (for timesteps previous day!)
    #{'sro':'SRO-M:ERA5L:5022:1:0:1:0'}, # Surface runoff (m)
    #{'ssro':'SSRO-M:ERA5L:5022:1:0:1:0'}, # Sub-surface runoff (m)
    {'evapp':'EVAPP-M:ERA5L:5022:1:0:1:0'}, # Potential evaporation (m)
    #{'evap':'EVAP-M:ERA5L:5022:1:0:1:0'}, # Total precipiation in meters (m), 24h sum (for timesteps previous day!)
    #{'tp':'RR-M:ERA5L:5022:1:0:1:0'}, # Total precipiation in meters (m), 24h sum (for timesteps previous day!)


]

# Hourly data
era5l0012 = [
    {'skt':'SKT-K:ERA5L:5022:1:0:1:0'}, # Skin temperature (K)
    #{'fal':'ALBEDOSLR-0TO1:ERA5L:5022:1:0:1:0'}, # Forecast albedo (0-1)
    #{'asn':'ASN-0TO1:ERA5L:5022:1:0:1:0'}, # Snow albedo (0-1)
    #{'es':'ES-M:ERA5L:5022:1:0:1:0'}, # Snow evaporation (m of water eq.)
    #{'evabs':'EVABS-M:ERA5L:5022:1:0:0'}, # Evaporation from bare soil (m of water eq.)
    #{'evaow':'EVAOW-M:ERA5L:5022:1:0:0'}, # Evaporation from open water surfaces excluding oceans (m of water eq.)
    #{'evatc':'EVATC-M:ERA5L:5022:1:0:0'}, # Evaporation from the top of canopy (m of water eq.)
    #{'evavt':'EVAVT-M:ERA5L:5022:1:0:0'}, # Evaporation from vegetation transpiration (m of water eq)
    {'laihv':'LAI_HV-M2M2:ERA5L:5022:1:0:1:0'}, # Leaf area index, high vegetation (m2 m-2) 
    {'lailv':'LAI_LV-M2M2:ERA5L:5022:1:0:1:0'}, # Leaf area index, low vegetation (m2 m-2) 
    #{'lshf':'LSHF:ERA5L:5022:1:0:1:0'}, # Lake shape factor (dimensionless)
    #{'sp':'PGR-PA:ERA5L:5022:1:0:1:0'}, # Surface pressure/ Pressure on ground (Pa)
    {'sd':'SD-M:ERA5L:5022:1:0:1:0'}, # Snow depth (m of water eq)
    #{'hsnow':'HSNOW-M:ERA5L:5022:1:0:0'}, # Snow depth (m)
    #{'smlt':'SMLT-M:ERA5L:5022:1:0:1:0'}, # Snowmelt (m of water eq)
    {'rsn':'SND-KGM3:ERA5L:5022:1:0:1:0'}, # Snow density (kg m-3) 
    #{'snowc':'SNOWC:ERA5L:5022:1:0:0'}, # Snow cover (%)
    #{'src':'SRC-M:ERA5L:5022:1:0:1:0'}, # Skin reservoir content (m of water eq)
    {'stl1':'STL1-K:ERA5L:5022:9:7:1:0'}, # Soil temperature level 1 (K) 
    {'stl2':'TSOIL-K:ERA5L:5022:9:1820:1:0'}, # Soil temperature level 2 (K)
    #{'stl3':'STL3-K:ERA5L:5022:9:7268:1:0'}, # Soil temperature level 3 (K)
    #{'stl4':'STL4-K:ERA5L:5022:9:25855:1:0'}, # Soil temperature level 4 (K)
    #{'swvl1':'SOILWET-M3M3:ERA5L:5022:9:7:1:0'}, # Soil wetness layer 1 (m3 m-3)  
    {'swvl2':'SWVL2-M3M3:ERA5L:5022:9:1820:1:0'}, # Soil wetness layer 2 (m3 m-3)  
    #{'swvl3':'SWVL3-M3M3:ERA5L:5022:9:7268:1:0'}, # Soil wetness layer 3 (m3 m-3)  
    #{'swvl4':'SWVL4-M3M3:ERA5L:5022:9:25855:1:0'}, # Soil wetness layer 4 (m3 m-3)  
    {'t2':'T2-K:ERA5L:5022:1:0:1:0'}, # 2 metre temperature (K)  
    {'td2':'TD2-K:ERA5L:5022:1:0:1:0'}, # 2 metre dewpoint temperature (K)  
    #{'tsn':'TSN-K:ERA5L:5022:1:0:1:0'}, # Temperature of snow layer (K) 
    {'u10':'U10-MS:ERA5L:5022:1:0:1:0'}, # 10 metre U wind component (m s-1)  
    {'v10':'V10-MS:ERA5L:5022:1:0:1:0'}, # 10 metre V wind component (m s-1)  
]


### 00 UTC parameters (24h accumulated)
hour = '00'
start = '20150101T000000Z'
end = '20221231T000000Z'
# function to merge dfs
def merge_df(df1,df2):
    if df1.empty:
        df = pd.concat([df1,df2],axis=1)
        # df = df.T.drop_duplicates().T
    else:
        df = pd.merge(df1, df2, how='inner', on=['utctime','latitude','longitude','pointID'])
    return df

def multi_merger_df(data_frames):
    df =reduce(lambda  left,right: pd.merge(left,
                                                  right,
                                                  on=['utctime','latitude','longitude','pointID'],
                                            how='inner'), data_frames)
    return df

era5l00_df = pd.DataFrame()
for pardict in era5l00:
    key,value = list(pardict.items())[0]
    print(source,start,end,hour,pardict,llpdict)
    temp_df = fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
    era5l00_df = merge_df(era5l00_df,temp_df)
print("*******************era5l00_df*********************")
print(era5l00_df.head())
# convert point id to int

era5l00_df['pointID'] = era5l00_df['pointID'].astype('Int64')

era5l0012_df = pd.DataFrame()
### 00 and 12 UTC parameters
for pardict in era5l0012:
    hour = '00'
    key,value = list(pardict.items())[0]
    temp_df = fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
    temp_df.rename({key:key+'-00'}, axis=1, inplace=True)
    print("**********00 and 12 UTC parameters****************")
    temp_df['utctime']=temp_df['utctime'].dt.date
    era5l0012_df = merge_df(era5l0012_df,temp_df)
    '''
    hour = '12'
    
    temp_df = fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)

    temp_df['utctime']=temp_df['utctime'].dt.date
    
    temp_df.rename({key:key+'-12'}, axis=1, inplace=True)
   
    era5l0012_df = merge_df(era5l0012_df,temp_df)
    '''
'''
lsasaf0012 = [{'sktd':'SKT-K:LSASAFC:5064:1:0:0'},
             ] # Skin temperature (K) 
start = '20150101T120000Z' 
end = '20221231T120000Z'
for pardict in lsasaf0012:
    
    hour = '12'
    key,value = list(pardict.items())[0]
    temp_df = fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)

    temp_df['utctime']=temp_df['utctime'].dt.date
    
    temp_df.rename({key:key+'-12'}, axis=1, inplace=True)
   
    era5l0012_df = merge_df(era5l0012_df,temp_df)
'''
print("*******************era5l0012_df*********************")
print(era5l0012_df.head())

era5l0012_df['pointID'] = era5l0012_df['pointID'].astype('Int64')
era5l0012_df['utctime'] = pd.to_datetime(era5l0012_df['utctime'])

# combine time series data

time_series_frames = [era5l00_df,era5l0012_df]

print(era5l0012_df.info())

time_series_df = multi_merger_df(data_frames=time_series_frames)

# save time_series_df 

time_series_df.to_csv("/home/ubuntu/data/ML/training-data/soiltemp/timeseries_features_latest.csv",index=False)


