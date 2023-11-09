#!/usr/bin/env python3
import requests, os, time, glob, json,sys
import pandas as pd
import functions as fcts
import numpy as np
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
### SmarMet-server timeseries query to fetch ERA5-Land training data for machine learning (copied from bin and edited )

startTime = time.time()

path_eu_data = "/home/ubuntu/ml-harvesterseasons/soil_temperature/data/Europe_soil_metadata.txt"

#lucas=data_dir+'soilwater/lucas_allA-Hclass_rep.csv'
# lucas=location_dir+'LUCAS_2018_Copernicus_attr+additions.csv'
# cols_own=['POINT_ID','TH_LAT','TH_LONG']#,'NUTS0','CPRN_LC','LC1_LABEL','DTM_height','DTM_slope','DTM_aspect','TCD','WAW','CorineLC']
# lucas_df=pd.read_csv(lucas,usecols=cols_own)
# station_loc = pd.read_csv(location_dir)
# station_loc = station_loc.join(station_loc['xy'].str.split('_', n=1, expand=True).rename(columns={0:'long', 1:'lat'}))
# station_loc = station_loc.drop("xy", axis='columns')
# station_loc["long"] = station_loc["long"].astype(float)
# station_loc["lat"] = station_loc["lat"].astype(float)

# eu_station_loc = pd.DataFrame()
# eu_station_loc = station_loc.loc[(station_loc["long"] >= -30.0) & (station_loc["long"] <= 50)]
# eu_station_loc = station_loc.loc[(station_loc['lat'] >= 25) & (station_loc['lat'] <= 75)]

eu_data = pd.read_csv(path_eu_data,sep=";")
pd.options.display.max_columns = None

#remove nan values in df

eu_data = eu_data.dropna(subset=['site_lat','site_long'])

lat = eu_data['site_lat'].tolist()
lon = eu_data['site_long'].tolist()
# points=lucas_df['POINT_ID'].values.tolist()
pointids = list(range(1,len(lon)))
# print(lon[:5]) 


llpdict = {i:[j, k] for i, j, k in zip(pointids,lat, lon)}


# EXAMPLE get subdict based on list of pointids:
pointids = list(range(1,10))
llpdict = dict((k, llpdict[k]) for k in pointids
           if k in llpdict)
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
    {'sf':'SNACC-KGM2:ERA5L:5022:1:0:1:0'} # Snowfall (m of water eq)  
]

# Hourly data
era5l0012 = [
    #{'skt':'SKT-K:ERA5L:5022:1:0:1:0'}, # Skin temperature (K)
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
    #{'stl2':'TSOIL-K:ERA5L:5022:9:1820:1:0'}, # Soil temperature level 2 (K)
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

y_start = '2015'
y_end = '2022'
rr_yrs = list(range(int(y_start),int(y_end)+1))
nan = float('nan')
source = 'desm.harvesterseasons.com:8080'

### 00 UTC parameters (24h accumulated)
hour = '00'

# end = '20221231T000000Z'
start = '20150101T000000Z'
end = '20150131T000000Z'
# end = '20221231T000000Z'
df = pd.DataFrame()
for pardict in era5l00:
    key,value = list(pardict.items())[0]
    print(source,start,end,hour,pardict,llpdict)
    temp_df = fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
    if df.empty:
        df = pd.concat([df,temp_df],axis=1)
        # df = df.T.drop_duplicates().T
    else:
        df = pd.merge(df, temp_df, how='inner', on=['utctime','latitude','longitude','pointID'])
    # print(key)




df['utctime']=df['utctime'].dt.date
'''
### 00 and 12 UTC parameters
for pardict in era5l0012:
    hour = '00'
    
    key,value = list(pardict.items())[0]
    temp_df = fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
    temp_df.rename({key:key+'-00'}, axis=1, inplace=True)
    print("**********00 and 12 UTC parameters****************")
    temp_df['utctime']=temp_df['utctime'].dt.date
    df = pd.merge(df, temp_df, how='inner', on=['utctime','latitude','longitude','pointID'])

    hour = '12'
    
    temp_df = fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)

    temp_df['utctime']=temp_df['utctime'].dt.date
    
    temp_df.rename({key:key+'-12'}, axis=1, inplace=True)
   
    df = pd.merge(df, temp_df, how='inner', on=['utctime','latitude','longitude','pointID'])
    


# start = '20140901T000000Z'
# end = '20161231T000000Z'
'''
hour = '00'
start = '20150101T000000Z'
end = '20150131T000000Z'
# end = '20221231T000000Z'


### Runoff rolling cumsums fot t-5, t-15, t-60 and t-100 days
rodict = {'ro':'RO-M:ERA5L:5022:1:0:1:0'} # Total precipiation in meters (m), 24h sum (for timesteps previous day!)
ro_df = fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,rodict,llpdict)    
ro_df = ro_df.set_index('utctime')
ro_df['utctime']=pd.to_datetime(ro_df.index)
print(ro_df)
# roll_df = pd.merge(roll_df, temp_df, how='inner', on=['utctime','latitude','longitude','pointID']) 

temp_df5 = fcts.rolling_cumsum(ro_df.copy(),'5d','ro')
print(temp_df5)
'''
temp_df15 = fcts.rolling_cumsum(df.copy(),'15d','ro')
temp_df60 = fcts.rolling_cumsum(df.copy(),'60d','ro')
temp_df100 = fcts.rolling_cumsum(df.copy(),'100d','ro')
temp_df = temp_df.loc[(temp_df['utctime'] >= '2015-01-01')] 
df = pd.merge(df, temp_df, how='inner', on=['utctime','latitude','longitude','pointID']) 

### Surface runoff rolling cumsums fot t-5, t-15, t-60 and t-100 days
srodict = {'sro':'SRO-M:ERA5L:5022:1:0:1:0'} # Surface runoff (m)
temp_df = fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,srodict,llpdict)    
temp_df = df.set_index('utctime')
temp_df['utctime'] = pd.to_datetime(temp_df.index)
temp_df5 = fcts.rolling_cumsum(df.copy(),'5d','sro')
temp_df15 = fcts.rolling_cumsum(df.copy(),'15d','sro')
temp_df60 = fcts.rolling_cumsum(df.copy(),'60d','sro')
temp_df100 = fcts.rolling_cumsum(df.copy(),'100d','sro')
temp_df = temp_df.loc[(temp_df['utctime'] >= '2015-01-01')] 

### Sub-surface runoff rolling cumsums fot t-5, t-15, t-60 and t-100 days
ssrodict = {'ssro':'SSRO-M:ERA5L:5022:1:0:1:0'} # Sub-surface runoff (m)  
temp_df = fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,ssrodict,llpdict)    
temp_df = temp_df.set_index('utctime')
temp_df['utctime'] = pd.to_datetime(temp_df.index)
temp_df5 = fcts.rolling_cumsum(temp_df.copy(),'5d','ssro')
temp_df15 = fcts.rolling_cumsum(temp_df.copy(),'15d','ssro')
temp_df60 = fcts.rolling_cumsum(temp_df.copy(),'60d','ssro')
temp_df100 = fcts.rolling_cumsum(temp_df.copy(),'100d','ssro')
temp_df = temp_df.loc[(temp_df['utctime'] >= '2015-01-01')]

### Potential evaporation rolling cumsums fot t-5, t-15, t-60 and t-100 days
pevdict = {'evapp':'EVAPP-M:ERA5L:5022:1:0:1:0'} # Potential evaporation (m)
df = fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pevdict,llpdict)    
df = df.set_index('utctime')
df['utctime'] = pd.to_datetime(df.index)
df5 = fcts.rolling_cumsum(df.copy(),'5d','evapp')
df15 = fcts.rolling_cumsum(df.copy(),'15d','evapp')
df60 = fcts.rolling_cumsum(df.copy(),'60d','evapp')
df100 = fcts.rolling_cumsum(df.copy(),'100d','evapp')
df = df.loc[(df['utctime'] >= '2015-01-01')]   

### Evaporation rolling cumsums fot t-5, t-15, t-60 and t-100 days
evapdict = {'evap':'EVAP-M:ERA5L:5022:1:0:1:0'} # Total precipiation in meters (m), 24h sum (for timesteps previous day!)
temp_df = fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,evapdict,llpdict)    
temp_df = temp_df.set_index('utctime')
temp_df['utctime']=pd.to_datetime(temp_df.index)
temp_df5 = fcts.rolling_cumsum(temp_df.copy(),'5d','evap')
temp_df15 = fcts.rolling_cumsum(temp_df.copy(),'15d','evap')
temp_df60 = fcts.rolling_cumsum(temp_df.copy(),'60d','evap')
temp_df100 = fcts.rolling_cumsum(temp_df.copy(),'100d','evap')
temp_df = temp_df.loc[(temp_df['utctime'] >= '2015-01-01')]   

### Precipitation rolling cumsums fot t-5, t-15, t-60 and t-100 days
tpdict = {'tp':'RR-M:ERA5L:5022:1:0:1:0'} # Total precipiation in meters (m), 24h sum (for timesteps previous day!)
df = fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,tpdict,llpdict)    
df = df.set_index('utctime')
df['utctime'] = pd.to_datetime(df.index)
df5 = fcts.rolling_cumsum(df.copy(),'5d','tp')
df15 = fcts.rolling_cumsum(df.copy(),'15d','tp')
df60 = fcts.rolling_cumsum(df.copy(),'60d','tp')
df100 = fcts.rolling_cumsum(df.copy(),'100d','tp')
df = df.loc[(df['utctime'] >= '2015-01-01')]   

executionTime = (time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))
'''