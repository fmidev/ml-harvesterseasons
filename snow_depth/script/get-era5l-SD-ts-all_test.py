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
#lucas=data_dir+'LUCAS_2018_Copernicus_attr+additions.csv'
#eobspoints=data_dir+'EOBS_orography_EU_sd.csv'
eobspoints=data_dir+'EOBS_orography_EU_sd10test.csv'
#cols_own=['POINT_ID','TH_LAT','TH_LONG']#,'NUTS0','CPRN_LC','LC1_LABEL','DTM_height','DTM_slope','DTM_aspect','TCD','WAW','CorineLC']
cols_own=['STAID','LAT','LON']
#lucas_df=pd.read_csv(lucas,usecols=cols_own)
#lat=lucas_df['TH_LAT'].values.tolist()
#lon=lucas_df['TH_LONG'].values.tolist()
#points=lucas_df['POINT_ID'].values.tolist()
eobs_df=pd.read_csv(eobspoints,usecols=cols_own)
lat=eobs_df['LAT'].values.tolist()
lon=eobs_df['LON'].values.tolist()
points=eobs_df['STAID'].values.tolist()
llpdict={i:[j, k] for i, j, k in zip(points,lat, lon)}

# EXAMPLE get subdict based on list of pointids:
'''pointids=[47982724,48022714,48022732]
llpdict = dict((k, llpdict[k]) for k in pointids
           if k in llpdict)
print(llpdict)
'''


### ERA5-Land predictors
# 24h accumulations
era5l00 = [
    {'slhf':'FLLAT-JM2:ERA5L:5022:1:0:1:0'}, # Surface latent heat flux (J m-2)  
    #{'sshf':'FLSEN-JM2:ERA5L:5022:1:0:1:0'}, # Surface sensible heat flux (J m-2)  
    #{'ssrd':'RADGLOA-JM2:ERA5L:5022:1:0:1:0'}, # Surface shortwave radiation downwards (J m-2)  
    #{'strd':'RADLWA-JM2:ERA5L:5022:1:0:1:0'}, # Surface longwave radiation downwards (J m-2)  
    #{'str':'RNETLWA-JM2:ERA5L:5022:1:0:1:0'}, # Net longwave radiation accumulation (J m-2)  
    #{'ssr':'RNETSWA-JM2:ERA5L:5022:1:0:1:0'}, # Net shortwave radiation accumulation (J m-2)  
    #{'sf':'SNACC-KGM2:ERA5L:5022:1:0:1:0'} # Snowfall (m of water eq)  
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
    #{'laihv':'LAI_HV-M2M2:ERA5L:5022:1:0:1:0'}, # Leaf area index, high vegetation (m2 m-2) 
    #{'lailv':'LAI_LV-M2M2:ERA5L:5022:1:0:1:0'}, # Leaf area index, low vegetation (m2 m-2) 
    #{'lshf':'LSHF:ERA5L:5022:1:0:1:0'}, # Lake shape factor (dimensionless)
    #{'sp':'PGR-PA:ERA5L:5022:1:0:1:0'}, # Surface pressure/ Pressure on ground (Pa)
    #{'sd':'SD-M:ERA5L:5022:1:0:1:0'}, # Snow depth (m of water eq)
    #{'hsnow':'HSNOW-M:ERA5L:5022:1:0:0'}, # Snow depth (m)
    #{'smlt':'SMLT-M:ERA5L:5022:1:0:1:0'}, # Snowmelt (m of water eq)
    #{'rsn':'SND-KGM3:ERA5L:5022:1:0:1:0'}, # Snow density (kg m-3) 
    #{'snowc':'SNOWC:ERA5L:5022:1:0:0'}, # Snow cover (%)
    #{'src':'SRC-M:ERA5L:5022:1:0:1:0'}, # Skin reservoir content (m of water eq)
    #{'stl1':'STL1-K:ERA5L:5022:9:7:1:0'}, # Soil temperature level 1 (K) 
    #{'stl2':'TSOIL-K:ERA5L:5022:9:1820:1:0'}, # Soil temperature level 2 (K)
    #{'stl3':'STL3-K:ERA5L:5022:9:7268:1:0'}, # Soil temperature level 3 (K)
    #{'stl4':'STL4-K:ERA5L:5022:9:25855:1:0'}, # Soil temperature level 4 (K)
    #{'swvl1':'SOILWET-M3M3:ERA5L:5022:9:7:1:0'}, # Soil wetness layer 1 (m3 m-3)  
    #{'swvl2':'SWVL2-M3M3:ERA5L:5022:9:1820:1:0'}, # Soil wetness layer 2 (m3 m-3)  
    #{'swvl3':'SWVL3-M3M3:ERA5L:5022:9:7268:1:0'}, # Soil wetness layer 3 (m3 m-3)  
    #{'swvl4':'SWVL4-M3M3:ERA5L:5022:9:25855:1:0'}, # Soil wetness layer 4 (m3 m-3)  
    {'t2':'T2-K:ERA5L:5022:1:0:1:0'}, # 2 metre temperature (K)  
    #{'td2':'TD2-K:ERA5L:5022:1:0:1:0'}, # 2 metre dewpoint temperature (K)  
    #{'tsn':'TSN-K:ERA5L:5022:1:0:1:0'}, # Temperature of snow layer (K) 
    #{'u10':'U10-MS:ERA5L:5022:1:0:1:0'}, # 10 metre U wind component (m s-1)  
    #{'v10':'V10-MS:ERA5L:5022:1:0:1:0'}, # 10 metre V wind component (m s-1)  
]

y_start='2015'
y_end='2022'
#rr_yrs=list(range(int(y_start),int(y_end)+1))
sd_yrs=list(range(int(y_start),int(y_end)+1))
nan=float('nan')
source='desm.harvesterseasons.com:8080'

### 00 UTC parameters (24h accumulated)
hour='00'
start='20150101T000000Z'
end='20221231T000000Z'
for pardict in era5l00:
    key,value=list(pardict.items())[0]
    df=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
#    print(key)
#    print(df)
    for point in llpdict.keys():
        dfpoint = df[df['pointID'] == point]
        dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-'+key+'_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)

### 00 and 12 UTC parameters
for pardict in era5l0012:
    hour='00'
    start='20150101T000000Z'
    end='20221231T000000Z'
    key,value=list(pardict.items())[0]
    df=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
#    print(key)
    df.rename({key:key+'-00'}, axis=1, inplace=True)
#    print(df)
    for point in llpdict.keys():
        dfpoint = df[df['pointID'] == point]
        dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-'+key+'-00_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
    hour='12'
    start='20150101T000000Z'
    end='20221231T000000Z' # should be 20230101T000000Z but we don't have data for 2023(?)
    df=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
    df.rename({key:key+'-12'}, axis=1, inplace=True)
    df['utctime']=df['utctime'].dt.date
#    print(df)
    for point in llpdict.keys():
        dfpoint = df[df['pointID'] == point]
        dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-'+key+'-12_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)

### Rolling cumsums
hour='00'
start='20140901T000000Z'
end='20221231T000000Z'

### Runoff rolling cumsums fot t-5, t-15, t-60 and t-100 days
rodict={'ro':'RO-M:ERA5L:5022:1:0:1:0'} # Total precipiation in meters (m), 24h sum (for timesteps previous day!)
df=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,rodict,llpdict)    
df=df.set_index('utctime')
df['utctime']=pd.to_datetime(df.index)
print(df)
#df5=fcts.rolling_cumsum(df.copy(),'5d','ro')
df15=fcts.rolling_cumsum(df.copy(),'15d','ro')
#df60=fcts.rolling_cumsum(df.copy(),'60d','ro')
#df100=fcts.rolling_cumsum(df.copy(),'100d','ro')
df = df.loc[(df['utctime'] >= '2015-01-01')]   
for point in llpdict.keys():
    dfpoint = df[df['pointID'] == point]
    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-ro'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
#    dfpoint = df5[df5['pointID'] == point]
#    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-ro5d_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
    dfpoint = df15[df15['pointID'] == point]
    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-ro15d_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
#    dfpoint = df60[df60['pointID'] == point]
#    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-ro60d_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
#    dfpoint = df100[df100['pointID'] == point]
#    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-ro100d_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)

### Surface runoff rolling cumsums fot t-5, t-15, t-60 and t-100 days
#srodict={'sro':'SRO-M:ERA5L:5022:1:0:1:0'} # Surface runoff (m)
#df=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,srodict,llpdict)    
#df=df.set_index('utctime')
#df['utctime']=pd.to_datetime(df.index)
#df5=fcts.rolling_cumsum(df.copy(),'5d','sro')
#df15=fcts.rolling_cumsum(df.copy(),'15d','sro')
#df60=fcts.rolling_cumsum(df.copy(),'60d','sro')
#df100=fcts.rolling_cumsum(df.copy(),'100d','sro')
#df = df.loc[(df['utctime'] >= '2015-01-01')]   
#for point in llpdict.keys():
#    dfpoint = df[df['pointID'] == point]
#    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-sro'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
#    dfpoint = df5[df5['pointID'] == point]
#    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-sro5d_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
#    dfpoint = df15[df15['pointID'] == point]
#    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-sro15d_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
#    dfpoint = df60[df60['pointID'] == point]
#    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-sro60d_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
#    dfpoint = df100[df100['pointID'] == point]
#    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-sro100d_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)

### Sub-surface runoff rolling cumsums fot t-5, t-15, t-60 and t-100 days
#ssrodict={'ssro':'SSRO-M:ERA5L:5022:1:0:1:0'} # Sub-surface runoff (m)  
#df=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,ssrodict,llpdict)    
#df=df.set_index('utctime')
#df['utctime']=pd.to_datetime(df.index)
#df5=fcts.rolling_cumsum(df.copy(),'5d','ssro')
#df15=fcts.rolling_cumsum(df.copy(),'15d','ssro')
#df60=fcts.rolling_cumsum(df.copy(),'60d','ssro')
#df100=fcts.rolling_cumsum(df.copy(),'100d','ssro')
#df = df.loc[(df['utctime'] >= '2015-01-01')]   
#for point in llpdict.keys():
#    dfpoint = df[df['pointID'] == point]
#    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-ssro'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
#    dfpoint = df5[df5['pointID'] == point]
#    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-ssro5d_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
#    dfpoint = df15[df15['pointID'] == point]
#    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-ssro15d_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
#    dfpoint = df60[df60['pointID'] == point]
#    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-ssro60d_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
#    dfpoint = df100[df100['pointID'] == point]
#    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-ssro100d_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)

### Potential evaporation rolling cumsums fot t-5, t-15, t-60 and t-100 days
#pevdict={'evapp':'EVAPP-M:ERA5L:5022:1:0:1:0'} # Potential evaporation (m)
#df=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pevdict,llpdict)    
#df=df.set_index('utctime')
#df['utctime']=pd.to_datetime(df.index)
#df5=fcts.rolling_cumsum(df.copy(),'5d','evapp')
#df15=fcts.rolling_cumsum(df.copy(),'15d','evapp')
#df60=fcts.rolling_cumsum(df.copy(),'60d','evapp')
#df100=fcts.rolling_cumsum(df.copy(),'100d','evapp')
#df = df.loc[(df['utctime'] >= '2015-01-01')]   
#for point in llpdict.keys():
#    dfpoint = df[df['pointID'] == point]
#    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-evapp'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
#    dfpoint = df5[df5['pointID'] == point]
#    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-evapp5d_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
#    dfpoint = df15[df15['pointID'] == point]
#    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-evapp15d_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
#    dfpoint = df60[df60['pointID'] == point]
#    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-evapp60d_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
#    dfpoint = df100[df100['pointID'] == point]
#    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-evapp100d_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)

### Evaporation rolling cumsums fot t-5, t-15, t-60 and t-100 days
evapdict={'evap':'EVAP-M:ERA5L:5022:1:0:1:0'} # Total precipiation in meters (m), 24h sum (for timesteps previous day!)
df=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,evapdict,llpdict)    
df=df.set_index('utctime')
df['utctime']=pd.to_datetime(df.index)
#df5=fcts.rolling_cumsum(df.copy(),'5d','evap')
df15=fcts.rolling_cumsum(df.copy(),'15d','evap')
#df60=fcts.rolling_cumsum(df.copy(),'60d','evap')
#df100=fcts.rolling_cumsum(df.copy(),'100d','evap')
df = df.loc[(df['utctime'] >= '2015-01-01')]   
for point in llpdict.keys():
    dfpoint = df[df['pointID'] == point]
    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-evap'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
#    dfpoint = df5[df5['pointID'] == point]
#    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-evap5d_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
    dfpoint = df15[df15['pointID'] == point]
    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-evap15d_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
 #   dfpoint = df60[df60['pointID'] == point]
 #   dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-evap60d_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
 #   dfpoint = df100[df100['pointID'] == point]
 #   dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-evap100d_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)

### Precipitation rolling cumsums fot t-5, t-15, t-60 and t-100 days
tpdict={'tp':'RR-M:ERA5L:5022:1:0:1:0'} # Total precipiation in meters (m), 24h sum (for timesteps previous day!)
df=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,tpdict,llpdict)    
df=df.set_index('utctime')
df['utctime']=pd.to_datetime(df.index)
#df5=fcts.rolling_cumsum(df.copy(),'5d','tp')
df15=fcts.rolling_cumsum(df.copy(),'15d','tp')
#df60=fcts.rolling_cumsum(df.copy(),'60d','tp')
#df100=fcts.rolling_cumsum(df.copy(),'100d','tp')
df = df.loc[(df['utctime'] >= '2015-01-01')]   
for point in llpdict.keys():
    dfpoint = df[df['pointID'] == point]
    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-tp_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
#    dfpoint = df5[df5['pointID'] == point]
#    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-tp5d_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
    dfpoint = df15[df15['pointID'] == point]
    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-tp15d_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
#    dfpoint = df60[df60['pointID'] == point]
#    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-tp60d_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
#    dfpoint = df100[df100['pointID'] == point]
#    dfpoint.to_csv(data_dir+'snowdepth/era5l-fixed/era5l-tp100d_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))