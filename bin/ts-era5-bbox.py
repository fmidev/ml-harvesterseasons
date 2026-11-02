#!/usr/bin/env python3
import time, warnings,requests,json,sys,os
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# SmarMet-server timeseries (ts) query to fetch ERA5/ERA5D data
# ts queries for grid points inside given bbox
# output is csv file
# EXAMPLE SCRIPT FOR CryoSCOPE
# usage: python ts-era5-bbox.py
# (AK 2025)

def ts_bbox_query(source,bbox,value,start,end,hours,name):
    ''' Timeseries query for bbox'''
    df=pd.DataFrame()
    query=f'http://{source}/timeseries?bbox={bbox}&param=utctime,latitude,longitude,{value}&starttime={start}&endtime={end}&hour={hours}&format=json&precision=full&tz=utc&timeformat=sql&origintime=20000101T000000Z'
    print(query) # debugging - copy to browser and change format=debug
    response=requests.get(url=query)
    results_json=json.loads(response.content)
    #print(results_json) # for debugging
    for i in range(len(results_json)):
        res1=results_json[i]
        for key,val in res1.items():
            if key!='utctime':   
                res1[key]=str(val).strip('[]').split()
    df=pd.DataFrame(results_json)  
    df.columns=['utctime','latitude','longitude',name] # change headers from FMI-KEYs to simpler names      
    expl_cols=['latitude','longitude',name]
    df=df.explode(expl_cols)
    return(df)

# path to your output directory
output_dir=f'/home/ubuntu/data/ML/'

# bbox latitudes and longitudes
latlons = {
        'min_lon': 23.604126,
        'min_lat': 59.642764,
        'max_lon': 26.251831,
        'max_lat': 60.777937        
    }
# convert to string for ts query
bbox = f"{latlons['min_lon']},{latlons['min_lat']},{latlons['max_lon']},{latlons['max_lat']}"

# ERA5 reanalysis parameters for query
# own name : FMI-KEY
ERA5_pars = [
    {'u10-ms':'U10-MS:ERA5:5021:1:0:1:0'},  # 10m u-component of wind (m/s)
    {'v10-ms':'V10-MS:ERA5:5021:1:0:1:0'},  # 10m v-component of wind (m/s)
    {'t2-K':'T2-K:ERA5:5021:1:0:1:0'},     # 2m temperature (K)
    {'t2-C':'SUM{T2-K:ERA5:5021:1:0:1:0;-273.15}'},     # 2m temperature (C)
    {'t850-K': 'T-K:ERA5:5021:2:850:1:0'}, # temperature (K), pressure level 850 hPa       
    {'t700-K': 'T-K:ERA5:5021:2:700:1:0'}, # temperature (K), pressure level 700 hPa 
    {'t500-K': 'T-K:ERA5:5021:2:500:1:0'}  # temperature (K), pressure level 500 hPa
]

# ERA5 daily statistics (ERA5D) parameters for query
# own name : FMI-KEY
ERA5D_pars = [
    {'tp-m':'RR-M:ERA5D:5021:1:0:1'},     # Previous day sum Total precipitation (m)
    {'tp-mm':'MUL{RR-M:ERA5D:5021:1:0:1;1000}'},     # Previous day sum Total precipitation (mm)
    {'mx2t-K':'TMAX-K:ERA5D:5021:1:0:1'}, # Previous day maximum Maximum temperature in the last 24h (K)
    {'mn2t-K':'TMIN-K:ERA5D:5021:1:0:1'}  # Previous day minimum Minimum temperature in the last 24h (K)
]

# SmartMet server hosting the data
source='desm.harvesterseasons.com:8080'

# start and end for timeseries period
# YYYYMMDDTHHMMSSZ
start='20200101T000000Z'
end='20221231T120000Z'

# timesteps for query 
hours_era5='00,12' # 00 and 12 UTC 
hours_era5d='00' # 00 UTC

# ERA5 queries
for pardict in ERA5_pars:
    key, value = list(pardict.items())[0]
    name=key # simpler name for output columns rather than FMI-KEY
    # ts query per parameter
    df_ERA5=ts_bbox_query(source,bbox,value,start,end,hours_era5,name)
    print(df_ERA5) # debugging
    # save to csv file per parameter
    df_ERA5.to_csv(f'{output_dir}ERA5_{name}_{start[:4]}-{end[:4]}.csv',index=False) 

# ERA5D queries
for pardict in ERA5D_pars:
    key, value = list(pardict.items())[0]
    name=key # simpler name for output columns rather than FMI-KEY
    # ts query per parameter
    df_ERA5=ts_bbox_query(source,bbox,value,start,end,hours_era5d,name)
    print(df_ERA5) # debugging
    # save to csv file per parameter
    df_ERA5.to_csv(f'{output_dir}ERA5D_{name}_{start[:4]}-{end[:4]}.csv',index=False) 