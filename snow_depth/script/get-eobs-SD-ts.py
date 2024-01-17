#!/usr/bin/env python3
import requests,os,time,glob,json,sys,re
import functions as fcts
import pandas as pd
import numpy as np

startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/'
eobs_dir='/home/ubuntu/data/eobs/sd_blend/'

def dms2dd(input):
    # EOBS lat lon +-degrees:minutes:seconds to degrees
    degrees,minutes,seconds=re.split('[:]',input)
    sign=degrees[0]
    degrees=degrees[1:]
    dd=float(degrees)+float(minutes)/60+float(seconds)/3600
    if sign=='-':
        dd=dd*-1.0
    return dd

# Read in station information and convert LAT&LON DMS to degrees
latloninfo=eobs_dir+'stations2.txt'
df_latlon=pd.read_csv(latloninfo, sep=',')
df_latlon['STANAME'] = df_latlon['STANAME'].str.strip()
df_latlon['CN'] = df_latlon['CN'].str.strip()
df_latlon['LAT'] = df_latlon['LAT'].str.strip()
df_latlon['LON'] = df_latlon['LON'].str.strip()
df_latlon['LAT'] = df_latlon['LAT'].apply(dms2dd)
df_latlon['LON'] = df_latlon['LON'].apply(dms2dd)
#print(df_latlon)
# use only EU domain
df_latlon = df_latlon[(df_latlon.LAT <= 75) & (df_latlon.LAT >= 25) & (df_latlon.LON >= -30) & (df_latlon.LON <= 50)]
#print(df_latlon)

print(df_latlon)
df_latlon.to_csv(data_dir+'EOBS_orography_EU_sd_noFIN.csv',index=False) 

points=df_latlon['STAID'].values.tolist()
staids=[]
for staid in points:
    staid=str(staid)
    if len(staid)<6:
        staid=(6-len(staid))*'0'+staid
        staids.append(staid)

# timeseries 2000-2020
y_start='2000'
y_end='2020'
sd_yrs=list(range(int(y_start),int(y_end)+1))

exists=[]
for staid in staids:
    staid=str(staid)
    print(str(int(staid)))
    sd_fname=eobs_dir+'/SD_STAID'+staid+'.txt'
    if os.path.exists(sd_fname): # check that file for each staid exists
        exists.append(str(int(staid)))
        sd_data=pd.DataFrame()
        sd_in=pd.read_csv(open(sd_fname,errors='replace'),sep=',',skiprows=22,skipinitialspace=True)
        sd_in['DATE']=pd.to_datetime(sd_in['DATE'],format='%Y%m%d')
        for y in sd_yrs:
            sd_data=pd.concat([sd_data,sd_in[sd_in['DATE'].dt.year == y]],ignore_index=True)
        sd_data['LAT']=df_latlon.LAT[df_latlon['STAID'] == int(staid)].item()
        sd_data['LON']=df_latlon.LON[df_latlon['STAID'] == int(staid)].item()
        sd_data['HGHT']=df_latlon.HGHT[df_latlon['STAID'] == int(staid)].item()
        sd_data=sd_data[['STAID','DATE','SD','LAT','LON','HGHT']]
        sd_data=sd_data.replace(-9999,np.nan)
        #rr_data['RR']=rr_data['RR']/10000.0 # from x0.1mm to m
        # should add here if df is empty, don't save (timeseries outside 2000-2020), also to exists list
        sd_data.to_csv(data_dir+'snowdepth/eobs/eobs_sd_'+y_start+'-'+y_end+'_'+str(int(staid))+'.csv',index=False)

# write list of existing staids in EU to file
df = pd.DataFrame(exists)
df.to_csv(data_dir+'snowdepth/EOBSstaids-sd-exists.csv',index=False)
