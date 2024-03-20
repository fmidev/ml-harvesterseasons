#!/usr/bin/env python3
import requests, os, time, glob, json,sys
import pandas as pd
import functions as fcts
import numpy as np
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
### SmarMet-server timeseries query to fetch CLMS training data for machine learning
# SWE, FMI key CLMS HSNOW-M:CLMS:5051:1:0:1 (ml.harvesterseasons)
# SWE (CLMS SD-M SNOW depth in meters), FMI Key SD-M:CLMS:5051:1:0:1 (desm.harvesterseasons)
# CLMS NDVI (Normalised difference vegetation index), FMI Key NDVI:CLMS:5052:1:0:0 (desm.harvesterseasons)

startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/'
#eobspoints=data_dir+'EOBS_orography_EU_sd_set1_1500.csv'
#eobspoints=data_dir+'EOBS_orography_EU_sd_set2_2500.csv'
#eobspoints=data_dir+'EOBS_orography_EU_sd_set3_3000.csv'
eobspoints=data_dir+'EOBS_orography_EU_sd_set4_2657.csv'

def smartmet_ts_query_multiplePointsByID_tstep(source,start,end,tstep,pardict,llpdict):
    #timeseries query to smartmet-server
    #start date, end date, timestep, list of lats&lons, parameters as dictionary
    #returns dataframe
    latlons=[]
    for pair in llpdict.values():
        for i in pair:
            latlons.append(i)
    staids=[]
    for key in llpdict.keys():
        staids.append(key)
    
    # Timeseries query
    query='http://'+source+'/timeseries?latlons='
    for nro in latlons:
        query+=str(nro)+','
    query=query[0:-1]
    query+='&param=utctime,'
    for par in pardict.values():
        query+=par+','
        query=query[0:-1]
    query+='&starttime='+start+'&endtime='+end+'&timestep='+tstep+'&format=json&precision=full&tz=utc&timeformat=sql&grouplocations=1'
    #query+='&starttime='+start+'&endtime='+end+'&timestep=data&format=json&precision=full&producer=CLMS&tz=utc&timeformat=sql&grouplocations=1'
    print(query)

    # Response to dataframe
    response=requests.get(url=query)
    results_json=json.loads(response.content)
    #print(results_json)
    for i in range(len(results_json)):
        res1=results_json[i]
        for key,val in res1.items():
            if key!='utctime':   
                res1[key]=val.strip('[]').split()
    df=pd.DataFrame(results_json)   
    df.columns=['utctime']+list(pardict.keys()) # change headers to params.keys
    df['utctime']=pd.to_datetime(df['utctime'])
    expl_cols=list(pardict.keys())
    df=df.explode(expl_cols)
    df['latlonsID'] = df.groupby(level=0).cumcount().add(1).astype(str).radd('')
    df=df.reset_index(drop=True)
    df['latlonsID'] = df['latlonsID'].astype('int')
    max=df['latlonsID'].max()
    df['latitude']=''
    df['longitude']=''
    df['pointID']=''
    i=1
    j=0
    while i <= max:
        df.loc[df['latlonsID']==i,'latitude']=latlons[j]
        df.loc[df['latlonsID']==i,'longitude']=latlons[j+1]
        df.loc[df['latlonsID']==i,'pointID']=staids[i-1]
        j+=2
        i+=1
    df=df.drop(columns='latlonsID')
    df=df.astype({col: 'float32' for col in df.columns[1:-1]})
    #print(df)
    return df


cols_own=['STAID','LAT','LON']
eobs_df=pd.read_csv(eobspoints,usecols=cols_own)
lat=eobs_df['LAT'].values.tolist()
lon=eobs_df['LON'].values.tolist()
points=eobs_df['STAID'].values.tolist()
llpdict={i:[j, k] for i, j, k in zip(points,lat, lon)}
#print(llpdict)

# EXAMPLE get subdict based on list of pointids:
'''pointids=[47982724,48022714,48022732]
llpdict = dict((k, llpdict[k]) for k in pointids
           if k in llpdict)
print(llpdict)
'''

clms00 = [
#    {'swe_clms':'HSNOW-M:CLMS:5051:1:0:1'}, # HSNOW-M (Surface snow thickness) (ml.harvesterseasons)
    {'swe_clms':'SD-M:CLMS:5051:1:0:1'}, # Snow depth (m) (desm.harvesterseasons)
#    {'ndvi_clms':'NDVI:CLMS:5052:1:0:0'}, # Normalized difference vegetation index  (desm.harvesterseasons)
]

y_start='2020'
#y_start='2020'
y_end='2020'
#y_end='2020'
sd_yrs=list(range(int(y_start),int(y_end)+1))
nan=float('nan')
source='desm.harvesterseasons.com:8080' # used for 2020 (fetch data 15.2.2024)
#source='ml.harvesterseasons.com:8080'   # use this for CLMS (6.2.2024)

### Note: SD-M is s daily product and NDVI is 10-daily product (day 1, day 11 and day 21)
#start=y_start+'0101T000000Z'
#end=y_end+'1231T000000Z'
#start=y_start+'0101T000000'
#end=y_end+'1231T000000'
start=y_start+'-01-01T00:00:00Z'
end=y_end+'-12-31T23:59:59Z'
tstep='data'
for pardict in clms00:
    key,value=list(pardict.items())[0]
    df=smartmet_ts_query_multiplePointsByID_tstep(source,start,end,tstep,pardict,llpdict)

    for point in llpdict.keys():
        dfpoint = df[df['pointID'] == point]
        #dfpoint=dfpoint.replace(np.nan,0)  # NDVI: replace missing data (NaN) with 0
        dfpoint.to_csv(data_dir+'snowdepth/clms_temp/'+key+'_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)


executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))
