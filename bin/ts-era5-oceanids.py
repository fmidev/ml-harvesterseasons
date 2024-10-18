#!/usr/bin/env python3
import time, warnings,requests,json
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# SmarMet-server timeseries query to fetch ERA5 training data for ML
# remember to: conda activate xgb 

startTime=time.time()

def filter_points(df,lat,lon,nro,name):
    df0=df.copy()
    filter1 = df0['latitude'] == lat
    filter2 = df0['longitude'] == lon
    df0.where(filter1 & filter2, inplace=True)
    df0.columns=['lat-'+str(nro),'lon-'+str(nro),name+'-'+str(nro)] # change headers      
    df0=df0.dropna()
    return df0

data_dir='/home/ubuntu/data/ML/training-data/OCEANIDS/'

predictors = [
        {'u10':'U10-MS:ERA5:5021:1:0:1:0'}, # 10m u-component of wind
        {'v10':'V10-MS:ERA5:5021:1:0:1:0'}, # 10m v-component of wind
        {'fg10':'FFG-MS:ERA5:5021:1:0:1:0'}, # 10m wind gust since previous post-processing AINA EDELLINEN TUNTI HAE ERIKSEEN
        ##{'td2':'TD2-K:ERA5:5021:1:0:1:0'}, # 2m dewpoint temperature
        {'t2':'T2-K:ERA5:5021:1:0:1:0'}, # 2m temperature
        ##{'ewss':'EWSS-NM2S:ERA5:5021:1:0:1:0'}, # eastward turbulent surface stress
        ##{'e':'EVAP-M:ERA5:5021:1:0:1:0'}, # evaporation
        #{'lsm':'LC-0TO1:ERA5:5021:1:0:1:0'}, # land-sea mask
        ##{'msl':'PSEA-HPA:ERA5:5021:1:0:1:0'}, # mean sea level pressure
        ##{'nsss':'NSSS-NM2S:ERA5:5021:1:0:1:0'}, # northward turbulent surface stress
        ##{'tsea':'TSEA-K:ERA5:5021:1:0:1'}, # sea surface temperature
        ##{'slhf':'FLLAT-JM2:ERA5:5021:1:0:1:0'}, # surface latent heat flux
        ##{'ssr':'RNETSWA-JM2:ERA5:5021:1:0:1:0'}, # surface net solar radiation
        ##{'str':'RNETLWA-JM2:ERA5:5021:1:0:1:0'}, # surface net thermal radiation
        ##{'sshf':'FLSEN-JM2:ERA5:5021:1:0:1:0'}, # surface sensible heat flux
        ##{'ssrd':'RADGLOA-JM2:ERA5:5021:1:0:1:0'}, # surface solar radiation downwards
        ##{'strd':'RADLWA-JM2:ERA5:5021:1:0:1:0'}, # surface thermal radiation downwards
        ##{'tcc':'N-0TO1:ERA5:5021:1:0:1:0'}, # total cloud cover
        ##{'tlwc':'TCLW-KGM2:ERA5:5021:1:0:1:0'}, # total column cloud liquid water
        {'tp':'RR-M:ERA5:5021:1:0:1:0'} # total precipitation
]

source='desm.harvesterseasons.com:8080' # server for timeseries query
#bbox='24.9459,60.45867,25.4459,59.95867' # Vuosaari harbor region, 4 grid points
bbox='4.23222,36.41611,4.73222,36.91611' # Malaga port/airport region, 4 grid points
#start='20130701T000000Z' # 2013-2023 period for ML fitting as observations (predictand) available 2013 onward
#end='20231231T210000Z'
start='20000101T000000Z'
end='20230831T000000Z'
tstep='3h'

# Timeseries query
for pred in predictors:
    key,value=list(pred.items())[0]
    name=key
    print(key)
    query='http://'+source+'/timeseries?bbox='+bbox+'&param=utctime,latitude,longitude,'+value+'&starttime='+start+'&endtime='+end+'&timestep='+tstep+'&format=json&precision=full&tz=utc&timeformat=sql'
    print(query)
    response=requests.get(url=query)
    results_json=json.loads(response.content)
    #print(results_json)    
    for i in range(len(results_json)):
        res1=results_json[i]
        for key,val in res1.items():
            if key!='utctime':   
                res1[key]=val.strip('[]').split()
    df=pd.DataFrame(results_json)  
    df.columns=['utctime','latitude','longitude',name] # change headers      
    expl_cols=['latitude','longitude',name]
    df=df.explode(expl_cols)
    print(df)
    df.set_index('utctime',inplace=True)
    # filter points
    lat='36.5000000000000000'
    lon='4.25000000000000000'
    df1=filter_points(df,lat,lon,1,name)
    lat='36.7500000000000000'
    lon='4.2500000000000000'
    df2=filter_points(df,lat,lon,2,name)
    lat='36.7500000000000000'
    lon='4.500000000000000'
    df3=filter_points(df,lat,lon,3,name)
    lat='36.500000000000000'
    lon='4.500000000000000'
    df4=filter_points(df,lat,lon,4,name)
    # merge dataframes
    df_new = pd.concat([df1,df2,df3,df4],axis=1,sort=False).reset_index()
    # save to csv file
    df_new.to_csv(data_dir+'era5-oceanids-'+name+'-'+start+'-'+end+'-all-check.csv',index=False) 
    df_new = df_new.drop(['utctime', 'lat-1','lon-1','lat-2','lon-2','lat-3','lon-3','lat-4','lon-4'], axis=1)
    df_new.to_csv(data_dir+'era5-oceanids-'+name+'-'+start+'-'+end+'-all-use.csv',index=False) 
    
    # save utctime, lat/lon info to csv file (run once with u10, then comment out)
    #df_new = df_new.drop(['u10-1', 'u10-2','u10-3','u10-4'], axis=1)
    #print(df_new)
    #df_new.to_csv(data_dir+'era5-oceanids-utctime-lat-lon-'+start+'-'+end+'-all.csv',index=False) 
    
executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))