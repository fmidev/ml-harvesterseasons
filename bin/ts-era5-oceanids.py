#!/usr/bin/env python3
import time, warnings,requests,json
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# SmarMet-server timeseries query to fetch ERA5 training data for ML
# remember to: conda activate xgb 

startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/OCEANIDS/'

predictors = [
        ##{'u10':'U10-MS:ERA5:5021:1:0:1:0'}, # 10m u-component of wind
        ##{'v10':'V10-MS:ERA5:5021:1:0:1:0'}, # 10m v-component of wind
        ##{'fg10':'FFG-MS:ERA5:5021:1:0:1:0'}, # 10m wind gust since previous post-processing AINA EDELLINEN TUNTI HAE ERIKSEEN
        {'td2':'TD2-K:ERA5:5021:1:0:1:0'}, # 2m dewpoint temperature
        #{'t2':'T2-K:ERA5:5021:1:0:1:0'}, # 2m temperature
        #{'ewss':'EWSS-NM2S:ERA5:5021:1:0:1:0'}, # eastward turbulent surface stress
        #{'e':'EVAP-M:ERA5:5021:1:0:1:0'}, # evaporation
        #{'lsm':'LC-0TO1:ERA5:5021:1:0:1:0'}, # land-sea mask
        #{'msl':'PSEA-HPA:ERA5:5021:1:0:1:0'}, # mean sea level pressure
        #{'nsss':'NSSS-NM2S:ERA5:5021:1:0:1:0'}, # northward turbulent surface stress
        #{'tsea':'TSEA-K:ERA5:5021:1:0:1'}, # sea surface temperature
        #{'slhf':'FLLAT-JM2:ERA5:5021:1:0:1:0'}, # surface latent heat flux
        #{'ssr':'RNETSWA-JM2:ERA5:5021:1:0:1:0'}, # surface net solar radiation
        #{'str':'RNETLWA-JM2:ERA5:5021:1:0:1:0'}, # surface net thermal radiation
        #{'sshf':'FLSEN-JM2:ERA5:5021:1:0:1:0'}, # surface sensible heat flux
        #{'ssrd':'RADGLOA-JM2:ERA5:5021:1:0:1:0'}, # surface solar radiation downwards
        #{'strd':'RADLWA-JM2:ERA5:5021:1:0:1:0'}, # surface thermal radiation downwards
        #{'tcc':'N-0TO1:ERA5:5021:1:0:1:0'}, # total cloud cover
        #{'tlwc':'TCLW-KGM2:ERA5:5021:1:0:1:0'}, # total column cloud liquid water
        #{'tp':'RR-M:ERA5:5021:1:0:1:0'} # total precipitation
]

source='desm.harvesterseasons.com:8080' # server for timeseries query
bbox='24.9459,60.45867,25.4459,59.95867' # Vuosaari harbor region, 4 grid points
start='20130701T000000Z' # 2013-2023 period for ML fitting as observations (predictand) available 2013 onward
end='20231231T210000Z'
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
    
    # save dataframe as csv
    df.to_csv(data_dir+'era5-oceanids-'+name+'-'+start+'-'+end+'-all.csv',index=False) 

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))