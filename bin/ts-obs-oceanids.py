#!/usr/bin/env python3
import time, warnings,requests,json
import pandas as pd
import functions as fcts
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# SmarMet-server timeseries query to fetch predictand data for ML
# remember to: conda activate xgb 

startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/OCEANIDS/'

# read in fmi-apikey from file
f=open("fmi-apikey","r")
lines=f.readlines()
apikey=lines[0]
f.close()

predictands='WS_PT1H_AVG,WG_PT1H_MAX'
source='data.fmi.fi'
fmisid='151028'
start='20130701T000000Z'
end='20231231T210000Z'
tstep='3h'
# Timeseries query
query='http://'+source+'/fmi-apikey/'+apikey+'/timeseries?FMISID='+fmisid+'&producer=observations_fmi&precision=double&timeformat=sql&tz=utc&starttime='+start+'&endtime='+end+'&timestep='+tstep+'&format=json&param=utctime,latitude,longitude,FMISID,'+predictands
print(query)
#print(query.replace(apikey, 'you-need-fmiapikey-here'))
response=requests.get(url=query)
results_json=json.loads(response.content)
#print(results_json)    
df=pd.DataFrame(results_json)  
df.columns=['utctime','latitude','longitude','FMISID','WS_PT1H_AVG','WG_PT1H_MAX'] # change headers      

# add day of year and hour of day as columns
df['utctime']=pd.to_datetime(df['utctime'])
df['dayOfYear'] = df['utctime'].dt.dayofyear
df['hour'] = df['utctime'].dt.hour
print(df)

# save dataframe as csv
df.to_csv(data_dir+'obs-oceanids-'+start+'-'+end+'-all.csv',index=False) 

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))