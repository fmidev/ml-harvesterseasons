#!/usr/bin/env python3
import time, warnings,requests,json
import pandas as pd
#from Vuosaari_151028 import *
#from Raahe_101785 import *
from Rauma_101061 import *

warnings.simplefilter(action='ignore', category=FutureWarning)
# SmarMet-server timeseries query to fetch predictand data for ML
# remember to: conda activate xgb 

startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/OCEANIDS/'+harbor+'/'

# read in fmi-apikey from file
f=open("fmi-apikey","r")
lines=f.readlines()
apikey=lines[0]
f.close()

predictand=qpred
print(qpred)
source='data.fmi.fi'

# Timeseries query
query='http://'+source+'/fmi-apikey/'+apikey+'/timeseries?FMISID='+FMISID+'&producer=observations_fmi&precision=double&timeformat=sql&tz=utc&starttime='+start+'&endtime='+end+'&hour=0&format=json&param=utctime,latitude,longitude,FMISID,'+predictand
print(query)
#print(query.replace(apikey, 'you-need-fmiapikey-here'))
response=requests.get(url=query)
results_json=json.loads(response.content)
#print(results_json)    
df=pd.DataFrame(results_json)  
df.columns=['utctime','latitude','longitude','FMISID',pred] # change headers      

# add day of year and hour of day as columns
df['utctime']=pd.to_datetime(df['utctime'])
df['dayOfYear'] = df['utctime'].dt.dayofyear
df['hour'] = df['utctime'].dt.hour
print(df)

# save dataframe as csv
df.to_csv(data_dir+obsfile,index=False) 

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))