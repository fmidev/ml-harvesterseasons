#!/usr/bin/env python3
import time, warnings,requests,json
import pandas as pd
import functions as fcts
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# SmarMet-server timeseries query to fetch ERA5 training data for ML
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
start='2020-08-25T00:00:00'
end='2020-08-26T00:00:00'

# Timeseries query
query='http://'+source+'/fmi-apikey/'+apikey+'/timeseries?FMISID='+fmisid+'&producer=observations_fmi&precision=double&timeformat=sql&tz=utc&starttime='+start+'&endtime='+end+'&format=debug&param=utctime,latitude,longitude,FMISID,'+predictands
print(query)
#print(query.replace(apikey, 'you-need-fmiapikey-here'))

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))