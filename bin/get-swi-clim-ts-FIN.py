#!/usr/bin/env python3
import time, warnings
import pandas as pd
import functions as fcts
#import sys
warnings.simplefilter(action='ignore', category=FutureWarning)
### SmarMet-server timeseries query to fetch daily SWI climate training data for machine learning
# SWI1, SWI2, SWI3, SWI4 climate for predictors in fitting
# conda activate xgb

startTime=time.time()

# read in point-ids, latitudes and longitudes from LUCAS to a dictonary
data_dir='/home/ubuntu/data/ML/training-data/'
lucas=data_dir+'LUCAS_2018_Copernicus_attr+additions.csv'
cols_own=['POINT_ID','TH_LAT','TH_LONG']
lucas_df=pd.read_csv(lucas,usecols=cols_own)
lat=lucas_df['TH_LAT'].values.tolist()
lon=lucas_df['TH_LONG'].values.tolist()
points=lucas_df['POINT_ID'].values.tolist()
llpdict={i:[j, k] for i, j, k in zip(points,lat, lon)}
llpdict={k:llpdict[k] for k in list(llpdict.keys())[:5000]} # max 5000 points per ts query [:5000], [5000:10000], ...

'''
# EXAMPLE get subdict based on list of pointids:
pointids=[47982724,48022714,48022732]
llpdict = dict((k, llpdict[k]) for k in pointids
           if k in llpdict)
print(llpdict)
'''
### SWI climate predictands
swi = [
    #{'swi1clim':'SWI1:SWIC:5059:1:0:0'}, # Soil wetness index layer 1
    {'swi2clim':'SWI2:SWIC:5059:1:0:0'}, # Soil wetness index layer 2
    #{'swi3clim':'SWI3:SWIC:5059:1:0:0'}, # Soil wetness index layer 3
    #{'swi4clim':'SWI4:SWIC:5059:1:0:0'} # Soil wetness index layer 4 
    ]

y_start='2020'
y_end='2020'
source='desm.harvesterseasons.com:8080'
hour='00'

# swi climate data only for 20200101-20210731, create date column for teaching period 2015-2022 to match other timeseries data
dftime=pd.DataFrame()
dftime['utctime']=pd.date_range(start='2015-01-01', end='2022-12-31')    

### timeseries query to server
for pardict in swi:
    key,value=list(pardict.items())[0]
    start=y_start+'0101T000000Z'
    end=y_end+'1231T000000Z'
    df=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
    df=df.astype({'pointID': 'int32'})
    for point in llpdict.keys():
        dfpoint = df[df['pointID'] == point]
        dfpoint=pd.concat([dfpoint]*8, ignore_index=True) # repeat dataframe for whole timeseries range (8 years)
        dfpoint=dfpoint.drop([59,791,1157,1523,2255,2621]) # drop 29.2. by index if not a leap year
        dfpoint=dfpoint.reset_index(drop=True)
        dfpoint['utctime']=dftime['utctime'] # change utctime to 2015-2022 (from repeating 2020)
        dfpoint.to_csv(data_dir+'soilwater/swi-clim/'+key+'/'+key+'_2015-2022_'+str(point)+'.csv',index=False)

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))