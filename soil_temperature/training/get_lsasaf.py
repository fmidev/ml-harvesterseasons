#!/usr/bin/env python3
import requests, os, time, glob, json,sys
import pandas as pd
from functools import reduce
import functions as fcts
import numpy as np
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
### SmarMet-server timeseries query to fetch ERA5-Land training data for machine learning (copied from bin and edited )

startTime = time.time()

path_eu_data = "/home/ubuntu/ml-harvesterseasons/soil_temperature/data/Europe_soil_metadata.txt"
source = 'desm.harvesterseasons.com:8080'

eu_data = pd.read_csv(path_eu_data,sep=";")
pd.options.display.max_columns = None

#remove nan values in df

eu_data = eu_data.dropna(subset=['site_lat','site_long'])

# remove duplicate long,lat 

eu_data = eu_data.drop_duplicates(subset=['site_lat','site_long'])

lat = eu_data['site_lat'].tolist()
lon = eu_data['site_long'].tolist()
# points=lucas_df['POINT_ID'].values.tolist()

pointids = list(range(len(eu_data)))
# pointids = list(range(1,6000)) # can only acc  vö


llpdict = {i:[j, k] for i, j, k in zip(pointids,lat, lon)}

# EXAMPLE get subdict based on list of pointids:
pointids = list(range(1,10))
llpdict = dict((k, llpdict[k]) for k in pointids
           if k in llpdict)
print(llpdict)


### 00 UTC parameters (24h accumulated)
hour = '00'
start = '20150101T000000Z'
end = '20221231T000000Z'
# function to merge dfs
def merge_df(df1,df2):
    if df1.empty:
        df = pd.concat([df1,df2],axis=1)
        # df = df.T.drop_duplicates().T
    else:
        df = pd.merge(df1, df2, how='inner', on=['utctime','latitude','longitude','pointID'])
    return df

def multi_merger_df(data_frames):
    df =reduce(lambda  left,right: pd.merge(left,
                                                  right,
                                                  on=['utctime','latitude','longitude','pointID'],
                                            how='inner'), data_frames)
    return df

########### lsasaf##########################    
era5l0012 = [{'sktn':'SKT-K:LSASAFC:5064:1:0:0'}, # Skin temperature (K) 
             ]
start = '20000101T000000Z'
end = '20221231T000000Z'

era5l0012_df = pd.DataFrame()
for pardict in era5l0012:
    hour = '00'
    key,value = list(pardict.items())[0]
    temp_df = fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
    temp_df.rename({key:key+'-00'}, axis=1, inplace=True)
    temp_df['utctime']=temp_df['utctime'].dt.date
    era5l0012_df = merge_df(era5l0012_df,temp_df)

# era5l0012 = [{'sktd':'SKT-K:LSASAFC:5064:1:0:0'},
#              ] # Skin temperature (K) 
# start = '20000101T120000Z' 
# end = '20221231T120000Z'
# for pardict in era5l0012:
    
#     hour = '12'
#     key,value = list(pardict.items())[0]
#     temp_df = fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)

#     temp_df['utctime']=temp_df['utctime'].dt.date
    
#     temp_df.rename({key:key+'-12'}, axis=1, inplace=True)
   
#     era5l0012_df = merge_df(era5l0012_df,temp_df)

######################################
print("*******************era5l0012_df*********************")
print(era5l0012_df.dropna())
'''
era5l0012_df['pointID'] = era5l0012_df['pointID'].astype('Int64')
era5l0012_df['utctime'] = pd.to_datetime(era5l0012_df['utctime'])

### Rolling cumsums
hour='00'
start='20140901T000000Z'
end='20221231T000000Z'

def get_rolling_mean(start,
                     end,
                     hour,
                     config_dict,
                     llpdict,
                     rolling_days,
                     column_name):
    df=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,config_dict,llpdict)    
    df.set_index('utctime',inplace=True)
    final_rolling_df = pd.DataFrame()
    for each_day in rolling_days:
        each_days = str(each_day)+'d'
        rolling_df = fcts.rolling_cumsum(df.copy(),each_days,column_name)
        final_rolling_df = merge_df(final_rolling_df,rolling_df)
    return final_rolling_df

ro_df = get_rolling_mean(start=start,
                          end=end,
                          hour=hour,
                          config_dict={'ro':'RO-M:ERA5L:5022:1:0:1:0'},
                          llpdict=llpdict,
                          rolling_days=[5],
                          column_name='ro'
                          )

sro_df = get_rolling_mean(start=start,
                          end=end,
                          hour=hour,
                          config_dict={'sro':'SRO-M:ERA5L:5022:1:0:1:0'},
                          llpdict=llpdict,
                          rolling_days=[5],
                          column_name='sro'
                          )

ssro_df = get_rolling_mean(start=start,
                          end=end,
                          hour=hour,
                          config_dict={'ssro':'SSRO-M:ERA5L:5022:1:0:1:0'},
                          llpdict=llpdict,
                          rolling_days=[5],
                          column_name='ssro'
                          )

evapp_df = get_rolling_mean(start=start,
                          end=end,
                          hour=hour,
                          config_dict={'evapp':'EVAPP-M:ERA5L:5022:1:0:1:0'},
                          llpdict=llpdict,
                          rolling_days=[5],
                          column_name='evapp'
                          )

tp_df = get_rolling_mean(start=start,
                          end=end,
                          hour=hour,
                          config_dict={'tp':'RR-M:ERA5L:5022:1:0:1:0'},
                          llpdict=llpdict,
                          rolling_days=[5],
                          column_name='tp'
                          )


# combine all rolling df's

data_frames = [ro_df,sro_df,ssro_df,evapp_df,tp_df]

rolling_dfs = multi_merger_df(data_frames=data_frames)

# convert utctime to datatime after merging in the previous step it is becoming object
rolling_dfs['utctime'] = pd.to_datetime(rolling_dfs['utctime'])

# combine time series data

time_series_frames = [era5l00_df,era5l0012_df,rolling_dfs]

print(era5l0012_df.info())

time_series_df = multi_merger_df(data_frames=time_series_frames)

# save time_series_df 

time_series_df.to_csv("/home/ubuntu/data/ML/training-data/soiltemp/timeseries_features.csv",index=False)
'''

