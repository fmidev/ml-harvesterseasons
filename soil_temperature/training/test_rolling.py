#!/usr/bin/env python3
import requests, os, time, glob, json,sys
from functools import reduce
import pandas as pd
import functions as fcts
import numpy as np
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

startTime = time.time()
source = 'desm.harvesterseasons.com:8080'
path_eu_data = "/home/ubuntu/ml-harvesterseasons/soil_temperature/data/Europe_soil_metadata.txt"
eu_data = pd.read_csv(path_eu_data,sep=";")
pd.options.display.max_columns = None

# function to merge dfs
def merge_df(df1,df2):
    if df1.empty:
        df = pd.concat([df1,df2],axis=1)
        # df = df.T.drop_duplicates().T
    else:
        df = pd.merge(df1, df2, how='inner', on=['utctime','latitude','longitude','pointID'])
    return df

#remove nan values in df

eu_data = eu_data.dropna(subset=['site_lat','site_long'])

lat = eu_data['site_lat'].tolist()
lon = eu_data['site_long'].tolist()
pointids = list(range(1,len(lon)))

llpdict = {i:[j, k] for i, j, k in zip(pointids,lat, lon)}


# EXAMPLE get subdict based on list of pointids:
pointids = list(range(1,10))
llpdict = dict((k, llpdict[k]) for k in pointids
           if k in llpdict)


### Rolling cumsums
hour='00'
start='20140901T000000Z'
end='20151231T000000Z'

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

rolling_dfs = reduce(lambda  left,right: pd.merge(left,
                                                  right,
                                                  on=['utctime','latitude','longitude','pointID'],
                                            how='inner'), data_frames)

print(rolling_dfs.head())