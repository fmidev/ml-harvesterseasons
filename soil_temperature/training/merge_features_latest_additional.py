import os, time, warnings,sys
import pandas as pd
import numpy as np
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
np.random.seed(42) 
import random
warnings.filterwarnings("ignore")

time_series_data = "/home/ubuntu/data/ML/training-data/soiltemp/timeseries_features_latest_add.csv"
tif_features_data = "/home/ubuntu/data/ML/training-data/soiltemp/soil_temp_tif_features_ts_add.csv"
train_data = "/home/ubuntu/data/ML/training-data/soiltemp/train_data_latest_additional.csv"

#forest_temp_df = pd.read_csv(forest_temp_data)
time_series_df  = pd.read_csv(time_series_data)
tif_features_df = pd.read_csv(tif_features_data)
tif_features_df.columns.values[0:2] =["longitude", "latitude"]

pd.options.display.max_columns = None

# function to merge dfs
def merge_df(df1,df2,keys,how="inner"):
    return pd.merge(df1, df2, how=how, on=keys)

path_filand_station = "/home/ubuntu/data/ismn/soiltemp_0.05_all_stations_00utc.txt"
europe_soil_temp = "/home/ubuntu/data/soiltempdb/Europe_soil_timeseries_combined_00utc.txt"
path_eu_data_1 = "/home/ubuntu/ml-harvesterseasons/soil_temperature/data/Europe_soil_metadata.txt"
path_eu_data_2 = "/home/ubuntu/data/soiltempdb/Europe_soil_metadata2.txt"
source = 'desm.harvesterseasons.com:8080'

# filter stations with "meta_ts_sensor_height" of height = -5
eu_data_1 = pd.read_csv(path_eu_data_1,sep=";",usecols=['meta_ts_owner_id',
                                                        'site_lat',
                                                        'site_long',
                                                        'meta_ts_sensor_height'
                              ])
eu_data_2 = pd.read_csv(path_eu_data_2,sep=";",usecols=['meta_ts_owner_id',
                                                        'site_lat',
                                                        'site_long',
                                                        'meta_ts_sensor_height'])
pd.options.display.max_columns = None

# read finland stations and name columns to match with europe data
fin_data = pd.read_csv(path_filand_station,
                       header=None,sep="\s+",
                       usecols=[0,4,7,8,12],
                       names=['clim_ts_date_time_local',
                              'meta_ts_owner_id',
                              'site_lat',
                              'site_long',
                              'clim_ts_value'
                              ])

def concat_df(dfs):
    result = pd.concat(dfs)
    return result
    
eu_data = pd.DataFrame()
eu_data = concat_df([eu_data_1,eu_data_2])
eu_data = eu_data.dropna(subset=['site_lat','site_long'])
eu_data = eu_data.loc[~eu_data['meta_ts_owner_id'].str.contains('Moisture')]
eu_data = eu_data.loc[(eu_data['meta_ts_sensor_height'] <= -5) & (eu_data['meta_ts_sensor_height'] >= -8)]
eu_data = eu_data.drop_duplicates(subset=['site_lat','site_long'])
fin_data=fin_data.drop_duplicates(subset=['site_lat','site_long'])

eu_data = eu_data.drop(['meta_ts_sensor_height'],axis=1)

eu_combined_data = concat_df([eu_data,fin_data.drop(["clim_ts_date_time_local",
                                            "clim_ts_value"],
                                           axis=1)])
print(len(eu_combined_data))
point_ids = list(range(len(eu_combined_data)))
eu_combined_data['pointID'] = point_ids

# split back eu_data to combine with soiltemp
eu_data = eu_combined_data.iloc[:len(eu_data),:]

# get point ids to finland, rest of the point ids are for finland after the length of eu_data
finland_point_ids = list(range(len(eu_data),len(eu_combined_data)))
fin_data['pointID'] = finland_point_ids

def read_csv(filepath,columns):
    df = pd.read_csv(filepath,sep=";",on_bad_lines='warn',engine='python',usecols=columns)
    return df

def filter_year(df,yr):
    #df['year'] = pd.to_datetime(df['clim_ts_date_time_local']).dt.year
    df['clim_ts_date_time_local'] = pd.to_datetime(df["clim_ts_date_time_local"].str[0:10])
    df['year'] = df['clim_ts_date_time_local'].dt.year
    df = df.loc[(df['year'] > yr )]
    return df

soiltemp_europe_df = read_csv(europe_soil_temp,columns=["meta_ts_owner_id",
                                                        "clim_ts_date_time_local",
                                                        "clim_ts_value"])


# filter only temparature values

soiltemp_labels_df = soiltemp_europe_df.loc[~soiltemp_europe_df['meta_ts_owner_id'].str.contains('Moisture')]

soiltemp_labels_europe = filter_year(soiltemp_labels_df,2014)

eu_data = merge_df(eu_data,soiltemp_labels_europe,keys=["meta_ts_owner_id"])

# combine finland data after filtering

fin_data = filter_year(fin_data,2014)

eu_data = concat_df([eu_data,fin_data])

eu_data['clim_ts_date_time_local'] = pd.to_datetime(eu_data['clim_ts_date_time_local'])

eu_data = eu_data.groupby([pd.Grouper(key='clim_ts_date_time_local',
                                      freq='D'),
                                      "meta_ts_owner_id",
                                      'site_long',
                                      'site_lat',
                                      'pointID',
                                      ])['clim_ts_value'].mean().reset_index()

all_features = merge_df(time_series_df,
                        tif_features_df,
                        keys=['utctime','pointID'])

eu_data['clim_ts_date_time_local'] = eu_data['clim_ts_date_time_local'].dt.date.astype(str)

print(eu_data.head(1).T)

print(all_features.head(1).T)
print(all_features.head(5))
print(eu_data.head(5))
merged_df = pd.merge(all_features,eu_data,
                     left_on=["utctime","pointID"],
                     right_on=['clim_ts_date_time_local','pointID'],
                     how="inner")

merged_df.drop(columns=["latitude_x","latitude_y",
                        "longitude_x","longitude_y"],
               axis=1,inplace=True)
# save dataset to train_data.csv
merged_df.to_csv(train_data,index=False)
