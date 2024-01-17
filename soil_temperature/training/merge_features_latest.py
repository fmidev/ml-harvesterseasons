import os, time, warnings,sys
import pandas as pd
import numpy as np
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
np.random.seed(42) 
import random
warnings.filterwarnings("ignore")

forest_temp_data = "/home/ubuntu/data/ML/training-data/soiltemp/forest_temp_features.csv"
time_series_data = "/home/ubuntu/data/ML/training-data/soiltemp/timeseries_features_latest.csv"
tif_features_data = "/home/ubuntu/data/ML/training-data/soiltemp/soil_temp_tif_features_ts.csv"
train_data = "/home/ubuntu/data/ML/training-data/soiltemp/train_data_latest.csv"
soiltemp_path_1a = "/home/ubuntu/data/soiltempdb/Europe_soil_timeseries_1a.txt"
soiltemp_path_1b = "/home/ubuntu/data/soiltempdb/Europe_soil_timeseries_1b.txt"
#soiltemp_path_1c = "/home/ubuntu/data/soiltempdb/Europe_soil_timeseries_1b.txt"
soiltemp_path_1d = "/home/ubuntu/data/soiltempdb/Europe_soil_timeseries_1d.txt"

forest_temp_df = pd.read_csv(forest_temp_data)
time_series_df  = pd.read_csv(time_series_data)
tif_features_df = pd.read_csv(tif_features_data)
tif_features_df.columns.values[0:2] =["longitude", "latitude"]

pd.options.display.max_columns = None

# function to merge dfs
def merge_df(df1,df2,keys):
    return pd.merge(df1, df2, how='inner', on=keys)


path_eu_data = "/home/ubuntu/ml-harvesterseasons/soil_temperature/data/Europe_soil_metadata.txt"
# path_old_eu = "/home/ubuntu/ml-harvesterseasons/soil_temperature/data/Metadata_soil_Europe_LAI.txt"

# filter stations with "meta_ts_sensor_height" of height = -5
eu_data = pd.read_csv(path_eu_data,sep=";")
eu_data = eu_data.loc[eu_data['meta_ts_sensor_height'] == -5 ]

# LAI_data = pd.read_csv(path_old_eu,sep=";",nrows=50)


eu_data = eu_data.drop_duplicates(subset=['site_lat','site_long'])

point_ids = list(range(len(eu_data)))
eu_data['pointID'] = point_ids

def read_csv(filepath):
    df = pd.read_csv(filepath,sep=";",on_bad_lines='warn',engine='python')
    return df

def filter_year(df,yr):
    df['year'] = pd.to_datetime(df['clim_ts_date_time_local']).dt.year
    df = df.loc[(df['year'] > yr )]
    return df
def concat_df(dfs):
    result = pd.concat(dfs)
    return result

soiltemp_1a_df = read_csv(soiltemp_path_1a)
soiltemp_1b_df = read_csv(soiltemp_path_1b)
#soiltemp_1c_df = read_csv(soiltemp_path_1c)
soiltemp_1d_df = read_csv(soiltemp_path_1d)


soiltemp_labels_df = concat_df(dfs=[soiltemp_1a_df,
                                soiltemp_1b_df,
                                #soiltemp_1c_df,
                                soiltemp_1d_df
                                ])

# filter only temparature values

soiltemp_labels_df = soiltemp_labels_df.loc[~soiltemp_labels_df['meta_ts_owner_id'].str.contains('Moisture')]

soiltemp_labels_final = filter_year(soiltemp_labels_df,2014)

eu_data = merge_df(eu_data,soiltemp_labels_final,keys=["meta_ts_owner_id"])

eu_data.drop(['meta_ts_start_date','meta_ts_end_date', 
              'meta_ts_comment', 'meta_ts_flags', 'meta_ts_doi', 'file_name.x',
              'site_owner_id','meta_ts_logger_shielding','meta_ts_homemade_shield',
              'meta_ts_clim_unit','meta_ts_clim_variable','meta_ts_clim_variable',
              'meta_ts_sensor_height_range',
              'site_habitat',
              'site_subhabitat',
              'meta_ts_clim_accuracy',
              'meta_ts_clim_temporal_res',
              'year',
              'meta_ts_sensor_height', 'meta_ts_timezone','meta_ts_licence','file_name.y',
              'expe_name','logger_id','clim_ts_flag',], axis=1,inplace=True)#

eu_data['clim_ts_date_time_local'] = pd.to_datetime(eu_data['clim_ts_date_time_local'])

eu_data = eu_data.groupby([pd.Grouper(key='clim_ts_date_time_local',
                                      freq='D'),
                                      "meta_ts_owner_id",
                                      'site_long',
                                      'site_lat',
                                      'pointID',
                                      ])['clim_ts_value'].mean().reset_index()

all_features = merge_df(time_series_df,
                        forest_temp_df,
                        keys=['utctime','pointID']).merge(tif_features_df,
                                            how='inner', 
                                            on=['utctime','pointID'])

eu_data['clim_ts_date_time_local'] = eu_data['clim_ts_date_time_local'].dt.date.astype(str)

print(eu_data.head(1).T)

print(all_features.head(1).T)
merged_df = pd.merge(all_features,eu_data,
                     left_on=["utctime","pointID"],
                     right_on=['clim_ts_date_time_local','pointID'],
                     how="inner")

merged_df.drop(columns=["latitude_x","latitude_y",
                        "longitude_x","longitude_y"],
               axis=1,inplace=True)
# save dataset to train_data.csv
merged_df.to_csv(train_data,index=False)
