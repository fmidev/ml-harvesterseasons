from typing import List
import os, time, random, warnings,sys
import pandas as pd
import numpy as np
from functools import reduce
warnings.filterwarnings("ignore")

utc_csv = '/home/ubuntu/data/ML/training-data/soilwater/utctime_2015-2022.csv'
forest_temp_output = '/home/ubuntu/data/ML/training-data/soiltemp/'
forest_temp = '/home/ubuntu/data/soiltempdb/'


months_num = {'January':1,'February':2,'March':3,'April':4,'May':5,
              'June':6,'July':7,'August':8,'September':9,'October':10,
              'November':11,'December':12}


path_eu_data_1 = "/home/ubuntu/ml-harvesterseasons/soil_temperature/data/Europe_soil_metadata.txt"
path_eu_data_2 = "/home/ubuntu/ml-harvesterseasons/soil_temperature/data/Europe_soil_metadata2.txt"
source = 'desm.harvesterseasons.com:8080'

eu_data_1,eu_data_2 = pd.read_csv(path_eu_data_1,sep=";"),pd.read_csv(path_eu_data_2,sep=";")
pd.options.display.max_columns = None

#remove nan values in df

eu_data_1 = eu_data_1.dropna(subset=['site_lat','site_long'])
eu_data_2 = eu_data_2.dropna(subset=['site_lat','site_long'])

# remove duplicate long,lat 

eu_data_1 = eu_data_1.drop_duplicates(subset=['site_lat','site_long'])
eu_data_2 = eu_data_2.drop_duplicates(subset=['site_lat','site_long'])

def concat_df(dfs,cols):
    result = pd.concat(dfs)
    result = result.drop_duplicates(subset=cols)
    return result

eu_data = pd.DataFrame()
eu_data = concat_df([eu_data_1,eu_data_2],['site_lat','site_long'])

lat = eu_data['site_lat'].tolist()
lon = eu_data['site_long'].tolist()


pointids = list(range(len(eu_data))) 
eu_data['pointID'] = pointids

# save utc_time as df,create month column
utc_time_df = pd.read_csv(utc_csv,usecols=["utctime"])
utc_time_df['month'] = utc_time_df['utctime'].astype(str).str.split('-').str[1].astype(int)
# raster files : forest temp, soil temp

forest_T_min_ls = []
forest_T_max_ls = []
forest_T_mean_ls = []

for i in range(len(months_num)):

    month_str = list(months_num.keys())[list(months_num.values()).index(i+1)]
    f_min = f"minT_offset_{str(month_str)}.tif"
    f_max = f"maxT_offset_{str(month_str)}.tif"
    f_mean = f"meanT_offset_{str(month_str)}.tif"

    forest_T_min_ls.append(f_min)
    forest_T_max_ls.append(f_max)
    forest_T_mean_ls.append(f_mean)

def get_forest_values(forest_file_list:List,name:str,long,lat,point_id):

    forest_df = utc_time_df.copy()

    for each_file in forest_file_list:
        
        raster_file = f"{forest_temp}{each_file}"
        
        
        if os.path.isfile(raster_file):
            month_from_file = str(each_file.split('_')[2]).split('.')[0]

            T_value = os.popen(f"gdallocationinfo -valonly -wgs84 {raster_file} {long} {lat}").read().splitlines()[0]

            forest_df.loc[forest_df['month'] == months_num[month_from_file],name] = T_value
            forest_df['longitude'] = long
            forest_df['latitude'] = lat
            forest_df['pointID'] = point_id
    # remove month
    return forest_df.drop('month',axis=1)

def multi_merger_df(data_frames):
    df =reduce(lambda  left,right: pd.merge(left,
                                                  right,
                                                  on=['utctime','latitude','longitude','pointID'],
                                            how='inner'), data_frames)
    return df

# function to merge dfs
def merge_df(df1,df2):
    if df1.empty:
        df = pd.concat([df1,df2],axis=1)
        # df = df.T.drop_duplicates().T
    else:
        df = pd.merge(df1, df2, how='inner', on=['utctime','latitude','longitude','pointID'])
    return df


forest_temp_df = pd.DataFrame()

for index, row in eu_data.iterrows():

    lat = row["site_lat"]
    long = row["site_long"]
    point_id = row["pointID"]

    min_df = get_forest_values(forest_T_min_ls,'Forest_T_Min',long,lat,point_id)
    max_df = get_forest_values(forest_T_max_ls,'Forest_T_Max',long,lat,point_id)
    mean_df = get_forest_values(forest_T_mean_ls,'Forest_T_Mean',long,lat,point_id)
    
    frames_to_combine = [min_df,max_df,mean_df]

    point_df = multi_merger_df(data_frames=frames_to_combine)

    forest_temp_df = pd.concat([forest_temp_df,point_df])

forest_temp_df.to_csv("/home/ubuntu/data/ML/training-data/soiltemp/forest_temp_features_add.csv",index=False)



    

