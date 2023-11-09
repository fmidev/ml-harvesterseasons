from typing import List
import os, time, random, warnings,sys
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

location_csv = '/home/ubuntu/ml-harvesterseasons/soil_temperature/data/LUCAS_2018_Copernicus_attr+additions.csv'
location_csv = 'location_dir = "C:\\Users\\prakasam\\Documents\\era5-land\\soil_temp\\data\\soiltemp-stations-lonlat.csv"'
utc_csv = '/home/ubuntu/ml-harvesterseasons/soil_temperature/data/utctime_2015-2022.csv'
forest_temp_output = '/home/ubuntu/ml-harvesterseasons/soil_temperature/data/forest_temp'
forest_temp = '/home/ubuntu/data/soiltempdb/'


months_num = {'January':1,'February':2,'March':3,'April':4,'May':5,
              'June':6,'July':7,'August':8,'September':9,'October':10,
              'November':11,'December':12}
months_num = {'January':1,'February':2,'March':3,'April':4,'May':5,
              'June':6,'July':7,'August':8,'September':9,'October':10,
              'November':11,'December':12}

# location_df = pd.read_csv(location_csv,sep=",",usecols=["TH_LAT","TH_LONG","POINT_ID"])
station_loc = pd.read_csv(location_csv)
station_loc = station_loc.join(station_loc['xy'].str.split('_', n=1, expand=True).rename(columns={0:'long', 1:'lat'}))
station_loc = station_loc.drop("xy", axis='columns')
station_loc["long"] = station_loc["long"].astype(float)
station_loc["lat"] = station_loc["lat"].astype(float)
eu_station_loc= pd.DataFrame()

location_df = station_loc.loc[(station_loc["long"] >= -30.0) & (station_loc["long"] <= 50)]
location_df = station_loc.loc[(station_loc['lat'] >= 25) & (station_loc['lat'] <= 75)]

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

def list_to_csv(ls:List,filename:str,pointID:float,path:str):

    point_file = f"{path}{filename}_2015-2022_{pointID}_fix.csv"
    write_file = open(point_file,"w")
    for i in range(len(ls)):
            
        write_file.write(str(ls[i]))
        write_file.write("\n")


def get_forest_values(forest_file_list:List,name:str,long,lat):

    forest_df = utc_time_df.copy()

    for each_file in forest_file_list:
        
        raster_file = f"{forest_temp}{each_file}"
        
        
        if os.path.isfile(raster_file):
            month_from_file = str(each_file.split('_')[2]).split('.')[0]

            T_value = os.popen(f"gdallocationinfo -valonly -wgs84 {raster_file} {long} {lat}").read().splitlines()[0]

            forest_df.loc[forest_df['month'] == months_num[month_from_file],name] = T_value
    return forest_df

for index, row in location_df.iterrows():

    lat = row["lat"]
    long = row["long"]
    pointID = row["POINT_ID"]
    print(pointID)

    min_df = get_forest_values(forest_T_min_ls,'Forest_T_Min',long,lat)
    max_df = get_forest_values(forest_T_max_ls,'Forest_T_Max',long,lat)
    mean_df = get_forest_values(forest_T_mean_ls,'Forest_T_Mean',long,lat)
    
    min_to_lst = min_df["Forest_T_Min"]

    max_to_lst = max_df["Forest_T_Max"]

    mean_to_lst = mean_df["Forest_T_Mean"]

    list_to_csv(min_to_lst,"Forest_T_Min",pointID,forest_temp_output)
    list_to_csv(max_to_lst,"Forest_T_Max",pointID,forest_temp_output)
    list_to_csv(mean_to_lst,"Forest_T_Mean",pointID,forest_temp_output)
        

