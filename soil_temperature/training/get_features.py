import pandas as pd
import numpy as np
import os
from osgeo import osr, ogr, gdal
import warnings
from tqdm import tqdm
gdal.PushErrorHandler('CPLQuietErrorHandler')
gdal.UseExceptions()


#location_dir = "/home/ubuntu/data/soiltempdb/soiltemp-stations-lonlat.csv"
#stations_dir = "/home/ubuntu/data/soiltempdb/soiltemp-stations.csv"
utc_csv = "C:\\Users\\prakasam\\Documents\\era5-land\\soil_temp\\data\\utc_csv.csv"
location_dir = "C:\\Users\\prakasam\\Documents\\era5-land\\soil_temp\\data\\soiltemp-stations-lonlat.csv"
height_file = "/home/ubuntu/data/ml-harvestability/training/Europe-dtm.vrt"
tcd_file = "/home/ubuntu/data/ml-harvestability/training/eu.vrt"
waw_file = "/home/ubuntu/data/ml-harvestability/training/WAW_2018_010m_eu_03035_v020.vrt"
slope_file = "/home/ubuntu/data/ml-harvestability/training/Europe-slope.vrt"
aspect_file = "/home/ubuntu/data/ml-harvestability/training/Europe-aspect.vrt"
sand_0_5_file = "/home/ubuntu/data/soilgrids/sand/202005_sand_0-5cm_mean_250.tif"
sand_5_15_file = "/home/ubuntu/data/soilgrids/sand/202005_sand_5-15cm_mean_250.tif"
silt_0_5_file = "/home/ubuntu/data/soilgrids/silt/202005_silt_0-5cm_mean_250.tif"
silt_5_15_file = "/home/ubuntu/data/soilgrids/silt/202005_silt_5-15cm_mean_250.tif"
clay_0_5_file = "/home/ubuntu/data/soilgrids/clay/202005_clay_0-5cm_mean_250.tif"
clay_5_15_file = "/home/ubuntu/data/soilgrids/clay/202005_clay_5-15cm_mean_250.tif"
soc_0_5_file = "/home/ubuntu/data/soilgrids/soc/202005_soc_0-5cm_mean_250.tif"
soc_5_15_file = "/home/ubuntu/data/soilgrids/soc/202005_soc_5-15cm_mean_250.tif"
meanT_warmestQ_5_15cm = "/home/ubuntu/data/SBI-soil/SBIO2_0_5cm_mean_diurnal_range.tif"
meanT_warmestQ_0_5cm = "/home/ubuntu/data/SBI-soil/SBIO10_0_5cm_meanT_warmestQ.tif"
mean_diurnal_0_5cm = "/home/ubuntu/data/SBI-soil/SBIO10_5_15cm_meanT_warmestQ.tif"

soiltemp_file = f"/home/ubuntu/data/soiltempdb/soil_temp_features.csv"

feature_file_soiltemp = open(soiltemp_file,"w")

feature_file_soiltemp.write("long,lat,TCD,WAW,DTM_height,DTM_slope,DTM_aspect,sand_0-5cm_mean,sand_5-15cm_mean,silt_0-5cm_mean,silt_5-15cm_mean,clay_0-5cm_mean,clay_5-15cm_mean,soc_0-5cm_mean,soc_5-15cm_mean")
feature_file_soiltemp.write("\n")

feature_files = [
                #f"{twi_dir}{country_file}",
                tcd_file,
                waw_file,
                height_file,
                slope_file,
                aspect_file,
                sand_0_5_file,
                sand_5_15_file,
                silt_0_5_file,
                silt_5_15_file,
                clay_0_5_file,
                clay_5_15_file,
                soc_0_5_file,
                soc_5_15_file,
                meanT_warmestQ_0_5cm,
                meanT_warmestQ_5_15cm,
                mean_diurnal_0_5cm
                ]

def get_feature_values(long_lat):

    results=[]
    nan=[]

    for i in  tqdm(range(0,len(long_lat)),desc ="Processing locations"):
        results_for_tcd = os.popen(f"gdallocationinfo -valonly -wgs84 {feature_files[0]} {long_lat[i][0]} {long_lat[i][1]}").read()
        results_for_waw = os.popen(f"gdallocationinfo -valonly -wgs84 {feature_files[1]} {long_lat[i][0]} {long_lat[i][1]}").read()
        results_for_height = os.popen(f"gdallocationinfo -valonly -wgs84 {feature_files[2]} {long_lat[i][0]} {long_lat[i][1]}").read()
        results_for_slope = os.popen(f"gdallocationinfo -valonly -wgs84 {feature_files[3]} {long_lat[i][0]} {long_lat[i][1]}").read()
        results_for_aspect = os.popen(f"gdallocationinfo -valonly -wgs84 {feature_files[4]} {long_lat[i][0]} {long_lat[i][1]}").read()
        ##
        results_sand_0_5_file = os.popen(f"gdallocationinfo -valonly -wgs84 {feature_files[5]} {long_lat[i][0]} {long_lat[i][1]}").read()
        results_sand_5_15_file= os.popen(f"gdallocationinfo -valonly -wgs84 {feature_files[6]} {long_lat[i][0]} {long_lat[i][1]}").read()
        results_silt_0_5_file = os.popen(f"gdallocationinfo -valonly -wgs84 {feature_files[7]} {long_lat[i][0]} {long_lat[i][1]}").read()
        results_silt_5_15_file = os.popen(f"gdallocationinfo -valonly -wgs84 {feature_files[8]} {long_lat[i][0]} {long_lat[i][1]}").read()
        results_clay_0_5_file = os.popen(f"gdallocationinfo -valonly -wgs84 {feature_files[9]} {long_lat[i][0]} {long_lat[i][1]}").read()
        results_clay_5_15_file = os.popen(f"gdallocationinfo -valonly -wgs84 {feature_files[10]} {long_lat[i][0]} {long_lat[i][1]}").read()
        results_soc_0_5_file = os.popen(f"gdallocationinfo -valonly -wgs84 {feature_files[11]} {long_lat[i][0]} {long_lat[i][1]}").read()
        results_soc_5_15_file = os.popen(f"gdallocationinfo -valonly -wgs84 {feature_files[12]} {long_lat[i][0]} {long_lat[i][1]}").read()
        results_meanT_warmestQ_0_5cm = os.popen(f"gdallocationinfo -valonly -wgs84 {feature_files[13]} {long_lat[i][0]} {long_lat[i][1]}").read()
        results_meanT_warmestQ_5_15cm = os.popen(f"gdallocationinfo -valonly -wgs84 {feature_files[14]} {long_lat[i][0]} {long_lat[i][1]}").read()
        results_mean_diurnal_0_5cm = os.popen(f"gdallocationinfo -valonly -wgs84 {feature_files[15]} {long_lat[i][0]} {long_lat[i][1]}").read()

        if results_for_tcd and results_for_waw and results_for_height and results_for_slope and results_for_aspect and results_sand_0_5_file and results_sand_5_15_file and results_silt_0_5_file and results_silt_5_15_file and results_clay_0_5_file and results_clay_5_15_file and results_soc_0_5_file and results_soc_5_15_file and results_meanT_warmestQ_0_5cm and results_meanT_warmestQ_5_15cm and results_mean_diurnal_0_5cm:
            result_for_tcd = int(results_for_tcd)
            result_for_waw = int(results_for_waw)
            result_for_height = float(results_for_height)
            result_for_slope = float(results_for_slope)
            result_for_aspect = float(results_for_aspect)
            result_for_sand_0_5 = int(results_sand_0_5_file)
            result_for_sand_5_15 = int(results_sand_5_15_file)
            result_for_silt_0_5 = int(results_silt_0_5_file)
            result_for_silt_5_15 = int(results_silt_5_15_file)
            result_for_clay_0_5 = int(results_clay_0_5_file)
            result_for_clay_5_15 = int(results_clay_5_15_file)
            result_for_soc_0_5 = int(results_soc_0_5_file)
            result_for_soc_5_15 = int(results_soc_5_15_file)
            results_meanT_warmestQ_0_5cm = float(results_meanT_warmestQ_0_5cm)
            results_meanT_warmestQ_5_15cm = float(results_meanT_warmestQ_5_15cm)
            results_mean_diurnal_0_5cm = float(results_mean_diurnal_0_5cm)

            results.append([long_lat[i][0],
                            long_lat[i][1],
                            result_for_tcd,
                            result_for_waw,
                            result_for_height,
                            result_for_slope,
                            result_for_aspect,
                            result_for_sand_0_5,
                            result_for_sand_5_15,
                            result_for_silt_0_5,
                            result_for_silt_5_15,
                            result_for_clay_0_5,
                            result_for_clay_5_15,
                            result_for_soc_0_5,
                            result_for_soc_5_15,
                            results_meanT_warmestQ_0_5cm,
                            results_meanT_warmestQ_5_15cm,
                            results_mean_diurnal_0_5cm])
        else:
            nan.append([long_lat[i][0],long_lat[i][1],0])
    return results,nan

station_loc = pd.read_csv(location_dir)
# station_loc["x","y"] = station_loc['xy'].str.split('_', n=1, expand=True)
station_loc = station_loc.join(station_loc['xy'].str.split('_', n=1, expand=True).rename(columns={0:'long', 1:'lat'}))
station_loc = station_loc.drop("xy", axis='columns')
station_loc["long"] = station_loc["long"].astype(float)
station_loc["lat"] = station_loc["lat"].astype(float)
eu_station_loc= pd.DataFrame()

eu_station_loc = station_loc.loc[(station_loc["long"] >= -30.0) & (station_loc["long"] <= 50)]
eu_station_loc = station_loc.loc[(station_loc['lat'] >= 25) & (station_loc['lat'] <= 75)]
print(eu_station_loc.head(-50))

long_ls = eu_station_loc['long'].tolist()
lat_ls = eu_station_loc['lat'].tolist()

long_lat = []

def traverse_latlong(TWI):
    for i in range(0,len(eu_station_loc)):

        long_lat.append([long_ls[i],lat_ls[i]])
        
    
    feature_list,nan = get_feature_values(long_lat)

    non_nan_df = pd.DataFrame(feature_list,columns=['long','lat','TCD','WAW','height','slope',
                                               'aspect','sand_0_5','sand_5_15',
                                               'silt_0_5','silt_5_15','clay_0_5','clay_5_15',
                                               'soc_0_5','soc_5_15','meanT_warmestQ_5_15cm','meanT_warmestQ_0_5cm','mean_diurnal_0_5cm'])
    nan_df = pd.DataFrame(nan,columns=['long','lat','TCD','WAW','height','slope',
                                               'aspect','sand_0_5','sand_5_15',
                                               'silt_0_5','silt_5_15','clay_0_5','clay_5_15',
                                               'soc_0_5','soc_5_15','meanT_warmestQ_5_15cm','meanT_warmestQ_0_5cm','mean_diurnal_0_5cm'])
    feature_df = pd.concat([non_nan_df,nan_df])

    write_to_csv(feature_df)

def write_to_csv(df_to_write):
    long_values = df_to_write["long"].values.tolist()
    lat_values = df_to_write["lat"].values.tolist()
    TCD = df_to_write["TCD"].values.tolist()
    WAW = df_to_write["WAW"].values.tolist()
    height = df_to_write["height"].values.tolist()
    slope = df_to_write["slope"].values.tolist()
    aspect = df_to_write["aspect"].values.tolist()
    sand_0_5 = df_to_write["sand_0_5"].values.tolist()
    sand_5_15 = df_to_write["sand_5_15"].values.tolist()
    silt_0_5 = df_to_write["silt_0_5"].values.tolist()
    silt_5_15 = df_to_write["silt_5_15"].values.tolist()
    clay_0_5 = df_to_write["clay_0_5"].values.tolist()
    clay_5_15 = df_to_write["clay_5_15"].values.tolist()
    soc_0_5 = df_to_write["soc_0_5"].values.tolist()
    soc_5_15 = df_to_write["soc_5_15"].values.tolist()
    meanT_warmestQ_5_15cm = df_to_write["meanT_warmestQ_5_15cm"].values.tolist()
    meanT_warmestQ_0_5cm = df_to_write["meanT_warmestQ_0_5cm"].values.tolist()
    mean_diurnal_0_5cm = df_to_write["mean_diurnal_0_5cm"].values.tolist()

    for i in range(len(long_values)):
        
        feature_file_soiltemp.write(str(long_values[i])+','+str(lat_values[i])+','+
                              str(TCD[i])+','+str(WAW[i])+','+str(height[i])+
                              ','+str(slope[i])+','+str(aspect[i])+','+str(sand_0_5[i])+
                              ','+str(sand_5_15[i])+','+str(silt_0_5[i])+','+str(silt_5_15[i])+
                              ','+str(clay_0_5[i])+','+str(clay_5_15[i])+','+str(soc_0_5[i])+
                              ','+str(soc_5_15[i])+','+str(meanT_warmestQ_5_15cm[i])+','+str(meanT_warmestQ_0_5cm[i])+
                              ','+str(mean_diurnal_0_5cm[i]))
        feature_file_soiltemp.write("\n")

###################### Copying/generating values for each month and date and year for all features #############################

station_features = pd.read_csv(soiltemp_file,usecols=['long','lat','TCD','WAW','height','slope',
                                               'aspect','sand_0_5','sand_5_15',
                                               'silt_0_5','silt_5_15','clay_0_5','clay_5_15',
                                               'soc_0_5','soc_5_15',
                                               'meanT_warmestQ_5_15cm','meanT_warmestQ_0_5cm','mean_diurnal_0_5cm'])

months_num = {'January':1,'February':2,'March':3,'April':4,'May':5,
              'June':6,'July':7,'August':8,'September':9,'October':10,
              'November':11,'December':12}

utc_time_df = pd.read_csv(utc_csv,usecols=["utctime"])

utc_time_df['month'] = utc_time_df['utctime'].astype(str).str.split('-').str[1].astype(int)



def get_time_values(station_features):

    
    appended_df =pd.DataFrame()
    # for each_file in station_features:
    for each_index in range(len(station_features)):
        time_df = utc_time_df.copy()
        time_df[['long','lat','TCD','WAW','height','slope',
                                               'aspect','sand_0_5','sand_5_15',
                                               'silt_0_5','silt_5_15','clay_0_5','clay_5_15',
                                               'soc_0_5','soc_5_15',
                                               'meanT_warmestQ_5_15cm','meanT_warmestQ_0_5cm','mean_diurnal_0_5cm']] = " "
        
        for i in range(len(time_df)):
            time_df.loc[i, "long"] = station_features.loc[i, "long"]
            time_df.loc[i, "lat"] = station_features.loc[i, "lat"]
            time_df.loc[i, "TCD"] = station_features.loc[i, "TCD"]
            time_df.loc[i, "WAW"] = station_features.loc[i, "WAW"]
            time_df.loc[i, "height"] = station_features.loc[i, "height"]
            time_df.loc[i, "slope"] = station_features.loc[i, "slope"]
            time_df.loc[i, "aspect"] = station_features.loc[i, "aspect"]
            time_df.loc[i, "sand_0_5"] = station_features.loc[i, "sand_0_5"]
            time_df.loc[i, "sand_5_15"] = station_features.loc[i, "sand_5_15"]
            time_df.loc[i, "silt_0_5"] = station_features.loc[i, "silt_0_5"]
            time_df.loc[i, "silt_5_15"] = station_features.loc[i, "silt_5_15"]
            time_df.loc[i, "clay_0_5"] = station_features.loc[i, "clay_0_5"]
            time_df.loc[i, "clay_5_15"] = station_features.loc[i, "clay_5_15"]
            time_df.loc[i, "soc_0_5"] = station_features.loc[i, "soc_0_5"]
            time_df.loc[i, "soc_5_15"] = station_features.loc[i, "soc_5_15"]
            time_df.loc[i, "meanT_warmestQ_5_15cm"] = station_features.loc[i, "meanT_warmestQ_5_15cm"]
            time_df.loc[i, "meanT_warmestQ_0_5cm"] = station_features.loc[i, "meanT_warmestQ_0_5cm"]
            time_df.loc[i, "mean_diurnal_0_5cm"] = station_features.loc[i, "mean_diurnal_0_5cm"]
           
        appended_df= pd.concat([appended_df,time_df])