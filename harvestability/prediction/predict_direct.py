import pandas as pd
import numpy as np
np.random.seed(42)
import xgboost as xgb
from datetime import datetime
from osgeo import osr, ogr, gdal
import fiona
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt
import os
import argparse     
import logging
from tqdm import tqdm
import warnings
gdal.PushErrorHandler('CPLQuietErrorHandler')
gdal.UseExceptions()

parser = argparse.ArgumentParser(description='Predicting harvestability',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c','--country_name',type=str,help='country name should be provided',required=True)
args = parser.parse_args()

# if args.country_name == "Europe":
#     country_file  = f"{args.country_name}-twi.tif"
# else:
#     country_file  = f"{args.country_name}-twi-cut.tif"

country_file  = f"{args.country_name}-twi.tif"

model_dir = "/home/ubuntu/data/ml-harvestability/models/"
binary_model = "xgbmodel_harvestability_binary_07102023083740.json"
peatland_model = "xgbmodel_harvestability_peatland_07102023104858.json"
mineral_model = "xgbmodel_harvestability_mineral_07102023113008.json"
#pred_save_dir = "../predictions/"
pred_file = f"/home/ubuntu/data/ml-harvestability/predictions/{args.country_name}/{args.country_name}.csv"
vrt_file = f"/home/ubuntu/data/ml-harvestability/predictions/{args.country_name}/{args.country_name}.vrt"
tif_file = f"/home/ubuntu/data/ml-harvestability/predictions/{args.country_name}/{args.country_name}.tif"
temp_file = f"/home/ubuntu/data/ml-harvestability/predictions/{args.country_name}/locations.txt"

twi_dir="/home/ubuntu/data/twi-dtm/"
prediction_dir="/home/ubuntu/data/ml-harvestability/predictions/"
twi_file = f"/vsicurl/https://copernicus.data.lit.fmi.fi/dtm/twi/{country_file}"
tcd_file="/home/ubuntu/data/ml-harvestability/training/eu.vrt"
waw_file="/home/ubuntu/data/ml-harvestability/training/WAW_2018_010m_eu_03035_v020.vrt"
height_file="/home/ubuntu/data/ml-harvestability/training/Europe-dtm.vrt"
slope_file="/home/ubuntu/data/ml-harvestability/training/Europe-slope.vrt"
aspect_file="/home/ubuntu/data/ml-harvestability/training/Europe-aspect.vrt"
sand_0_5_file="/home/ubuntu/data/soilgrids/sand/202005_sand_0-5cm_mean_250.tif"
sand_5_15_file="/home/ubuntu/data/soilgrids/sand/202005_sand_5-15cm_mean_250.tif"
silt_0_5_file="/home/ubuntu/data/soilgrids/silt/202005_silt_0-5cm_mean_250.tif"
silt_5_15_file="/home/ubuntu/data/soilgrids/silt/202005_silt_5-15cm_mean_250.tif"
clay_0_5_file="/home/ubuntu/data/soilgrids/clay/202005_clay_0-5cm_mean_250.tif"
clay_5_15_file="/home/ubuntu/data/soilgrids/clay/202005_clay_5-15cm_mean_250.tif"
soc_0_5_file="/home/ubuntu/data/soilgrids/soc/202005_soc_0-5cm_mean_250.tif"
soc_5_15_file="/home/ubuntu/data/soilgrids/soc/202005_soc_5-15cm_mean_250.tif"

# features files
columns = ["long","lat","TWI","TCD","WAW","DTM_height","DTM_slope","DTM_aspect",
           "sand_0-5cm_mean","sand_5-15cm_mean","silt_0-5cm_mean","silt_5-15cm_mean","clay_0-5cm_mean",
           "clay_5-15cm_mean","soc_0-5cm_mean","soc_5-15cm_mean"]

prediction_file = open(pred_file,"w")

prediction_file.write("long,lat,harvestability")
prediction_file.write("\n")


# load ml model from directory and predict

binary_harvest_model = xgb.XGBClassifier()
binary_harvest_model.load_model(f"{model_dir}{binary_model}")

mineral_harvest_model = xgb.XGBClassifier()
mineral_harvest_model.load_model(f"{model_dir}{mineral_model}")

peatland_harvest_model = xgb.XGBClassifier()
peatland_harvest_model.load_model(f"{model_dir}{peatland_model}")

feature_files = [
                #f"{twi_dir}{country_file}",
                twi_file,
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
                soc_5_15_file
                ]

def get_feature_values(locations_file,long_lat):
    results=[]
    nan=[]
    results_for_twis = os.popen(f"cat {locations_file} | gdallocationinfo -valonly -wgs84 {feature_files[0]}").read().splitlines()
    results_for_tcds = os.popen(f"cat {locations_file} | gdallocationinfo -valonly -wgs84 {feature_files[1]}").read().splitlines()
    results_for_waws = os.popen(f"cat {locations_file} | gdallocationinfo -valonly -wgs84 {feature_files[2]}").read().splitlines()
    results_for_heights = os.popen(f"cat {locations_file} | gdallocationinfo -valonly -wgs84 {feature_files[3]}").read().splitlines()
    results_for_slopes = os.popen(f"cat {locations_file} | gdallocationinfo -valonly -wgs84 {feature_files[4]}").read().splitlines()
    results_for_aspects = os.popen(f"cat {locations_file} | gdallocationinfo -valonly -wgs84 {feature_files[5]}").read().splitlines()
    ##
    results_sand_0_5_files = os.popen(f"cat {locations_file} | gdallocationinfo -valonly -wgs84 {feature_files[6]}").read().splitlines()
    results_sand_5_15_files= os.popen(f"cat {locations_file} | gdallocationinfo -valonly -wgs84 {feature_files[7]}").read().splitlines()
    results_silt_0_5_files = os.popen(f"cat {locations_file} | gdallocationinfo -valonly -wgs84 {feature_files[8]}").read().splitlines()
    results_silt_5_15_files = os.popen(f"cat {locations_file} | gdallocationinfo -valonly -wgs84 {feature_files[9]}").read().splitlines()
    results_clay_0_5_files = os.popen(f"cat {locations_file} | gdallocationinfo -valonly -wgs84 {feature_files[10]}").read().splitlines()
    results_clay_5_15_files = os.popen(f"cat {locations_file} | gdallocationinfo -valonly -wgs84 {feature_files[11]}").read().splitlines()
    results_soc_0_5_files = os.popen(f"cat {locations_file} | gdallocationinfo -valonly -wgs84 {feature_files[12]}").read().splitlines()
    results_soc_5_15_files = os.popen(f"cat {locations_file} | gdallocationinfo -valonly -wgs84 {feature_files[13]}").read().splitlines()

    for i in range(len(results_for_twis)):
        results_for_twi = results_for_twis[i]
        results_for_tcd = results_for_tcds[i]
        results_for_waw = results_for_waws[i]
        results_for_height = results_for_heights[i]
        results_for_slope = results_for_slopes[i]
        results_for_aspect = results_for_aspects[i]
        results_sand_0_5_file = results_sand_0_5_files[i]
        results_sand_5_15_file = results_sand_5_15_files[i]
        results_silt_0_5_file = results_silt_0_5_files[i]
        results_silt_5_15_file = results_silt_5_15_files[i]
        results_clay_0_5_file = results_clay_0_5_files[i]
        results_clay_5_15_file = results_clay_5_15_files[i]
        results_soc_0_5_file = results_soc_0_5_files[i]
        results_soc_5_15_file = results_soc_5_15_files[i]

        if results_for_twi and results_for_tcd and results_for_waw and results_for_height and results_for_slope and results_for_aspect and results_sand_0_5_file and results_sand_5_15_file and results_silt_0_5_file and results_silt_5_15_file and results_clay_0_5_file and results_clay_5_15_file and results_soc_0_5_file and results_soc_5_15_file:
            result_for_twi = float(results_for_twi)
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

            results.append([long_lat[i][0],
                            long_lat[i][1],
                            result_for_twi,
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
                            result_for_soc_5_15])
        else:
            nan.append([long_lat[i][0],long_lat[i][1],0])
    return results,nan
                    
def predict(results):
    result_df = pd.DataFrame(results, columns=columns)
    result_df = result_df[['long','lat','sand_0-5cm_mean', 'sand_5-15cm_mean', 'silt_0-5cm_mean', 'silt_5-15cm_mean', 'clay_0-5cm_mean', 
                            'clay_5-15cm_mean', 'soc_0-5cm_mean', 'soc_5-15cm_mean', 'DTM_height', 
                            'DTM_slope', 'DTM_aspect', 'TCD', 'WAW', 'TWI']]
    y_predict = binary_harvest_model.predict(result_df.drop(["long","lat"],axis=1))
    result_df["binary_result"] = y_predict
    mineral_df = result_df[result_df["binary_result"]== 0]
    peatland_df = result_df[result_df["binary_result"]== 1]

    if len(mineral_df)>0:
        mineral_df = mineral_df.drop("binary_result",axis=1)
        mineral_pred = mineral_harvest_model.predict(mineral_df.drop(["long","lat"],axis=1))
        mineral_pred=mineral_pred+1
        mineral_df['harvestability']=mineral_pred

    if len(peatland_df)>0:
        peatland_df = peatland_df.drop("binary_result",axis=1)
        peatland_pred = peatland_harvest_model.predict(peatland_df.drop(["long","lat"],axis=1))
        peatland_df['harvestability']=peatland_pred
        peatland_df['harvestability'] = peatland_df['harvestability'].map({0: 4, 1: 5,2: 6})

    predictions_df = pd.concat([mineral_df,peatland_df])

    return predictions_df
    #predicted results are +1 as in training we converted from 0
    # y_predict = predictions_df["harvestability"].values.tolist()
    # long_values = predictions_df["long"].values.tolist()
    # lat_values = predictions_df["lat"].values.tolist()

def write_to_csv(harvestability_df):
    long_values = harvestability_df["long"].values.tolist()
    lat_values = harvestability_df["lat"].values.tolist()
    y_predict = harvestability_df["harvestability"].values.tolist()
    for i in range(len(y_predict)):
        
        prediction_file.write(str(long_values[i])+','+str(lat_values[i])+','+str(y_predict[i]))
        prediction_file.write("\n")


def write_to_locations(long_lat):
    location_file = open(temp_file,"w")
    for i in range(len(long_lat)):
            
        location_file.write(str(long_lat[i][0])+' '+str(long_lat[i][1]))
        location_file.write("\n")


# loop through twi pixels and get longitude and lattitude

def traverse_twi(TWI):
    long_lat = []
    feature_list = []
    xoff, a, b, yoff, d, e = TWI.GetGeoTransform() # GDAL affine transform parameters, According to gdal documentation xoff/yoff are image left corner, a/e are pixel wight/height and b/d is rotation and is zero if image is north up. 

    def pixel_to_coord(x, y):
        """Returns global coordinates from pixel x, y coords"""
        xp = a * x + b * y + a * 0.5 + b * 0.5 + xoff
        yp = d * x + e * y + d * 0.5 + e * 0.5 + yoff
        return(xp, yp)
   


    # get columns and rows of your image from gdalinfo
    cols = TWI.RasterXSize
    rows = TWI.RasterYSize
    i = 0
    for row in  tqdm(range(0,rows),desc ="Processing locations"):
        for col in  range(0,cols): 
            long, lat = pixel_to_coord(col,row)
            if i != 10**7:
                i += 1
                long_lat.append([long,lat])
            else:
                # write long lat to temp file
                write_to_locations(long_lat)
                feature_list,nan = get_feature_values(temp_file,long_lat)
                prediction = predict(feature_list)
                nan_df = pd.DataFrame(nan,columns=['long','lat','harvestability'])
                harvestability_df = pd.concat([prediction,nan_df])
                #harvestability_df = combine_df.sort_values(by=['lat','long'])
                write_to_csv(harvestability_df)
                i = 0
                long_lat = []
        
    # write remaining
    write_to_locations(long_lat)
    feature_list,nan = get_feature_values(temp_file,long_lat)
    prediction = predict(feature_list)
    nan_df = pd.DataFrame(nan,columns=['long','lat','harvestability'])
    harvestability_df = pd.concat([prediction,nan_df])
    #harvestability_df = combine_df.sort_values(by=['lat','long'])
    write_to_csv(harvestability_df)

# start and end timestamps
start_ts = datetime.now().timestamp()

#read twi
#twi = gdal.Open(f"{twi_dir}{country_file}")

twi = gdal.Open(twi_file)

traverse_twi(twi)


print(f"Saving prediction to {pred_file}")
# use gdallocationinfo or use the method in notebook


end_ts = datetime.now().timestamp()

# convert timestamps to datetime object
dt1 = datetime.fromtimestamp(start_ts)
print('Datetime Start for prediction:', dt1)
dt2 = datetime.fromtimestamp(end_ts)
print('Datetime End for prediction:', dt2)
# Difference between two timestamps
# in hours:minutes:seconds format
delta = dt2 - dt1
print('Total execution time for prediction is:', delta)

# save as tif
if os.path.exists(vrt_file):
    os.remove(vrt_file)

f = open(vrt_file, "w")
f.write(f"<OGRVRTDataSource>\n \
    <OGRVRTLayer name=\"{args.country_name}\">\n \
        <SrcDataSource>{pred_file}</SrcDataSource>\n \
        <GeometryType>wkbPoint</GeometryType>\n \
        <GeometryField encoding=\"PointFromColumns\" x=\"long\" y=\"lat\" z=\"harvestability\"/>\n \
    </OGRVRTLayer>\n \
</OGRVRTDataSource>")

f.close()

twi_country = gdal.Info(twi_file,format= "json")

RasterXSize=twi_country["size"][0]
RasterYSize=twi_country["size"][1]

minx = twi_country["geoTransform"][0]
maxy = twi_country["geoTransform"][3]
maxx = minx + twi_country["geoTransform"][1] * RasterXSize
miny = maxy + twi_country["geoTransform"][5] * RasterYSize

r = gdal.Rasterize(tif_file,vrt_file, outputSRS="EPSG:4326", width=RasterXSize, height=RasterYSize, attribute="harvestability",layers=[args.country_name],noData=0,outputType=gdal.GDT_Byte)
r = None


ds = gdal.Open(tif_file, 1)
band = ds.GetRasterBand(1)

# create color table
colors = gdal.ColorTable()

# set color for each value
colors.SetColorEntry(0, (128, 128, 128))
colors.SetColorEntry(1, (0, 97, 0))
colors.SetColorEntry(2, (97, 153, 0))
colors.SetColorEntry(3, (160, 219, 0))
colors.SetColorEntry(4, (255, 250, 0))
colors.SetColorEntry(5, (255, 132, 0))
colors.SetColorEntry(6, (255, 38, 0))

# set color table and color interpretation
band.SetRasterColorTable(colors)
band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

# close and save file
del band, ds

#os.remove(vrt_file)

raster_end_ts = datetime.now().timestamp()

# convert timestamps to datetime object
dt1 = datetime.fromtimestamp(end_ts)
print('Datetime Start for rasterization:', dt1)
dt2 = datetime.fromtimestamp(raster_end_ts)
print('Datetime End for rasterization:', dt2)
# Difference between two timestamps
# in hours:minutes:seconds format
delta = dt2 - dt1
print('Total execution time for rasterization is:', delta)