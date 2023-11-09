#!/bin/bash

twi_file="/vsicurl/https://copernicus.data.lit.fmi.fi/dtm/twi/Finland-twi.tif"
tcd_file="/home/ubuntu/data/ml-harvestability/training/eu.vrt"
waw_file="/home/ubuntu/data/ml-harvestability/training/WAW_2018_010m_eu_03035_v020.vrt"
height_file="/home/ubuntu/data/ml-harvestability/training/Europe-dtm.vrt"
slope_file="/home/ubuntu/data/ml-harvestability/training/Europe-slope.vrt"
aspect_file="/home/ubuntu/data/ml-harvestability/training/Europe-aspect.vrt"
harvestability_file="/home/ubuntu/data/ml-harvestability/training/soildata/harvestability.tif"

cat train_locations.txt | gdallocationinfo -wgs84 -valonly $harvestability_file > train_harvestability.txt
cat train_locations.txt | gdallocationinfo -wgs84 -valonly $tcd_file > train_TCD.txt
cat train_locations.txt | gdallocationinfo -wgs84 -valonly $waw_file > train_WAW.txt
cat train_locations.txt | gdallocationinfo -wgs84 -valonly $height_file > train_Height.txt
cat train_locations.txt | gdallocationinfo -wgs84 -valonly $slope_file > train_Slope.txt
cat train_locations.txt | gdallocationinfo -wgs84 -valonly $aspect_file > train_Aspect.txt
cat train_locations.txt | gdallocationinfo -wgs84 -valonly $twi_file > train_TWI.txt
cat train_locations.txt | gdallocationinfo -wgs84 -valonly /home/ubuntu/data/soilgrids/sand/202005_sand_0-5cm_mean_250.tif > train-locs-202005_sand_0-5cm_mean_250.txt
cat train_locations.txt | gdallocationinfo -wgs84 -valonly /home/ubuntu/data/soilgrids/sand/202005_sand_5-15cm_mean_250.tif > train-locs-202005_sand_5-15cm_mean_250.txt

cat train_locations.txt | gdallocationinfo -wgs84 -valonly /home/ubuntu/data/soilgrids/silt/202005_silt_0-5cm_mean_250.tif > train-locs-202005_silt_0-5cm_mean_250.txt
cat train_locations.txt | gdallocationinfo -wgs84 -valonly /home/ubuntu/data/soilgrids/silt/202005_silt_5-15cm_mean_250.tif > train-locs-202005_silt_5-15cm_mean_250.txt

cat train_locations.txt | gdallocationinfo -wgs84 -valonly /home/ubuntu/data/soilgrids/clay/202005_clay_0-5cm_mean_250.tif > train-locs-202005_clay_0-5cm_mean_250.txt
cat train_locations.txt | gdallocationinfo -wgs84 -valonly /home/ubuntu/data/soilgrids/clay/202005_clay_5-15cm_mean_250.tif > train-locs-202005_clay_5-15cm_mean_250.txt

cat train_locations.txt | gdallocationinfo -wgs84 -valonly /home/ubuntu/data/soilgrids/soc/202005_soc_0-5cm_mean_250.tif > train-locs-202005_soc_0-5cm_mean_250.txt
cat train_locations.txt | gdallocationinfo -wgs84 -valonly /home/ubuntu/data/soilgrids/soc/202005_soc_5-15cm_mean_250.tif > train-locs-202005_soc_5-15cm_mean_250.txt

paste -d, train-locs-202005_sand_0-5cm_mean_250.txt train-locs-202005_sand_5-15cm_mean_250.txt train-locs-202005_silt_0-5cm_mean_250.txt train-locs-202005_silt_5-15cm_mean_250.txt train-locs-202005_clay_0-5cm_mean_250.txt train-locs-202005_clay_5-15cm_mean_250.txt train-locs-202005_soc_0-5cm_mean_250.txt train-locs-202005_soc_5-15cm_mean_250.txt train_locations.txt train_Height.txt train_Slope.txt train_Aspect.txt train_TCD.txt train_WAW.txt train_harvestability.txt train_TWI.txt  > train_data.csv
