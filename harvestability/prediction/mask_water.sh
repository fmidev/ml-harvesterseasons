#!/bin/bash


country=$1

file_path=/home/ubuntu/data/ml-harvestability/predictions/$country/$country-nowater.tif
wbm_new_resolution=/home/ubuntu/data/ml-harvestability/predictions/$country/$country-wbm-r30.tif
cropped_wbm=/home/ubuntu/data/ml-harvestability/predictions/$country/$country-wbm-cut.tif
final_output=/home/ubuntu/data/ml-harvestability/predictions/$country/$country-$(date '+%Y')-trfy-r30m.tif
pred_file=s3://copernicus/harvestability/$country-$(date '+%Y')-trfy-r30m.tif

eval "$(conda shell.bash hook)"
conda activate xgb


s3cmd get $pred_file $file_path

x=$(gdalinfo $file_path | grep -P "Pixel Size = " | cut -d' ' -f4 | cut -d',' -f1 | tr -d '(')
y=$(gdalinfo $file_path | grep -P "Pixel Size = " | cut -d' ' -f4 | cut -d',' -f2 | tr -d ')')

gdal_translate -tr $x $y -ot Byte /vsicurl/https://copernicus.data.lit.fmi.fi/dtm/wbm/$country-wbm.tif $wbm_new_resolution

db=/home/smartmet/data/gadm36_0.shp
layer=gadm36_0
cntr=$(echo $country | tr "_" " ")
# modified script of "Cut and crop TWI files to country border - 14.5.2021 Mikko Strahlendorff"
gdalwarp -cutline $db -cl $layer -cwhere "name_0 = '$cntr'" -crop_to_cutline -of COG -co COMPRESS=DEFLATE -co PREDICTOR=YES -co BIGTIFF=IF_SAFER $wbm_new_resolution $cropped_wbm

gdal_calc.py -A $file_path -B $cropped_wbm --NoDataValue=0 --type=Byte --outfile=$final_output --calc="numpy.where(B>0,B+6,A)"

python -u map_rgb.py -c $country

s3cmd put --multipart-chunk-size-mb=128 -P $final_output s3://copernicus/harvestability/${country}-$(date '+%Y')-trfy-r30m.tif

rm $file_path
rm $wbm_new_resolution
rm $cropped_wbm
rm $final_output