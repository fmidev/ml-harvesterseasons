#!/bin/bash
# Cut and crop TWI files to country border
# Usage: give arguments Country name and twi file(s). Quote multiple files. Run in directory of the TWI files.
# 14.5.2021 Mikko Strahlendorff


country=$1

pred_file=s3://copernicus/harvestability/$country-$(date '+%Y')-trfy-r30m.tif
log_dir=/home/ubuntu/data/ml-harvestability/logs/
input_file=/home/ubuntu/data/ml-harvestability/predictions/$country/$country-$(date '+%Y')-trfy-r30m.tif
out_file=/home/ubuntu/data/ml-harvestability/predictions/$country/$country-cut-$(date '+%Y')-trfy-r30m.tif



#modified script of "Cut and crop TWI files to country border - 14.5.2021 Mikko Strahlendorff"
cntr=$(echo $country | tr "_" " ")
#[ ! "$2" == "" ] && ifiles=$2 || ifiles=$file.tif
s3cmd get $pred_file $input_file
s3cmd mv $pred_file s3://copernicus/harvestability/predictions/$country/
echo "$input_file will be cut according to $cntr"
db=/home/smartmet/data/gadm36_0.shp
layer=gadm36_0
gdalwarp -cutline $db -cl $layer -cwhere "name_0 = '$cntr'" -crop_to_cutline -of COG -co COMPRESS=DEFLATE -co PREDICTOR=YES -co BIGTIFF=IF_SAFER $input_file $out_file
s3cmd -P -q put $out_file s3://copernicus/harvestability/$country-$(date '+%Y')-trfy-r30m.tif
rm $out_file
rm $input_file