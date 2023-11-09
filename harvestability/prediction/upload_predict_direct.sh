#!/bin/bash


country=$1

log_dir=/home/ubuntu/data/ml-harvestability/logs/
predict=/home/ubuntu/ml-harvesterseasons/harvestability/prediction/predict_direct.py
csv_file=/home/ubuntu/data/ml-harvestability/predictions/${country}/${country}.csv

tif_file="/home/ubuntu/data/ml-harvestability/predictions/$country/$country.tif"
vrt_file=/home/ubuntu/data/ml-harvestability/predictions/${country}/${country}.vrt
temp_locations_file=/home/ubuntu/data/ml-harvestability/predictions/${country}/locations.txt

mkdir -p /home/ubuntu/data/ml-harvestability/predictions/${country}

${country}
eval "$(conda shell.bash hook)"
conda activate xgb
python -u $predict -c $country >> $log_dir$country-predictions.log 

if [[ -f $tif_file ]]; then
    s3cmd put --multipart-chunk-size-mb=128 -P $tif_file s3://copernicus/harvestability/${country}-$(date '+%Y')-trfy-r30m.tif
    gzip $csv_file
    s3cmd put --multipart-chunk-size-mb=128 -P $csv_file.gz s3://copernicus/harvestability/predictions/${country}/${country}.csv.gz
    rm -f $csv_file.gz 
    rm -f $tif_file
    rm -f $vrt_file
    rm -f $temp_locations_file

fi
