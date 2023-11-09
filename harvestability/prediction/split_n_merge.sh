#!/bin/bash


countries=/home/ubuntu/data/ml-harvestability/scripts/split_n_merge_countries.txt
log_dir=/home/ubuntu/data/ml-harvestability/logs/

eval "$(conda shell.bash hook)"
conda activate xgb

# download each file from a text file
while IFS=$'\n' read country
do
    echo "Splitting ... $country"
    
    country_path=/home/ubuntu/data/ml-harvestability/predictions/${country}/

    # s3cmd csv.gz to respective directory
    s3cmd get s3://copernicus/harvestability/predictions/${country}/${country}.csv.gz $country_path

    # gzip -d csv.gz
    gzip -d $country_path${country}.csv.gz
    
    # using split command to split the csv.. make the chunk with 10GB files
    
    file_nm=${country}.csv
    file_names=${country}_split_
    HEADER=$(head -1 $country_path$file_nm)
    sed -i -e 1d $country_path$file_nm
    split $country_path$file_nm --line-bytes=100KB $country_path$file_names
    rm  $country_path$file_nm
    for i in ${country_path}$file_names*;
    do
        sed -i -e "1i$HEADER" "$i"
    done
    
    # go over a forloop for splitted files ,create vrt file for each of them( use a string replace with file name accordingly )
   
    for each_split in ${country_path}$file_names*
    do
        fname=$(echo $each_split | cut -d "/" -f8)
        mv $each_split $each_split.csv
        vrt_file=/home/ubuntu/data/ml-harvestability/predictions/${country}/$fname.vrt
        if [[ -f $vrt_file ]]; then
            rm $vrt_file
        fi
        echo "<OGRVRTDataSource>" >> $vrt_file
        echo "    <OGRVRTLayer name=\"$fname\">" >> $vrt_file
        echo "        <SrcDataSource>$each_split.csv</SrcDataSource>" >> $vrt_file
        echo "        <GeometryType>wkbPoint</GeometryType>" >> $vrt_file
        echo "        <GeometryField encoding=\"PointFromColumns\" x=\"long\" y=\"lat\" z=\"harvestability\"/>" >> $vrt_file
        echo "    </OGRVRTLayer>" >> $vrt_file
        echo "</OGRVRTDataSource>" >> $vrt_file

        gdal_rasterize -a harvestability -l $fname -a_nodata 0 -tr 0.000277777800000 -0.000277777800000 -a_srs EPSG:4326 -ot Byte $vrt_file $country_path$fname.tif
        echo $country_path$fname.tif >> ${country_path}all_splits.txt
        rm $vrt_file
        rm $each_split.csv
    done
    tif_file=$country_path$country-$(date '+%Y')-trfy-r30m.tif
    gdal_merge.py -a_nodata 0 -o $tif_file --optfile ${country_path}all_splits.txt
    python -u map_rgb.py -c $country
    if [[ -f $tif_file ]]; then
        s3cmd put --multipart-chunk-size-mb=128 -P $tif_file s3://copernicus/harvestability/${country}-$(date '+%Y')-trfy-r30m.tif
        rm -f $tif_file
        rm -f ${country_path}*.tif
    fi

    rm ${country_path}all_splits.txt

done < $countries




# go over a forloop for splitted files ,create vrt file for each of them( use a string replace with file name accordingly )
# and run gdal_rasterize command 

# get every tif files to a text file

# use gdal merge to combine all tifs

# run gdal grid  to get the final output if necessary