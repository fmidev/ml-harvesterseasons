#!/bin/env bash
# Extract data from EC-Earth3 historical and SSP434 runs for the period 2015-2023 and write to csv files
# Usage: ./make-cmip-data.sh lon lat

# Ensure the script is called with the required arguments
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <latitude> <longitude> <location> <id> <start_date> <end_date>"
    echo "Example: $0 60.39475 24.4459 horta 002956 2013-01-01 2023-08-31"
    exit 1
fi

# Input arguments
LAT=$1
LON=$2
LOCATION=$3
LOCATION_ID=$4
START_DATE=$5
END_DATE=$6

# Calculate the bounding box with a 0.703-degree margin
BBOX=$(echo "$LON - 0.703" | bc -l),$(echo "$LON + 0.703" | bc -l),$(echo "$LAT - 0.703" | bc -l),$(echo "$LAT + 0.703" | bc -l)
echo "Extracting data for bbox: $BBOX, location: $LOCATION"

# Run the CDO command in parallel for the specified variables
parallel 'cdo outputtab,date,lat,lon,value -sellonlatbox,'$BBOX' \
    -mergetime {1}_day_EC-Earth3-CC_historical_r1i1p1f1_gr_20000101-20141231.nc \
    -selyear,'2015/2023' {1}_day_EC-Earth3_ssp434_r101i1p1f1_gr_20150101-21001231.nc \
    > ece3-'$LOCATION'-{1}.csv' ::: tasmin tasmax pr sfcWind

if [ $? -eq 0 ]; then
    echo "cdo command completed successfully."
else
    echo "cdo command failed. Exiting."
    exit 1
fi

parallel 'sed -i "s/ \+/,/g" {}' ::: ece3-$LOCATION-*.csv

# Call the first Python script with the location name as input
echo "Running first Python script for location: $LOCATION"
python3 /home/ubuntu/ml-harvesterseasons/bin/cmip6_csv_operations.py --location $LOCATION

# Call the second Python script with the location, ID, start year, and end year as inputs
echo "Running second Python script for location: $LOCATION, ID: $LOCATION_ID, dates: $START_DATE to $END_DATE"
python3 /home/ubuntu/ml-harvesterseasons/bin/eobs_file_operations.py --location $LOCATION --id $LOCATION_ID --start_date $START_DATE --end_date $END_DATE

echo "All tasks complete."
