#!/bin/bash

log_dir=/home/ubuntu/data/ml-harvestability/logs/
countries=/home/ubuntu/ml-harvesterseasons/harvestability/all_countries.txt
upload_predict=/home/ubuntu/ml-harvesterseasons/harvestability/prediction/upload_predict_direct.sh

# read txt file line ny line
while IFS=$'\n' read country
do
    #predict for the country
    result=`ps aux | grep -i "upload_predict_direct.sh" | grep -v "grep" | wc -l`
    
    while [[ $result -ge 3 ]]
    do
        sleep 120
        result=`ps aux | grep -i "upload_predict_direct.sh" | grep -v "grep" | wc -l`
    done
    echo "Running ... $country"
    nohup bash $upload_predict $country >> $log_dir$country-predictions.log 2>&1 &
    #fi
 

done < $countries
