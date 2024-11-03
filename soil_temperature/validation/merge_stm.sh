#!/bin/bash

# Define the parent directory containing station folders
parent_dir="/home/ubuntu/data/ML/validation-data/soiltemp/ismn/20231101_20241101"
# parent_dir="/mnt/c/Users/prakasam/Documents/era5-land/soil_temp/data/val"

# Loop through each station directory
for station_dir in "$parent_dir"/*; do
    # Ensure it's a directory
    if [ -d "$station_dir" ]; then
        # Get the station name from the directory path
        station_name=$(basename "$station_dir")
        
        # Define the output file name within each station folder
        output_file="$parent_dir/${station_name}.txt"
        
        # Clear the output file if it exists
        > "$output_file"

        # Loop through each substation directory within the station
        for substation_dir in "$station_dir"/*; do
            # Ensure it's a directory
            if [ -d "$substation_dir" ]; then
                # Find the .stm files with the specific pattern
                for file in "$substation_dir"/*_ts_*.stm; do
                    # Check if the file exists to handle cases where there might be no match
                    if [ -f "$file" ]; then
                        # Append the content of the file to the station-specific output file
                        cat "$file" >> "$output_file"
                    fi
                done
            fi
        done

        echo "Merged .stm files for $station_name are saved in $output_file"
    fi
done
