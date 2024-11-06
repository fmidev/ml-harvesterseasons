#!/bin/bash

# Create or clear the output CSV file
cd /home/ubuntu/data/ML/training-data/RR/eobs/
output_file="extracted_data.csv"
echo "STAID,LAT,LON" > "$output_file"  # Add headers to the output file

for file in eobs_2000-2020_*.csv; do
    echo "Processing $file"
    # Extract the STAID, LAT, and LON values from the first data row of each file
    awk -F, 'NR==2 {print $1 "," $4 "," $5}' "$file" >> "$output_file"
done