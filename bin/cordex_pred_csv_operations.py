#!/bin/env python3
import pandas as pd
import os
import sys

loc = sys.argv[1]
model = sys.argv[2]
preds = ["pr", "sfcWind", "tasmax", "tasmin", "maxWind"]

# Construct file paths for each predictor
file_paths = [f"{model}-{loc}-rcp45-{pred}.csv" for pred in preds]
all_dfs = []

def custom_date_parser(date_str):
    # Implement your custom date parsing logic here
    # For example, you might convert "YYYYMMDD" to a standard date format
    # Assuming date_str is in the format "YYYYMMDD"
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    
    # Adjust the day and month to fit a 30-day month calendar
    day = (day - 1) % 30 + 1
    month = (month - 1) % 12 + 1
    
    return pd.Timestamp(year, month, day)

for pred_index, file_path in enumerate(file_paths):

    # Step 1: Load the CSV data
    df = pd.read_csv(f"/home/ubuntu/data/cordex/{file_path}", sep='\s+', header=0, names=['date', 'lat', 'lon', 'value'])
    df['date'] = df['date'].apply(custom_date_parser)

    # Step 2: Create a unique identifier for each (lat, lon) pair
    df['Point'] = df.groupby(['lat', 'lon']).ngroup() + 1

    # Step 3: Pivot the DataFrame to make each point's value a separate column
    df_pivot = df.pivot(index='date', columns='Point', values='value').reset_index()

    # Step 4: Extract unique lat and lon values
    if pred_index == 0:
        lat_lon_df = df[['Point', 'lat', 'lon']].drop_duplicates().set_index('Point')
        lat_columns = {f'lat-{point}': lat_lon_df.loc[point, 'lat'] for point in lat_lon_df.index}
        lon_columns = {f'lon-{point}': lat_lon_df.loc[point, 'lon'] for point in lat_lon_df.index}

    # Step 5: Rename the columns as pred1-1, pred1-2, etc.
    df_pivot.columns = ['date'] + [f"{preds[pred_index]}-{i}" for i in range(1, len(df_pivot.columns))]

    # Append the processed DataFrame to the list
    all_dfs.append(df_pivot)

# Step 6: Merge DataFrames on 'date' to combine predictor columns side by side
combined_df = all_dfs[0]
for df in all_dfs[1:]:
    combined_df = combined_df.merge(df, on='date', how='outer')

# Step 7: Add lat and lon columns to the combined DataFrame
lat_lon_columns = []
for point in lat_lon_df.index:
    combined_df[f'lat-{point}'] = lat_columns[f'lat-{point}']
    combined_df[f'lon-{point}'] = lon_columns[f'lon-{point}']
    lat_lon_columns.extend([f'lat-{point}', f'lon-{point}'])

# Rename 'date' column to 'utctime'
combined_df.rename(columns={'date': 'utctime'}, inplace=True)

# Add dayofyear column
combined_df['dayofyear'] = pd.to_datetime(combined_df['utctime']).dt.dayofyear

# Reorder columns to place lat and lon columns after the date column
combined_df = combined_df[['utctime'] + lat_lon_columns + [col for col in combined_df.columns if col not in ['utctime'] + lat_lon_columns]]

# Save the combined DataFrame to a new CSV file
combined_df.to_csv(f"/home/ubuntu/data/ML/training-data/OCEANIDS/{model}-{loc}-cordex.csv", index=False)

#print(combined_df)