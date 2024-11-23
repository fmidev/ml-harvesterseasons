import pandas as pd
import os
import sys

loc = sys.argv[1]
preds = ["pr", "sfcWind", "tasmax", "tasmin"]

# Construct file paths for each predictor
file_paths = [f"ece3-pred-{loc}-{pred}.csv" for pred in preds]
all_dfs = []

for pred_index, file_path in enumerate(file_paths):

    temp_file_path = f"/home/ubuntu/data/cmip6/temp_{os.path.basename(file_path)}"
    
    # Remove first 3 and last 4 lines from the file
    with open(f"/home/ubuntu/data/cmip6/{file_path}", 'r') as infile, open(temp_file_path, 'w') as outfile:
        lines = infile.readlines()
        outfile.writelines(lines[2:-3])

    # Step 1: Load the CSV data
    df = pd.read_csv(temp_file_path)

    os.remove(temp_file_path)

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

# Reorder columns to place lat and lon columns after the date column
combined_df = combined_df[['date'] + lat_lon_columns + [col for col in combined_df.columns if col not in ['date'] + lat_lon_columns]]

combined_df = combined_df.rename(columns={'date': 'utctime'})

combined_df['utctime'] = pd.to_datetime(combined_df['utctime'], errors='coerce')
combined_df['dayofyear'] = combined_df['utctime'].dt.dayofyear

# Save the combined DataFrame to a new CSV file
combined_df.to_csv(f"/home/ubuntu/data/ML/training-data/OCEANIDS/prediction_data_ece3-{loc}-2015-2050.csv", index=False)

print(combined_df)