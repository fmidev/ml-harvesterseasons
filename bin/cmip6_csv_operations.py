import pandas as pd
import os

# Adjust files with something like(depends on file names): parallel 'sed "s/ \+/,/g" {} > ece3-(name of loc)-{= s:ece3-([^--]+)--.*\.csv$:\1: =}.csv' ::: *(**coordinates**)*.csv first

loc = "ploumanach"
preds = ["pr", "sfcWind", "tasmax", "tasmin"]

# Construct file paths for each predictor
file_paths = [f"ece3-{loc}-{pred}.csv" for pred in preds]
all_dfs = []

for pred_index, file_path in enumerate(file_paths):

    temp_file_path = f"/home/ubuntu/data/cmip6/temp_{os.path.basename(file_path)}"
    
    # Remove first 3 and last 4 lines from the file
    with open(f"/home/ubuntu/data/cmip6/{file_path}", 'r') as infile, open(temp_file_path, 'w') as outfile:
        lines = infile.readlines()
        outfile.writelines(lines[3:-4])

    # Step 1: Load the CSV data
    df = pd.read_csv(temp_file_path)

    os.remove(temp_file_path)

    # Step 2: Create a unique identifier for each (lat, lon) pair
    df['Point'] = df.groupby(['lat', 'lon']).ngroup() + 1

    # Step 3: Pivot the DataFrame to make each point's value a separate column
    df_pivot = df.pivot(index='date', columns='Point', values='value').reset_index()

    # Step 4: Rename the columns as pred1-1, pred1-2, etc.
    df_pivot.columns = ['date'] + [f"{preds[pred_index]}-{i}" for i in range(1, len(df_pivot.columns))]

    # Append the processed DataFrame to the list
    all_dfs.append(df_pivot)

# Step 5: Merge DataFrames on 'date' to combine predictor columns side by side
combined_df = all_dfs[0]
for df in all_dfs[1:]:
    combined_df = combined_df.merge(df, on='date', how='outer')

# Step 6: Save the combined DataFrame to a new CSV file
combined_df.to_csv(f"/home/ubuntu/data/cmip6/ece3-{loc}.csv", index=False)

print(combined_df)