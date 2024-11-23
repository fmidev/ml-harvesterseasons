import pandas as pd
import numpy as np
import sys

loc = sys.argv[1]
preds = {
    'FG': 'WS_PT24H_AVG', 
    'RR': 'TP_PT24H_SUM',
    'FX': 'WG_PT24H_MAX',
    'TN': 'TN_PT24H_MIN',
    'TX': 'TX_PT24H_MAX',
    # Add more predictors here as needed
}
id = sys.argv[2]
file_name = f'training_data_oceanids-{loc}-sf_2000-2023.csv'
output_file_name = f'training_data_oceanids-{loc}-sf_2000-2023.csv'

# Define the start and end dates
start_date = '2000-01-01'
start_date = pd.to_datetime(start_date)
end_date = '2023-08-31'
end_date = pd.to_datetime(end_date)

# Load your main dataset
main_df = pd.read_csv(f'/home/ubuntu/data/ML/training-data/OCEANIDS/{loc}/{file_name}')

# Ensure the 'utctime' column is in datetime format
main_df['utctime'] = pd.to_datetime(main_df['utctime'], format='%Y-%m-%d', errors='coerce')  # Handle invalid dates if needed

# Filter out rows before and after specified dates
main_df = main_df[(main_df['utctime'] >= start_date) & (main_df['utctime'] <= end_date)].copy()

# Loop through each predictor, read data from its respective text file, and add to main_df
for pred, pred_name in preds.items():
    selected_lines = []
    dates = []

    file_path = f'/home/ubuntu/data/eobs/{pred.lower()}_blend/{pred}_STAID{id}.txt'

    # Read the specific lines from the text file for the current predictor
    with open(file_path) as f:
        for i, line in enumerate(f):
                columns = line.strip().split(',')  # Split by comma
                if len(columns) >= 4:  # Check if there is a fourth column
                    dates.append(columns[2].strip())  # Append the date column
                    value = columns[3].strip()
                    selected_lines.append(np.nan if value == '-9999' else value)  # Convert -9999 to NaN

    # Debugging: Print the first few selected lines and dates
    #print(f"First few selected lines for {pred}: {selected_lines[:5]}")
    #print(f"First few dates for {pred}: {dates[:5]}")

    # Convert the selected lines to a DataFrame
    new_column_df = pd.DataFrame({'utctime': dates, pred_name: selected_lines})

    # Ensure data type is consistent (e.g., float), divide by 10
    new_column_df[pred_name] = pd.to_numeric(new_column_df[pred_name], errors='coerce') / 10

    # Debugging: Print the first few converted values
    #print(f"First few converted values for {pred}: {new_column_df[pred_name].head()}")

    new_column_df['utctime'] = pd.to_datetime(new_column_df['utctime'], format='%Y%m%d', errors='coerce')

    # Merge the new column with the main DataFrame based on the date column
    main_df = main_df.merge(new_column_df, on='utctime', how='left')



#Add dayofyear column
main_df['dayofyear'] = main_df['utctime'].dt.dayofyear

# Save the updated dataset to a new CSV file
main_df.to_csv(f"/home/ubuntu/data/ML/training-data/OCEANIDS/{output_file_name}", index=False)
print(main_df)
