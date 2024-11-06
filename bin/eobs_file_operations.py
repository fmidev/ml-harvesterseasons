import pandas as pd
import numpy as np

loc = 'sitia'
preds = {
    'FG': 'WS_PT1H_AVG', 
    #'PR': 'PR_PT24H_SUM',
    #'FX': 'WG_PT1H_MAX',
    # Add more predictors here as needed
}
id = '023330'

# Define the start and end dates
start_date = pd.to_datetime('2000-01-01')
end_date = pd.to_datetime('2023-08-31')

# Load your main dataset
main_df = pd.read_csv(f'/home/ubuntu/data/ML/training-data/OCEANIDS/ece3-{loc}.csv')

# Ensure the 'date' column is in datetime format
main_df['date'] = pd.to_datetime(main_df['date'], format='%Y-%m-%d', errors='coerce')  # Handle invalid dates if needed

# Filter out rows before and after specified dates
main_df = main_df[(main_df['date'] >= start_date) & (main_df['date'] <= end_date)].copy()

# Function to find the starting line with real values
def find_start_line(file_path):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            columns = line.strip().split(',')
            if len(columns) >= 4 and columns[3].strip() != '-9999':
                return i
    return -1  # Return -1 if no valid value is found

# Loop through each predictor, read data from its respective text file, and add to main_df
for pred, pred_name in preds.items():
    selected_lines = []
    dates = []

    file_path = f'/home/ubuntu/data/eobs/{pred.lower()}_blend/{pred}_STAID{id}.txt'
    start_line = find_start_line(file_path)

    if start_line == -1:
        print(f"No valid values found for predictor '{pred}'.")
        continue

    # Read the specific lines from the text file for the current predictor
    with open(file_path) as f:
        for i, line in enumerate(f):
            if i >= start_line:
                columns = line.strip().split(',')  # Split by comma
                if len(columns) >= 4:  # Check if there is a fourth column
                    dates.append(columns[2].strip())  # Append the date column
                    value = columns[3].strip()
                    selected_lines.append(np.nan if value == '-9999' else value)  # Convert -9999 to NaN

    # Debugging: Print the first few selected lines and dates
    #print(f"First few selected lines for {pred}: {selected_lines[:5]}")
    #print(f"First few dates for {pred}: {dates[:5]}")

    # Convert the selected lines to a DataFrame
    new_column_df = pd.DataFrame({'date': dates, pred_name: selected_lines})

    # Ensure data type is consistent (e.g., float), divide by 10
    new_column_df[pred_name] = pd.to_numeric(new_column_df[pred_name], errors='coerce') / 10

    # Debugging: Print the first few converted values
    #print(f"First few converted values for {pred}: {new_column_df[pred_name].head()}")

    new_column_df['date'] = pd.to_datetime(new_column_df['date'], format='%Y%m%d', errors='coerce')

    # Merge the new column with the main DataFrame based on the date column
    main_df = main_df.merge(new_column_df, on='date', how='left')

# Rename the 'date' column to 'utctime'
main_df.rename(columns={'date': 'utctime'}, inplace=True)

# Replace -9999 with NaN across the DataFrame
main_df.replace(-9999, np.nan, inplace=True)

# Save the updated dataset to a new CSV file
#main_df.to_csv(f"/home/ubuntu/data/ML/training-data/OCEANIDS/ece3-{loc.capitalize()}.csv", index=False)
print(main_df)
