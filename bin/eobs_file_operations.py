import pandas as pd

loc='sitia'
id='023330'
pred='FG'

# Load your main dataset
main_df = pd.read_csv('/home/ubuntu/data/ML/training-data/OCEANIDS/ece3-'+loc+'.csv')

# Define the range of lines you want to extract from the text file
start_line = 13170  # Start line (e.g., line 10)

selected_lines = []

# Read the specific lines from the text file and store them in a list
with open('/home/ubuntu/data/eobs/'+pred.lower()+'_blend/'+pred+'_STAID'+id+'.txt') as f:
    for i, line in enumerate(f):
        if i >= start_line:
            columns = line.strip().split(',')  # Split by comma
            if len(columns) >= 4:  # Check if there is a fourth column
                selected_lines.append(columns[3].strip())  # Append the fourth column

# Convert the list to a DataFrame
new_column_df = pd.DataFrame(selected_lines, columns=[''])
print(new_column_df)

# Check if the row counts match before adding the column
if len(main_df) == len(new_column_df):
    # Add the new column to the main DataFrame
    main_df['new_column'] = new_column_df['new_column']
else:
    print("Row count mismatch: check selected lines or main data rows.")
    print(len(main_df))
    print(len(new_column_df))

# Save the updated dataset to a new CSV file
#main_df.to_csv(f"ece3-{loc.capitalize()}.csv", index=False)
print(main_df)