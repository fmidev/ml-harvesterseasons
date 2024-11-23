import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('/home/ubuntu/data/ML/training-data/OCEANIDS/prediction_data_ece3-bremerhaven-2015-2050.csv')

# Convert 'utctime' to datetime and extract month
df['utctime'] = pd.to_datetime(df['utctime'])
df['month'] = df['utctime'].dt.month

# Sum the values for columns like 'sfcWind-1' to 'sfcWind-4' into a single column
df['sfcWind_sum'] = df[['sfcWind-1', 'sfcWind-2', 'sfcWind-3', 'sfcWind-4']].sum(axis=1) / 4

# Group by month and calculate the average, minimum, and maximum for each month
monthly_stats = df.groupby('month').agg({
    'sfcWind_sum': ['mean', 'min', 'max'],
    'WS_PT24H_AVG': ['mean', 'min', 'max']
})

# Flatten the MultiIndex columns
monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns.values]

# Reset index to make 'month' a column again
monthly_stats.reset_index(inplace=True)

# Calculate the difference between 'sfcWind_sum' and 'WS_PT24H_AVG'
monthly_stats['sfcWind_WS_diff_mean'] = monthly_stats['sfcWind_sum_mean'] - monthly_stats['WS_PT24H_AVG_mean']
monthly_stats['sfcWind_WS_diff_min'] = monthly_stats['sfcWind_sum_min'] - monthly_stats['WS_PT24H_AVG_min']
monthly_stats['sfcWind_WS_diff_max'] = monthly_stats['sfcWind_sum_max'] - monthly_stats['WS_PT24H_AVG_max']

# Merge the calculated statistics back to the original DataFrame
df = df.merge(monthly_stats, on='month', how='left')

# Save the updated DataFrame to a new CSV file
df.to_csv('/home/ubuntu/data/ML/training-data/OCEANIDS/prediction_data_ece3-bremerhaven-2015-2050-updated.csv', index=False)
print(df)