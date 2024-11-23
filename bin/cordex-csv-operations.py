import pandas as pd
import os

# Read the CSV files
df1 = pd.read_csv('/home/ubuntu/data/ML/training-data/OCEANIDS/Bremerhaven/training_data_oceanids-Bremerhaven-sf_2000-2023.csv')
df2 = pd.read_csv('/home/ubuntu/data/ML/training-data/OCEANIDS/ece3-Bremerhaven.csv')

# Convert the date column to datetime format
df1['utctime'] = pd.to_datetime(df1['utctime'])
df2['utctime'] = pd.to_datetime(df2['utctime'])

# Filter df1 to include only data from 2013-07-01 to 2024-01-01
start_date = '2000-01-01'
end_date = '2023-08-31'
df1 = df1[(df1['utctime'] >= start_date) & (df1['utctime'] <= end_date)]

# Merge df1 and df2 on the 'utctime' column
df_merged = pd.merge(df1, df2[['utctime', 'WG_PT24H_MAX','WS_PT24H_AVG','TP_PT24H_SUM','TN_PT24H_MIN','TX_PT24H_MAX']], on='utctime', how='left')

#Add dayofyear column
df_merged['dayofyear'] = df_merged['utctime'].dt.dayofyear

# Save the updated df_merged to a new CSV file if needed
df_merged.to_csv('/home/ubuntu/data/ML/training-data/OCEANIDS/training_data_oceanids-Bremerhaven-sf_2000-2023.csv', index=False)
print(df_merged)
