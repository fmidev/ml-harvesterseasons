import pandas as pd

df = pd.read_csv('/home/ubuntu/data/ML/training-data/RR-fin/DEM_10000_training_locs.csv')
df = df[['POINT_ID', 'TH_LAT', 'TH_LONG'] + [col for col in df.columns if col not in ['POINT_ID', 'TH_LAT', 'TH_LONG']]]
df.to_csv('/home/ubuntu/data/ML/training-data/RR-fin/DEM_10000_training_locs-fix.csv', index=False)