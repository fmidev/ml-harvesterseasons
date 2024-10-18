import pandas as pd

df = pd.read_csv("/home/ubuntu/data/ML/training-data/OCEANIDS/training_data_oceanids_Malaga-simple-sf_2000-2023.csv")


print(df)

df.to_csv("/home/ubuntu/data/ML/training-data/OCEANIDS/training_data_oceanids_Malaga-simple-sf_2000-2023.csv", index=False)