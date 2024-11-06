import pandas as pd
from functools import reduce
import time, glob, os, sys
from sklearn.metrics import mean_absolute_error, mean_squared_error

# import functions as fcts
import numpy as np
import itertools
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


startTime = time.time()

path_ismn_dir = "/home/ubuntu/data/ML/validation-data/soiltemp/ismn/20231101_20241101/"
path_prediction_file = "/home/ubuntu/data/ML/validation-data/soiltemp/ismn/timeseries_stl1x_20231101_20241101.csv"


# read text files and conactenate them
txt_files = glob.glob(os.path.join(path_ismn_dir, "*.txt"))
if not txt_files:
    print("No .txt files found in the directory.")
    sys.exit()


column_names = [
    "nominal_date",
    "nominal_datetime",
    "actual_date",
    "datetime",
    "networkname",
    "network",
    "station",
    "latitude",
    "longitude",
    "elevation",
    "depth_from",
    "depth_to",
    "value",
    "ismn_qc",
    "data_prov_qc",
]
ismn_df = pd.DataFrame(columns=column_names)
for file in txt_files:
    df = pd.read_csv(
        file, sep=r"\s+", header=None
    )  # Using regex to handle multiple types of tabs and spaces
    df.columns = column_names

    ismn_df = pd.concat([ismn_df, df], ignore_index=True)

print(ismn_df.dtypes, ismn_df.shape)
unique_values = ismn_df["ismn_qc"].unique()
print(unique_values, "unique values of ismn_qc")

# soil temperature;M;parameter value missing
# soil temperature;G;good
# Filter the dataframe to get rows where 'depth_from' values are less than 0.10
filtered_df = ismn_df[(ismn_df["depth_from"] >= 0.05) & (ismn_df["depth_to"] <= 0.10)]
print(filtered_df.shape, "rows with depth_from values less than 0.10 and greater than 0.05")
# Discard rows that have 'ismn_qc' not equal to 'G'
filtered_df = filtered_df[filtered_df["ismn_qc"] == "G"]
print(
    filtered_df.shape, "rows left after discarding rows with ismn_qc not equal to 'G'"
)

print(filtered_df['depth_from'].unique())
print(filtered_df['depth_to'].unique())
# Drop unnecessary columns
columns_to_drop = [
    "nominal_date",
    "nominal_datetime",
    "networkname",
    "network",
    "ismn_qc",
    "data_prov_qc",
    "elevation",
]
filtered_df.drop(columns=columns_to_drop, inplace=True)


filtered_df["actual_date"] = pd.to_datetime(filtered_df["actual_date"])

filtered_df = (
    filtered_df.groupby(
        [pd.Grouper(key="actual_date", freq="H"), "latitude", "longitude"]
    )["value"]
    .mean()
    .reset_index()
)
filtered_df = filtered_df.dropna(subset=["latitude", "longitude"])
filtered_df = filtered_df.dropna(axis=0, how="any")

print(filtered_df.head())
print(filtered_df.tail())
# Rename columns
filtered_df.rename(
    columns={"actual_date": "utctime", "value": "stl1x_true"}, inplace=True
)
print(filtered_df.head())
filtered_df["dayOfYear"] = filtered_df["utctime"].dt.dayofyear


print(filtered_df.head())
prediction_df = pd.read_csv(path_prediction_file)
# Drop the 'depth_to' column from the filtered dataframe
prediction_df.drop(columns=["pointID"], inplace=True)
prediction_df.rename(columns={"stl1x": "stl1x_pred"}, inplace=True)
prediction_df = prediction_df.dropna(axis=0, how="any")
prediction_df["utctime"] = pd.to_datetime(prediction_df["utctime"])
print(prediction_df.dtypes)
print(prediction_df.head())
prediction_df["dayOfYear"] = prediction_df["utctime"].dt.dayofyear
print(prediction_df.head())

# merge columns in the two dataframes
merged_df = pd.merge(
    filtered_df,
    prediction_df,
    on=["utctime", "latitude", "longitude", "dayOfYear"],
    how="inner",
)
merged_df = merged_df.dropna(axis=0, how="any")
print(merged_df.head())

mae = mean_absolute_error(merged_df["stl1x_true"], merged_df["stl1x_pred"])
rmse = mean_squared_error(
    merged_df["stl1x_true"], merged_df["stl1x_pred"], squared=False
)
print(rmse)
print(mae)
