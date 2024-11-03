#!/usr/bin/env python3
import requests, os, time, glob, json, sys
import pandas as pd
from functools import reduce
# import functions as fcts
import numpy as np
import itertools
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
### SmarMet-server timeseries query to fetch ERA5-Land training data for machine learning (copied from bin and edited )

startTime = time.time()

path_ismn_dir = "/home/ubuntu/data/ML/validation-data/soiltemp/ismn/20231101_20241101/"
source = "desm.harvesterseasons.com:8080"

y_start='2023'
y_end='2024'
sd_yrs=list(range(int(y_start),int(y_end)+1))
nan=float('nan')

start=y_start+'-11-01T00:00:00Z' # start date
end=y_end+'-11-01T23:59:59Z' # end date
tstep='data'
# eu_data = pd.read_csv(path_eu_data,sep=";")
pd.options.display.max_columns = None
# function to merge dfs
def merge_df(df1, df2):
    if df1.empty:
        df = pd.concat([df1, df2], axis=1)
        # df = df.T.drop_duplicates().T
    else:
        df = pd.merge(
            df1, df2, how="inner", on=["utctime", "latitude", "longitude", "pointID"]
        )
    return df

def multi_merger_df(data_frames):
    df = reduce(
        lambda left, right: pd.merge(
            left, right, on=["utctime", "latitude", "longitude", "pointID"], how="inner"
        ),
        data_frames,
    )
    return df


# read text files and conactenate them
txt_files = glob.glob(os.path.join(path_ismn_dir, "*.txt"))
if not txt_files:
    print("No .txt files found in the directory.")
    sys.exit()

ismn_df = pd.DataFrame()

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
for file in txt_files:
    df = pd.read_csv(
        file, sep=r"\s+"
    )  # Using regex to handle multiple types of tabs and spaces
    print(df.head())
    # ismn_df.append(df)

ismn_df.columns = column_names

print(ismn_df.head())

# remove nan values in df

ismn_df = ismn_df.dropna(subset=["latitude", "longitude"])

# remove duplicate long,lat

ismn_df = ismn_df.drop_duplicates(subset=["latitude", "longitude"])
staion_lat_long = ismn_df[
    ["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)

print(staion_lat_long.head())
print(len(staion_lat_long))

# lat = staion_lat_long["latitude"].tolist()
# lon = staion_lat_long["longitude"].tolist()
# # points=staion_lat_long['POINT_ID'].values.tolist()

# pointids = list(range(len(ismn_df)))
# # pointids = list(range(1,6000)) # can only acc


# llpdict = {i: [j, k] for i, j, k in zip(pointids, lat, lon)}


# # EXAMPLE get subdict based on list of pointids:
# # pointids = list(range(1,10))
# # llpdict = dict((k, llpdict[k]) for k in pointids
# #            if k in llpdict)
# # print(llpdict)

# ### EDTE predictors
# # 24h accumulations
# edte = [
#     {'stl1x':'STL1X-C:EDTE:5068:1:0:1:0'},  # Surface latent heat flux (J m-2)
#     ]
# stl1_df = pd.DataFrame(columns=["utctime", "latitude", "longitude", "pointID"])
# for pardict in edte:
#     for hour in ["00", "06", "12", "18", "23"]:
#         key, value = list(pardict.items())[0]
#         print(source, start, end, hour, pardict, llpdict)
#         temp_df = fcts.smartmet_ts_query_multiplePointsByID_hour(
#             source, start, end, hour, pardict, llpdict
#         )
#         stl1_df = merge_df(stl1_df, temp_df)

# stl1_df.to_csv(
#     "/home/ubuntu/data/ML/validation-data/soiltemp/ismn/20231101_20241101/timeseries_stl1x_20231101_20241101.csv",
#     index=False,
# )
### 00 UTC parameters (24h accumulated)
hour = "00"
start = "20150101T000000Z"
end = "20221231T000000Z"



# combine time series data

# time_series_frames = [era5l00_df, era5l0012_df]


# time_series_df = multi_merger_df(data_frames=time_series_frames)

# # save time_series_df

# time_series_df.to_csv(
#     "/home/ubuntu/data/ML/training-data/soiltemp/timeseries_features_latest.csv",
#     index=False,
# )
