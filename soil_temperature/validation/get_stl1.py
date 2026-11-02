#!/usr/bin/env python3
import glob
import itertools
import json
import os
import sys
import time
import warnings
from functools import reduce

import functions as fcts
import numpy as np
import pandas as pd
import requests

warnings.simplefilter(action="ignore", category=FutureWarning)
# SmarMet-server timeseries query to fetch ERA5-Land training data for machine learning (copied from bin and edited )

startTime = time.time()


source = "desm.harvesterseasons.com:8080"


pd.options.display.max_columns = None


def concat_df(dfs):
    result = pd.concat(dfs)
    return result


norway_lat_long = (
    "/home/ubuntu/data/ML/validation-data/soiltemp/norway_24hr/norway_latlong.csv"
)
locations_df = pd.read_csv(norway_lat_long)
lat = locations_df["Latitude"].tolist()
lon = locations_df["Longitude"].tolist()
# points=lucas_df['POINT_ID'].values.tolist()

pointids = list(range(len(locations_df)))
# pointids = list(range(1,6000)) # can only acc

llpdict = {i: [j, k] for i, j, k in zip(pointids, lat, lon)}

# EXAMPLE get subdict based on list of pointids:
pointids = list(range(1, 10))
llpdict = dict((k, llpdict[k]) for k in pointids if k in llpdict)
print(llpdict)

# Hourly data
era5l0012 = [
    {"stl1": "STL1-K:ERA5L:5022:9:7:1:0"},  # Soil temperature level 1 (K)
]


### 00 UTC parameters (24h accumulated)
hour = "00"
start = "20230101T000000Z"
end = "20241231T000000Z"


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


era5l0012_df = pd.DataFrame()
### 00 and 12 UTC parameters
for pardict in era5l0012:
    hour = "00"
    key, value = list(pardict.items())[0]
    # Get the data from the API
    temp_df = fcts.smartmet_ts_query_multiplePointsByID_hour(
        source, start, end, hour, pardict, llpdict
    )
    temp_df.rename({key: key + "-00"}, axis=1, inplace=True)
    print("**********00 and 12 UTC parameters****************")
    temp_df["utctime"] = temp_df["utctime"].dt.date
    era5l0012_df = merge_df(era5l0012_df, temp_df)

    hour = "12"

    temp_df = fcts.smartmet_ts_query_multiplePointsByID_hour(
        source, start, end, hour, pardict, llpdict
    )

    temp_df["utctime"] = temp_df["utctime"].dt.date

    temp_df.rename({key: key + "-12"}, axis=1, inplace=True)

    era5l0012_df = merge_df(era5l0012_df, temp_df)


print("*******************era5l0012_df*********************")
print(era5l0012_df.head())

era5l0012_df["pointID"] = era5l0012_df["pointID"].astype("Int64")
era5l0012_df["utctime"] = pd.to_datetime(era5l0012_df["utctime"])

era5l0012_df.to_csv(
    "/home/ubuntu/data/ML/validation-data/soiltemp/norway_24hr/timeseries_stl1_norway_00_12.csv",
    index=False,
)


print(era5l0012_df.info())

print("Running for fmi_ground_temp_3hr")

norway_lat_long = "/home/ubuntu/data/ML/validation-data/soiltemp/fmi_ground_temp_3hr/fmi_tg_latlong.csv"
locations_df = pd.read_csv(norway_lat_long)
lat = locations_df["Latitude"].tolist()
lon = locations_df["Longitude"].tolist()
# points=lucas_df['POINT_ID'].values.tolist()

pointids = list(range(len(locations_df)))
# pointids = list(range(1,6000)) # can only acc

llpdict = {i: [j, k] for i, j, k in zip(pointids, lat, lon)}

# EXAMPLE get subdict based on list of pointids:
pointids = list(range(1, 10))
llpdict = dict((k, llpdict[k]) for k in pointids if k in llpdict)
print(llpdict)

# Hourly data
era5l0012 = [
    {"stl1": "STL1-K:ERA5L:5022:9:7:1:0"},  # Soil temperature level 1 (K)
]


### 00 UTC parameters (24h accumulated)
hour = "00"
start = "20230101T000000Z"
end = "20241231T000000Z"

era5l0012_df = pd.DataFrame()
### 00 and 12 UTC parameters
for pardict in era5l0012:
    hour = "00"
    key, value = list(pardict.items())[0]
    # Get the data from the API
    temp_df = fcts.smartmet_ts_query_multiplePointsByID_hour(
        source, start, end, hour, pardict, llpdict
    )
    temp_df.rename({key: key + "-00"}, axis=1, inplace=True)
    print("**********00 and 12 UTC parameters****************")
    temp_df["utctime"] = temp_df["utctime"].dt.date
    era5l0012_df = merge_df(era5l0012_df, temp_df)

    hour = "12"

    temp_df = fcts.smartmet_ts_query_multiplePointsByID_hour(
        source, start, end, hour, pardict, llpdict
    )

    temp_df["utctime"] = temp_df["utctime"].dt.date

    temp_df.rename({key: key + "-12"}, axis=1, inplace=True)

    era5l0012_df = merge_df(era5l0012_df, temp_df)


print("*******************era5l0012_df*********************")
print(era5l0012_df.head())

era5l0012_df["pointID"] = era5l0012_df["pointID"].astype("Int64")
era5l0012_df["utctime"] = pd.to_datetime(era5l0012_df["utctime"])

era5l0012_df.to_csv(
    "/home/ubuntu/data/ML/validation-data/soiltemp/fmi_ground_temp_3hr/timeseries_stl1_fmi_00_12.csv",
    index=False,
)
