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
source =  "desm.harvesterseasons.com:8080"


# nan values
nan = float("nan")


# eu_data = pd.read_csv(path_eu_data,sep=";")
pd.options.display.max_columns = None



def smartmet_ts_query_multiplePointsByID_tstep(source,start,end,tstep,pardict,llpdict):
    #timeseries query to smartmet-server
    #start date, end date, timestep, list of lats&lons, parameters as dictionary
    #returns dataframe
    latlons=[]
    for pair in llpdict.values():
        for i in pair:
            latlons.append(i)
    staids=[]
    for key in llpdict.keys():
        staids.append(key)
    
    # Timeseries query
    query='http://'+source+'/timeseries?latlons='
    for nro in latlons:
        query+=str(nro)+','
    query=query[0:-1]
    query+='&param=utctime,'
    for par in pardict.values():
        query+=par+','
        query=query[0:-1]
    #query+='&starttime='+start+'&endtime='+end+'&timestep='+tstep+'&hour=00&format=json&precision=full&tz=utc&timeformat=sql&grouplocations=1'  # test
    query+='&starttime='+start+'&endtime='+end+'&timestep='+tstep+'&format=json&precision=full&tz=utc&timeformat=sql&grouplocations=1'  # this one
    #query+='&starttime='+start+'&endtime='+end+'&timestep=data&format=json&precision=full&producer=EDTE&tz=utc&timeformat=sql&grouplocations=1'
    print(query)

    # Response to dataframe
    response=requests.get(url=query)
    results_json=json.loads(response.content)
    #print(results_json)
    for i in range(len(results_json)):
        res1=results_json[i]
        for key,val in res1.items():
            if key != 'utctime' and isinstance(val, str):   
                res1[key] = val.strip('[]').split()
    df=pd.DataFrame(results_json)   
    df.columns=['utctime']+list(pardict.keys()) # change headers to params.keys
    df['utctime']=pd.to_datetime(df['utctime'])
    expl_cols=list(pardict.keys())
    df=df.explode(expl_cols)
    df['latlonsID'] = df.groupby(level=0).cumcount().add(1).astype(str).radd('')
    df=df.reset_index(drop=True)
    df['latlonsID'] = df['latlonsID'].astype('int')
    max=df['latlonsID'].max()
    df['latitude']=''
    df['longitude']=''
    df['pointID']=''
    i=1
    j=0
    while i <= max:
        df.loc[df['latlonsID']==i,'latitude']=latlons[j]
        df.loc[df['latlonsID']==i,'longitude']=latlons[j+1]
        df.loc[df['latlonsID']==i,'pointID']=staids[i-1]
        j+=2
        i+=1
    df=df.drop(columns='latlonsID')
    df=df.astype({col: 'float32' for col in df.columns[1:-1]})
    #print(df)
    return df




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


print(ismn_df.head())

print(len(ismn_df), "rows in the dataframe")
# remove nan values in df

ismn_df = ismn_df.dropna(subset=["latitude", "longitude"])
print(len(ismn_df), "rows in the dataframe after removing nan values")

# remove duplicate long,lat

# ismn_df = ismn_df.drop_duplicates(subset=["latitude", "longitude"])
staion_lat_long = (
    ismn_df[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
)

print(staion_lat_long.head())
print(len(staion_lat_long), "rows in the dataframe after removing duplicate values")

lat = staion_lat_long["latitude"].tolist()
lon = staion_lat_long["longitude"].tolist()
# points=staion_lat_long['POINT_ID'].values.tolist()

pointids = list(range(len(staion_lat_long)))
# pointids = list(range(1,6000)) # can only acc


llpdict = {i: [j, k] for i, j, k in zip(pointids, lat, lon)}


# EXAMPLE get subdict based on list of pointids:
# pointids = list(range(1, 2))
# llpdict = dict((k, llpdict[k]) for k in pointids if k in llpdict)
# print(llpdict)

# ### EDTE predictors
# # 24h accumulations
edte = [
    {"stl1x": "STL1X-C:EDTE:5068:1:0:0"},  # Surface latent heat flux (J m-2)
]
temp_df = pd.DataFrame(columns=["utctime", "latitude", "longitude", "pointID"])

# start = begin_year + "-11-01T00:00:00Z"  # start date
# end = end_year + "-11-01T23:59:59Z"  # end date
# start = "20231101T000000Z"
# end = "20241131T000000Z"

timestamp = "data"

y_start='2023'
y_end='2024'
nan=float('nan')
source='desm.harvesterseasons.com:8080'
start=y_start+'-11-01T00:00:00Z'
end=y_end+'-11-01T23:59:59Z'

for pardict in edte:
    

    key, value = list(pardict.items())[0]

    temp_df = smartmet_ts_query_multiplePointsByID_tstep(source,start,end,timestamp,pardict,llpdict)
    print(temp_df.head(), "head")
    print(len(temp_df), "rows in the dataframe")
    print(temp_df.columns, "columns")


temp_df.to_csv(
    "/home/ubuntu/data/ML/validation-data/soiltemp/ismn/20231101_20241101/timeseries_stl1x_20231101_20241101.csv",
    index=False,
)
# Query URL: http://desm.harvesterseasons.com:8080/timeseries?latlon=41.26504,-5.38049,41.39392,-5.32146,41.3001,-5.24704&param=utctime,STL1X-C:EDTE:5068:1:0:1:0&starttime=20231101T060000Z&endtime=20241101T060000Z&hour=06&format=json&precision=full&tz=utc&timeformat=sql&grouplocations=1

# combine time series data

# time_series_frames = [era5l00_df, era5l0012_df]


# time_series_df = multi_merger_df(data_frames=time_series_frames)

# # save time_series_df

# time_series_df.to_csv(
#     "/home/ubuntu/data/ML/training-data/soiltemp/timeseries_features_latest.csv",
#     index=False,
# )
