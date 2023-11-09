import os, time, random, warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings("ignore")
### XGBoost for precipitation downscaling and bias-adjusting (10/2023)

startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/precipitation-harmonia/' # training data
mod_dir='/home/ubuntu/data/ML/models/precipitation-harmonia/' # saved mdl
res_dir='/home/ubuntu/data/ML/results/precipitation-harmonia/'

### Read in 2D tabular training data
cols_own=[
    'utctime','latitude','longitude','pointID',
    'anor','evap','kx-00','lsm','mn2t-00','msl-00','mx2t-00',
    'q500-00','q700-00','q850-00','ro','rsn-00','sd-00','sdor',
    'sf','skt-00','slhf','slor','sshf','ssr','ssrd','strd',#'sst-00',
    't2m-00','t500-00','t700-00','t850-00','tcc-00','td2m-00','tp',
    'tsr','ttr','u10-00','u500-00','u700-00','u850-00','v10-00',
    'v500-00','v700-00','v850-00','z','z500-00','z700-00','z850-00'
]
#tp=pd.read_csv('/lustre/tmp/strahlen/mlbias/data/era5+_eobs_dtw_1995-2015_all6173_RRweight.csv', iterator=True,chunksize=1000,usecols=cols_own)
#df = pd.concat(tp, ignore_index=True) # read in chunks too large csv
fname='RR-training-all6766/RR_training_era5_431_2000-2020_all.csv' # training data
print(fname)
df_era5=pd.read_csv(data_dir+fname,usecols=cols_own)
fname='/home/ubuntu/data/ML/training-data/eobs/eobs_2000-2020_431.csv'
df_eobs=pd.read_csv(fname)
print(df_era5)

df_eobs.rename(columns={'DATE':'utctime','STAID':'pointID','LAT': 'latitude', 'LON': 'longitude'}, inplace=True)
print(df_eobs)


df=pd.merge(df_era5,df_eobs, on=['utctime','pointID','latitude','longitude'])

print(df_era5)
print(df_eobs)
print(df)


