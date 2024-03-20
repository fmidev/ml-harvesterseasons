import os, time, random, warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings("ignore")
### XGBoost for precipitation downscaling and bias-adjusting (2/2024)

startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/snowdepth/' # training data
mod_dir='/home/ubuntu/data/ML/models/snowdepth/' # saved mdl
res_dir='/home/ubuntu/data/ML/results/snowdepth/'

### Read in 2D tabular training data
# XGBoost fit 1.: latitude,longitude and tp not included
# XGBoost fit 2: tp not included
# tp had many NaN values -> leave out, 
# also u10-00 and v10-00 have NaN values, but are included so far
cols_own=[
    'utctime','pointID','latitude','longitude','slhf','sshf','ssrd',
    'strd','str','ssr','sf','laihv-00','lailv-00','sd-00','rsn-00',
    'stl1-00','swvl2-00','t2-00','td2-00','u10-00','v10-00',
    'ro','evap'
]

# CLMS columns: utctime, swe_clms, latitude, longitude, pointID
cols_clms=['utctime','pointID','swe_clms']

cols_eobs=['STAID','DATE','SD','HGHT']

# note: eobs and clms do not always have full years time series, unlike era5l
fera5l='SD_era5l_training_9790stations_2006-2020_all.csv.gz' # (predictors) era5-L training data
fclms='SD_clms_swe_training_9790stations_2006-2020.csv' # (predictors) clms training data
feobs='SD_eobs_fmi_training_9790stations_2006-2020.csv.gz' # eobs (predictand) training data

print(fera5l)
print(fclms)
print(feobs)

df_era5l=pd.read_csv(data_dir+fera5l,usecols=cols_own)
print(df_era5l)
df_clms=pd.read_csv(data_dir+fclms,usecols=cols_clms)
print(df_clms)
df_eobs=pd.read_csv(data_dir+feobs,usecols=cols_eobs)
print(df_eobs)
#df_eobs.rename(columns={'DATE':'utctime','STAID':'pointID','LAT': 'latitude', 'LON': 'longitude'}, inplace=True)
df_eobs.rename(columns={'DATE':'utctime','STAID':'pointID'}, inplace=True)
# SD column: replace nan's with 0
df_eobs['SD']=df_eobs['SD'].replace(np.nan,0)

# merge SD eobs, clms and era5l data
df_eobs_clms=pd.merge(df_clms,df_eobs, on=['utctime','pointID'])
df=pd.merge(df_era5l,df_eobs_clms, on=['utctime','pointID'])
# note; lat,lon not used in merging, thats's why commented:
#df=pd.merge(df_era5l,df_eobs, on=['utctime','pointID','latitude','longitude'])
print(df)

executionTime=(time.time()-startTime)
print('Files read in, in minutes: %.2f'%(executionTime/60))

# drop NaN values 
s1=df.shape[0]
df=df.dropna(axis=0,how='any')
s2=df.shape[0]
print('From '+str(s1)+' dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')
print(df)

# - - - - - - - - -
# print number of NaN values of each training set 
s1=df_eobs.shape[0]
df_eobs=df_eobs.dropna(axis=0,how='any')
s2=df_eobs.shape[0]
print('eobs From '+str(s1)+' dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')
s1=df_era5l.shape[0]
df_era5l=df_era5l.dropna(axis=0,how='any')
s2=df_era5l.shape[0]
print('era5l From '+str(s1)+' dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')
s1=df_clms.shape[0]
df_clms=df_clms.dropna(axis=0,how='any')
s2=df_clms.shape[0]
print('CLMS From '+str(s1)+' dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')
# - - - - - - - - -

df_eobs,df_era5l=[],[]
df_clms = []
df_eobs_clms = []

# - - - - - - - - -
# list of stations involved in training
#stations=df.pointID.drop_duplicates().to_list()
#print(stations)
#file = open(data_dir+'training_stations9790_SDfixed.txt','w')
#for sta in stations:
#	file.write(str(sta)+"\n")
#file.close()
#print(len(stations))
# - - - - - - - - -

# add day of year to predictors
df['utctime']=pd.to_datetime(df['utctime'])
df['dayOfYear'] = df['utctime'].dt.dayofyear

# Split to train and test by years, KFold best split (k=5)
test_y=[2009, 2016, 2018]
train_y=[2006, 2007, 2008, 2010, 2011, 2012, 2013, 2014, 2015, 2017, 2019, 2020]
print('test ',test_y,' train ',train_y)
train_stations,test_stations=pd.DataFrame(),pd.DataFrame()
for y in train_y:
        train_stations=pd.concat([train_stations,df[df['utctime'].dt.year == y]],ignore_index=True)
for y in test_y:
        test_stations=pd.concat([test_stations,df[df['utctime'].dt.year == y]],ignore_index=True)

# split data to predidctors (preds) and variable to be predicted (var)
# Note: era5L tp excluded since had many NaN values
preds=[
    'latitude','longitude',
    'slhf','sshf','ssrd','strd','str','ssr','sf',
    'laihv-00','lailv-00','sd-00','rsn-00',
    'stl1-00','swvl2-00','t2-00','td2-00','u10-00','v10-00',
    'ro','evap','swe_clms',
    'dayOfYear']
var=['SD'] 
preds_train=train_stations[preds] 
preds_test=test_stations[preds]
var_train=train_stations[var]
var_test=test_stations[var]

df=[]

### XGBoost
# Define model hyperparameters
# Values taken from Optuna hyperparameter optimization
nstm=897
lrte=0.10
max_depth=17
subsample=0.94
colsample_bytree=0.63
num_parallel_tree=9
al=0.37

# initialize and tune model
xgbr=xgb.XGBRegressor(
            objective= 'reg:squarederror',
            n_estimators=nstm,
            learning_rate=lrte,
            max_depth=max_depth,
            alpha=al,#gamma=0.01
            num_parallel_tree=num_parallel_tree,
            n_jobs=64,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric='rmse',
            random_state=99,
            early_stopping_rounds=50
            )

# train model 
eval_set=[(preds_test,var_test)]
xgbr.fit(
        preds_train,var_train,
        eval_set=eval_set)
print(xgbr)

# predict var and compare with test
var_pred=xgbr.predict(preds_test)
mse=mean_squared_error(var_test,var_pred)
mae=mean_absolute_error(var_test,var_pred)

# save model (REMEMBER TO CHANGE name for different models)
# 26.2.2024 run: mdl_SD_2006-2020_9790points-1.txt : 21 predictors, lat,lon and tp excluded
# 1.3.2024 run: mdl_SD_2006-2020_9790points-2.txt : 23 predictors, tp excluded
xgbr.save_model(mod_dir+'mdl_SD_2006-2020_9790points-3.txt')

print("RMSE: %.5f" % (mse**(1/2.0)))
print("MAE: %.5f" % (mae))
#print("RMSE: %.5f" % (mse**(1/2.0)*1000))
#print("MAE: %.5f" % (mae*1000))

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))
