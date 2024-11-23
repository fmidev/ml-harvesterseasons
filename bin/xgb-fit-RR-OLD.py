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
# v700 and z500 had many NaN values, training set dropped ~80% rows -> leave out
# sst nan when not sea surface -> leave out
cols_own=[
    'utctime','latitude','longitude','pointID',
    'anor','evap','kx-00','lsm','mn2t-00','msl-00','mx2t-00',
    'q500-00','q700-00','q850-00','ro','rsn-00','sd-00','sdor',
    'sf',#'skt-00',
    'slhf','slor','sshf','ssr','ssrd','strd',#'sst-00',
    't2m-00','t500-00','t700-00','t850-00','tcc-00','td2m-00','tp',
    'tsr','ttr','u10-00','u500-00','u700-00','u850-00','v10-00',
    'v500-00',#'v700-00',
    'v850-00','z',#'z500-00',
    'z700-00','z850-00'
]

# read in training data from era5 and eobs
# note: eobs does not always have full 20 years time series, unlike era5
fera5='RR_era5_training_6766stations_2000-2020_all.csv.gz' # era5 training data
feobs='RR_eobs_training_6766stations_2000-2020_all.csv.gz' # eobs training data
print(fera5)
print(feobs)
df_era5=pd.read_csv(data_dir+fera5,usecols=cols_own)
print(df_era5)
eobs_cols=['STAID','DATE','RR','HGHT']
df_eobs=pd.read_csv(data_dir+feobs,usecols=eobs_cols)
print(df_eobs)

df_eobs.rename(columns={'DATE':'utctime','STAID':'pointID'}, inplace=True)#,'LAT': 'latitude', 'LON': 'longitude'
df=pd.merge(df_era5,df_eobs, on=['utctime','pointID'])#,'latitude','longitude'])
print(df)

executionTime=(time.time()-startTime)
print('Files read in, in minutes: %.2f'%(executionTime/60))

# drop NaN values and negative RR 
s1=df.shape[0]
df=df.replace(-99999.0,np.nan)
df=df.replace(-9999.0,np.nan)
df=df.dropna(axis=0,how='any')
df = df[df['RR'] >= 0]
s2=df.shape[0]
print('From '+str(s1)+' dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')
print(df)

s1=df_eobs.shape[0]
df_eobs=df_eobs.dropna(axis=0,how='any')
s2=df_eobs.shape[0]
print('eobs From '+str(s1)+' dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')
s1=df_era5.shape[0]
df_era5=df_era5.dropna(axis=0,how='any')
s2=df_era5.shape[0]
print('era5 From '+str(s1)+' dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')

df_eobs,df_era5=[],[]

stations=df.pointID.drop_duplicates().to_list()
#print(stations)
file = open(data_dir+'training_stations_negRRfixed.txt','w')
for sta in stations:
	file.write(str(sta)+"\n")
file.close()
print(len(stations))

# add day of year to predictors
df['utctime']=pd.to_datetime(df['utctime'])
df['dayOfYear'] = df['utctime'].dt.dayofyear

# Weight E-OBS station rainfall observations (rain > 12 mm: x1.4, other x1.1)
df.loc[ df['RR'] >= 0.012, 'RR'] = df['RR']*1.4 
df.loc[ df['RR'] < 0.012, 'RR'] = df['RR']*1.1

# Split to train and test by years, KFold best split (k=5)
test_y=[2003, 2009, 2011, 2015]
train_y=[2000, 2001, 2002, 2004, 2005, 2006, 2007, 2008, 2010, 2012, 2013, 2014, 2016, 2017, 2018, 2019, 2020]
print('test ',test_y,' train ',train_y)
train_stations,test_stations=pd.DataFrame(),pd.DataFrame()
for y in train_y:
        train_stations=pd.concat([train_stations,df[df['utctime'].dt.year == y]],ignore_index=True)
for y in test_y:
        test_stations=pd.concat([test_stations,df[df['utctime'].dt.year == y]],ignore_index=True)

# split data to predidctors (preds) and variable to be predicted (var)
preds=[
    'latitude','longitude',
    #'HGHT','SLOPE','ASPECT','TPI','TRI','DTW','DTR','DTL','DTO','FCH',
    'anor','evap','kx-00','lsm','mn2t-00','msl-00','mx2t-00',
    'q500-00','q700-00','q850-00','ro','rsn-00','sd-00','sdor',
    'sf',#'skt-00',
    'slhf','slor','sshf','ssr','ssrd','strd',#'sst-00',
    't2m-00','t500-00','t700-00','t850-00','tcc-00','td2m-00','tp',
    'tsr','ttr','u10-00','u500-00','u700-00','u850-00','v10-00',
    'v500-00',#'v700-00',
    'v850-00','z',#'z500-00',
    'z700-00','z850-00',
    'dayOfYear'
]
var=['RR'] 
preds_train=train_stations[preds] 
preds_test=test_stations[preds]
var_train=train_stations[var]
var_test=test_stations[var]

df=[]

### XGBoost
# Define model hyperparameters 
nstm=857
lrte=0.044
max_depth=9
subsample=0.96
colsample_bytree=0.37
num_parallel_tree=2
al=0.72

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

# save model 
#xgbr.save_model(mod_dir+'mdl_RRweight_5441sta_2000-2020-5.txt')

print("RMSE: %.5f" % (mse**(1/2.0)*1000))
print("MAE: %.5f" % (mae*1000))

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))