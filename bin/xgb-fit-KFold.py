import os, time, datetime, random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold 
from sklearn.metrics import mean_squared_error, mean_absolute_error
### XGBoost with KFold
# note: need to be modified if want to use at desm
startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/precipitation-harmonia/' # training data
mod_dir='/home/ubuntu/data/ML/models/precipitation-harmonia/' # saved mdl
res_dir='/home/ubuntu/data/ML/results/precipitation-harmonia/'

cols_own=[
    'utctime','latitude','longitude','pointID',
    'anor','evap','kx-00','lsm','mn2t-00','msl-00','mx2t-00',
    'q500-00','q700-00','q850-00','ro','rsn-00','sd-00','sdor',
    'sf','skt-00','slhf','slor','sshf','ssr','ssrd','strd',#'sst-00',
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
df_eobs=pd.read_csv(data_dir+feobs)
print(df_eobs)
df_eobs.rename(columns={'DATE':'utctime','STAID':'pointID','LAT': 'latitude', 'LON': 'longitude'}, inplace=True)
df=pd.merge(df_era5,df_eobs, on=['utctime','pointID','latitude','longitude'])
print(df)

# drop NaN values
s1=df.shape[0]
df=df.dropna(axis=0,how='any')
s2=df.shape[0]
print('From '+str(s1)+' dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')
df_eobs,df_era5=[],[]

# add day of year to predictors
df['utctime']=pd.to_datetime(df['utctime'])
df['dayOfYear'] = df['utctime'].dt.dayofyear

# Read predictor (preds) and predictand (var) data
preds=['utctime','latitude','longitude',#'HGHT','SLOPE','ASPECT',
    #'TPI','TRI','DTW','DTR','DTL','DTO','FCH',
    'anor','evap','kx-00','lsm','mn2t-00','msl-00','mx2t-00',
    'q500-00','q700-00','q850-00','ro','rsn-00','sd-00','sdor',
    'sf','skt-00','slhf','slor','sshf','ssr','ssrd','strd',#'sst-00',
    't2m-00','t500-00','t700-00','t850-00','tcc-00','td2m-00','tp',
    'tsr','ttr','u10-00','u500-00','u700-00','u850-00','v10-00',
    'v500-00',#'v700-00',
    'v850-00','z',#'z500-00',
    'z700-00','z850-00',
    'dayOfYear']
var=['utctime','RR'] # variable to be predicted

# Define XGBRegressor model parameters
nstm=500
lrte=0.1 
max_depth=7
subsample=0.7
colsample_bytree=0.7
colsample_bynode=1
num_parallel_tree=10

# KFold cross-validation; splitting to train/test sets by years
#allyears=np.arange(1995,2016).astype(int)
allyears=np.arange(2000,2021).astype(int)

kf=KFold(5,shuffle=True,random_state=20)
fold=0
mdls=[]
print(df)
for train_idx, test_idx in kf.split(allyears):
        fold+=1
        train_years=allyears[train_idx]
        test_years=allyears[test_idx]
        train_idx=np.isin(df['utctime'].dt.year,train_years)
        test_idx=np.isin(df['utctime'].dt.year,test_years)
        train_set=df[train_idx].reset_index(drop=True)
        test_set=df[test_idx].reset_index(drop=True)
      
        # Split to predictors and target variable
        preds_train=train_set[preds].drop(columns=['utctime'])
        preds_test=test_set[preds].drop(columns=['utctime'])
        var_train=train_set[var].drop(columns=['utctime'])
        var_test=test_set[var].drop(columns=['utctime'])

        # Define the model without...
        eval_met='rmse'
        
        xgbr=xgb.XGBRegressor(
                objective= 'reg:squarederror',
                n_estimators=nstm,
                learning_rate=lrte,
                max_depth=max_depth,
                gamma=0.01, #alpha=0.01
                num_parallel_tree=num_parallel_tree,
                n_jobs=64,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                #colsample_bynode=colsample_bynode,
                random_state=99,
                eval_metric=eval_met,
                early_stopping_rounds=50
                )
        
        # Train the model
        eval_set=[(preds_test,var_test)]
        fitted_mdl=xgbr.fit(
                preds_train,var_train,
                eval_set=eval_set,
                verbose=False #True
        )

        # Predict var and compare with test
        var_pred=fitted_mdl.predict(preds_test)
        mse=mean_squared_error(var_test,var_pred)
        print("Fold: %s RMSE: %.2f" % (fold,mse**(1/2.0)*1000))
        print('Train: ',train_years,'Test: ',test_years)
        mdls.append(fitted_mdl)

# Save GB models
for i,mdl in enumerate(mdls):
        mdl.save_model(mod_dir+'mdl_RR_KFold_2000-2020_nro'+str(i+1)+'.txt')

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))