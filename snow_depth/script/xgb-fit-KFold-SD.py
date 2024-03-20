import os, time, datetime, random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold 
from sklearn.metrics import mean_squared_error, mean_absolute_error
### XGBoost with KFold
# note: need to be modified if want to use at desm
startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/snowdepth/' # training data
mod_dir='/home/ubuntu/data/ML/models/snowdepth/' # saved mdl
res_dir='/home/ubuntu/data/ML/results/snowdepth/'

 #'utctime','latitude','longitude','pointID'
cols_own=[
    'utctime','pointID','slhf','sshf','ssrd',
    'strd','str','ssr','sf','laihv-00','lailv-00','sd-00','rsn-00',
    'stl1-00','swvl2-00','t2-00','td2-00','u10-00','v10-00',
    'ro','evap','tp'
]

# CLMS columns: utctime, swe_clms, latitude, longitude, pointID
cols_clms=[
    'utctime','pointID','swe_clms'
]

'''
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
'''

# read in training data from era5l, CLMS and eobs
# note: eobs does not always have full years time series, unlike era5l
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
df_eobs=pd.read_csv(data_dir+feobs)
print(df_eobs)

df_eobs.rename(columns={'DATE':'utctime','STAID':'pointID','LAT': 'latitude', 'LON': 'longitude'}, inplace=True)

# merge SD eobs, clms and era5l data
df_eobs_clms=pd.merge(df_clms,df_eobs, on=['utctime','pointID'])
df=pd.merge(df_era5l,df_eobs_clms, on=['utctime','pointID'])
#df=pd.merge(df_era5l,df_eobs, on=['utctime','pointID','latitude','longitude'])

print(df)

# drop NaN values
s1=df.shape[0]
df=df.dropna(axis=0,how='any')
df=df[(df.HGHT !=-999)]  # is this needed? HGHT from eobs(eca&d) data
df = df[df['rsn-00'] !=99.999985]
s2=df.shape[0]
print('From '+str(s1)+' dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')
df_eobs,df_era5l=[],[]
df_clms = []
df_eobs_clms = []

# add day of year to predictors
df['utctime']=pd.to_datetime(df['utctime'])
df['dayOfYear'] = df['utctime'].dt.dayofyear

# Read predictor (preds) and predictand (var) data
# Note excluded here: 'latitude','longitude',pointID
preds=['utctime',
    'slhf','sshf','ssrd','strd','str','ssr','sf',
    'laihv-00','lailv-00','sd-00','rsn-00',
    'stl1-00','swvl2-00','t2-00','td2-00','u10-00','v10-00',
    'ro','evap','tp','swe_clms',
    'dayOfYear']
var=['utctime','SD'] # variable to be predicted

# Define XGBRegressor model parameters
nstm=500
lrte=0.1 
max_depth=7
subsample=0.7
colsample_bytree=0.7
colsample_bynode=1
num_parallel_tree=10
eval_met='rmse'

# KFold cross-validation; splitting to train/test sets by years
y1,y2=2006,2020
allyears=np.arange(y1,y2+1).astype(int)
#allyears=np.arange(2006,2021).astype(int)

kf=KFold(5,shuffle=True,random_state=20)
fold=0
mdls=[]
#print(df)
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
    #eval_met='rmse'
        
    # Define the model 
    xgbr=xgb.XGBRegressor(
        objective= 'reg:squarederror', # 'count:poisson'
        n_estimators=nstm,
        learning_rate=lrte,
        max_depth=max_depth,
        alpha=0.01, # gamma=0.01
        num_parallel_tree=num_parallel_tree,
        n_jobs=64, #n_jobs=24,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        colsample_bynode=colsample_bynode,
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

    print("Fold: %s RMSE: %.2f" % (fold,mse**(1/2.0)))
    print('Train: ',train_years,'Test: ',test_years)
    mdls.append(fitted_mdl)

# Save GB models
for i,mdl in enumerate(mdls):
    mdl.save_model(mod_dir+'Kfold_mdl_SD_2006-2020_9790points_nro'+str(i+1)+'.txt')

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))
