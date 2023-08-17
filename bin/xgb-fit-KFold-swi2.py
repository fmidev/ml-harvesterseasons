import os, time, datetime, random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold 
from sklearn.metrics import mean_squared_error, mean_absolute_error
### XGBoost with KFold for swi2
startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/soilwater/' # training data
mod_dir='/home/ubuntu/data/ML/models/soilwater/' # saved mdl
res_dir='/home/ubuntu/data/ML/results/soilwater/'

cols_own=['utctime','swi2','evap','evap5d','evap15d','evap60d','evap100d','evapp','evapp5d','evapp15d','evapp60d','evapp100d',
'laihv-00','laihv-12','lailv-00','lailv-12','ro','ro5d','ro15d','ro60d','ro100d','rsn-00','rsn-12','sd-00','sd-12','sf',
'skt-00','skt-12','slhf','sro','sro5d','sro15d','sro60d','sro100d','sshf','ssr','ssrd','ssro','ssro5d','ssro15d','ssro60d',
'ssro100d','stl1-00','stl1-12','str','strd','swvl1-00','swvl1-12','swvl2-00','swvl2-12','swvl3-00','swvl3-12','swvl4-00',
'swvl4-12','t2-00','t2-12','td2-00','td2-12','tp','tp5d','tp15d','tp60d','tp100d','u10-00','u10-12','v10-00','v10-12',
'TH_LAT','TH_LONG','DTM_height','DTM_slope'
]
#fname='swi2_training_236lucasPoints_2015-2022_all_FIXED.csv'
fname='swi2_training_4108lucasPoints_2015-2022_all.csv'
print(fname)
df=pd.read_csv(data_dir+fname,usecols=cols_own)

df['utctime']=pd.to_datetime(df['utctime'])
df['dayOfYear'] = df['utctime'].dt.dayofyear
print(df)

# drop negative swi2, NaN and -99999 values 
s1=df.shape[0]
df=df.dropna(axis=0,how='any')
df=df[(df.DTM_slope !=-99999.000000) & (df.DTM_height !=-99999.000000)] 
df = df[df['swi2'] >= 0]
s2=df.shape[0]
print('From '+str(s1)+' dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')
print(df)

# Read predictor (preds) and predictand (var) data
preds=['utctime','evap','evap15d','evapp','evapp15d',
'laihv-00','laihv-12','lailv-00','lailv-12','ro','ro15d','rsn-00','rsn-12','sd-00','sd-12','sf',
'skt-00','skt-12','slhf','sro','sro15d','sshf','ssr','ssrd','ssro','ssro15d',
'stl1-00','stl1-12','str','strd','swvl2-00','swvl2-12','t2-00','t2-12','td2-00',
'td2-12','tp','tp15d','u10-00','u10-12','v10-00','v10-12',
'TH_LAT','TH_LONG','DTM_height','DTM_slope',
'dayOfYear'
]
var=['utctime','swi2']

# Define XGBRegressor model parameters (from gridSearchCV)
nstm=500
lrte=0.1 
max_depth=7
subsample=0.7
colsample_bytree=0.7
colsample_bynode=1
num_parallel_tree=10
eval_met='rmse'

# KFold cross-validation; splitting to train/test sets by years
y1,y2=2015,2022
allyears=np.arange(y1,y2+1).astype(int)

kf=KFold(5,shuffle=True,random_state=20)
fold=0
mdls=[]
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
        
        # Define the model        
        xgbr=xgb.XGBRegressor(
                objective='reg:squarederror', # 'count:poisson'
                n_estimators=nstm,
                learning_rate=lrte,
                max_depth=max_depth,
                alpha=0.01, #gamma=0.01
                num_parallel_tree=num_parallel_tree,
                n_jobs=24,
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
        mdl.save_model(mod_dir+'KFold-mdl_swi2_2015-2022_4108points_'+str(i+1)+'.txt')

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))