import os, time, datetime, random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold 
from sklearn.metrics import mean_squared_error, mean_absolute_error
#from Raahe_101785 import * # import bbox, harbor, FMISID, cols_own, fname, test_y, train_y, mdl_name
#from Vuosaari_151028 import *
#from Rauma_101061 import *
#from Malaga_000231 import *
#from Malaga_ece3 import *
#from Sitia_023330_ece3 import *
#from Cadiz_000415_ece3 import *
#from Ploumanach_011245_ece3 import *
#from Vuosaari_151028_cordex_cnrm_knmi import *
#from Aktio_023317_cordex_ncc_smhi import *
from Bremerhaven_004885_ece3 import *

### XGBoost with KFold for OCEANIDS
startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/OCEANIDS/' # training data
mod_dir='/home/ubuntu/data/ML/models/OCEANIDS' # saved mdl
res_dir='/home/ubuntu/data/ML/results/OCEANIDS'

### Read in 2D tabular training data
print(harbor)
print(pred)
print(fname)
#df=pd.read_csv(data_dir+fname,usecols=cols_own)
df=pd.read_csv(data_dir+fname)

# drop NaN values and columns
df=df.dropna(axis=1, how='all') 
s1=df.shape[0]
df=df.dropna(axis=0,how='any')
s2=df.shape[0]
print('From '+str(s1)+' rows dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')
df['utctime']= pd.to_datetime(df['utctime'])
print(df)
headers=list(df) # list column headers

# Read predictor (preds) and predictand (var) data
var=df[[pred]]
preds=df[headers].drop(['utctime',pred], axis=1)
var_headers=list(var) 
preds_headers=list(preds)

# Define hyperparameters for XGBoost
nstm=645
lrte=0.067
max_depth=10
subsample=0.29
colsample_bytree=0.56
#colsample_bynode=1
num_parallel_tree=10
eval_met='rmse'
a=0.54

# KFold cross-validation; splitting to train/test sets by years
y1,y2=int(starty),int(endy)
print(y1,y2)
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
        preds_train=train_set[preds_headers]
        preds_test=test_set[preds_headers]
        var_train=train_set[var_headers]
        var_test=test_set[var_headers]
        
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
        
        print("Fold: %s RMSE: %.2f" % (fold,mse**(1/2.0)))
        print('Train: ',train_years,'Test: ',test_years)
        mdls.append(fitted_mdl)

# Save XGB models
for i,mdl in enumerate(mdls):
        mdl.save_model(mod_dir+'KFold-mdl_'+pred+'_'+harbor+'_'+starty+'-'+endy+'_'+str(i+1)+'.json')

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))