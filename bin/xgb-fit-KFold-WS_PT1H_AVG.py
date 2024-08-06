import os, time, datetime, random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold 
from sklearn.metrics import mean_squared_error, mean_absolute_error
### XGBoost with KFold for OCEANIDS
startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/OCEANIDS/' # training data
mod_dir='/home/ubuntu/data/ML/models/OCEANIDS' # saved mdl
res_dir='/home/ubuntu/data/ML/results/OCEANIDS'

'''
all cols 
utctime,latitude,longitude,FMISID,WS_PT1H_AVG,WG_PT1H_MAX,dayOfYear,hour,
utctime,lat-1,lon-1,lat-2,lon-2,lat-3,lon-3,lat-4,lon-4,e-1,e-2,e-3,e-4,
ewss-1,ewss-2,ewss-3,ewss-4,fg10-1,fg10-2,fg10-3,fg10-4,lsm-1,lsm-2,lsm-3,lsm-4,
msl-1,msl-2,msl-3,msl-4,nsss-1,nsss-2,nsss-3,nsss-4,slhf-1,slhf-2,slhf-3,slhf-4,
sshf-1,sshf-2,sshf-3,sshf-4,ssr-1,ssr-2,ssr-3,ssr-4,ssrd-1,ssrd-2,ssrd-3,ssrd-4,
str-1,str-2,str-3,str-4,strd-1,strd-2,strd-3,strd-4,t2-1,t2-2,t2-3,t2-4,
tcc-1,tcc-2,tcc-3,tcc-4,td2-1,td2-2,td2-3,td2-4,tlwc-1,tlwc-2,tlwc-3,tlwc-4,
tp-1,tp-2,tp-3,tp-4,tsea-1,tsea-2,tsea-3,tsea-4,u10-1,u10-2,u10-3,u10-4,
v10-1,v10-2,v10-3,v10-4
'''
### Read in 2D tabular training data
cols_own=['utctime','WS_PT1H_AVG','dayOfYear','hour',
#'lat-1','lon-1','lat-2','lon-2','lat-3','lon-3','lat-4','lon-4',
'e-1','e-2','e-3','e-4','ewss-1','ewss-2','ewss-3','ewss-4',
'fg10-1','fg10-2','fg10-3','fg10-4','lsm-1','lsm-2','lsm-3','lsm-4',
'msl-1','msl-2','msl-3','msl-4','nsss-1','nsss-2','nsss-3','nsss-4',
'slhf-1','slhf-2','slhf-3','slhf-4','sshf-1','sshf-2','sshf-3','sshf-4',
'ssr-1','ssr-2','ssr-3','ssr-4','ssrd-1','ssrd-2','ssrd-3','ssrd-4',
'str-1','str-2','str-3','str-4','strd-1','strd-2','strd-3','strd-4',
't2-1','t2-2','t2-3','t2-4','tcc-1','tcc-2','tcc-3','tcc-4',
'td2-1','td2-2','td2-3','td2-4','tlwc-1','tlwc-2','tlwc-3','tlwc-4',
'tp-1','tp-2','tp-3','tp-4','tsea-1','tsea-2','tsea-3','tsea-4',
'u10-1','u10-2','u10-3','u10-4','v10-1','v10-2','v10-3','v10-4'
]
fname = 'training_data_oceanids_Vuosaari_2013-2023.csv' # training input data file
print(fname)
df=pd.read_csv(data_dir+fname,usecols=cols_own)

# drop NaN values and columns
df=df.dropna(axis=1, how='all') 
s1=df.shape[0]
df=df.dropna(axis=0,how='any')
s2=df.shape[0]
print('From '+str(s1)+' rows dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')
df['utctime']= pd.to_datetime(df['utctime'])
#print(df)
headers=list(df) # list column headers

# Read predictor (preds) and predictand (var) data
var=df[['WS_PT1H_AVG']]
preds=df[headers].drop(['utctime','WS_PT1H_AVG'], axis=1)
var_headers=list(var) 
preds_headers=list(preds)

# Define hyperparameters for XGBoost
nstm=500
lrte=0.1 
max_depth=7
subsample=0.7
colsample_bytree=0.7
colsample_bynode=1
num_parallel_tree=10
eval_met='rmse'

# KFold cross-validation; splitting to train/test sets by years
y1,y2=2013,2023
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

# Save XGB models
for i,mdl in enumerate(mdls):
        mdl.save_model(mod_dir+'KFold-mdl_OCEANIDS_2013-2023_'+str(i+1)+'.json')

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))