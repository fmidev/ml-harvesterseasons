import os, time, random, warnings,sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings("ignore")
### XGBoost, give target parameter as cmd (soilwater or snowdepth or soiltemp) 

startTime=time.time()

target=sys.argv[1] # soilwater / snowdepth / soiltemp

data_dir='/home/ubuntu/data/ML/training-data/'+par+'/' # training data
mod_dir='/home/ubuntu/data/ML/models/'+par+'/' # saved mdl
res_dir='/home/ubuntu/data/ML/results/'+par+'/' # other results, Fscore f.ex.

### Read in 2D tabular training data (predictors and predictand=target)
cols_own=['utctime',
        'POINT_ID','TH_LAT','TH_LONG','CPRN_LC','LC1_LABEL','DTM_height','DTM_slope','DTM_aspect','TCD','WAW','CorineLC', #lucas+attr
        target # predictand
]

#tp=pd.read_csv('/lustre/tmp/strahlen/mlbias/data/era5+_eobs_dtw_1995-2015_all6173_RRweight.csv', iterator=True,chunksize=1000,usecols=cols_own)
#df = pd.concat(tp, ignore_index=True) # read in chunks too large csv
fname='era5-SL-PL-terrain-EOBS_1995-2020_all194.csv' # training data file
print(fname)
df=pd.read_csv(data_dir+fname,usecols=cols_own)
    
df['utctime']=pd.to_datetime(df['utctime'])

# Split to train and test by years, choose years
#test_y=[2000,2003,2005,2015]
test_y=[2003, 2009, 2011, 2015]
#train_y=[1995,1996,1997,1998,1999,2001,2002,2004,2006,2007,2008,2009,2010,2011,2012,2013,2014]
train_y=[2000, 2001, 2002, 2004, 2005, 2006, 2007, 2008, 2010, 2012, 2013, 2014, 2016, 2017, 2018, 2019, 2020, 2021]
print('test ',test_y,' train ',train_y)
train_stations,test_stations=pd.DataFrame(),pd.DataFrame()
for y in train_y:
        train_stations=pd.concat([train_stations,df[df['utctime'].dt.year == y]],ignore_index=True)
for y in test_y:
        test_stations=pd.concat([test_stations,df[df['utctime'].dt.year == y]],ignore_index=True)

# drop rows with any NaN (XGBoost dies) 
testnans1=test_stations.shape[0] # just checking how many rows were dropped due to NaNs
trainnans1=train_stations.shape[0]
test_stations=test_stations.dropna(axis=0,how='any')
train_stations=train_stations.dropna(axis=0,how='any')
testnans2=test_stations.shape[0]
trainnans2=train_stations.shape[0]
varOut=test_stations[['utctime','STAID','RR','tp-m']].copy()
print('test set dropped '+str(testnans1-testnans2)+' rows')
print('train set dropped '+str(trainnans1-trainnans2)+' rows')
train_stations=train_stations.drop(columns=['utctime','STAID','staid'])
test_stations=test_stations.drop(columns=['utctime','STAID','staid'])

# split data to predidctors (preds) and variable to be predicted (var)
preds=[
]
var=[target] 
preds_train=train_stations[preds] 
preds_test=test_stations[preds]
var_train=train_stations[var]
var_test=test_stations[var]
print(preds_train)    

### XGBoost
# Define model hyperparameters 
nstm=500
lrte=0.1
max_depth=7
subsample=0.7
colsample_bytree=0.7
colsample_bynode=1
num_parallel_tree=10

# initialize and tune model
xgbr=xgb.XGBRegressor(
            objective= 'count:poisson', # 'reg:squarederror'
            n_estimators=nstm,
            learning_rate=lrte,
            max_depth=max_depth,
            alpha=0.01,#gamma=0.01
            num_parallel_tree=num_parallel_tree,
            n_jobs=24,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bynode=colsample_bynode,
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
varOut['RR-pred']=var_pred.tolist()
varOut.to_csv(res_dir+'predicted_results_194sta.csv')

# save model 
xgbr.save_model(mod_dir+'mdl_194sta_2000-2020.txt')

print("RMSE: %.5f" % (mse**(1/2.0)))

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))