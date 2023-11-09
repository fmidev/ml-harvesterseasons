import os, time, random, warnings,sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings("ignore")
### XGBoost for swi2 

startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/soilwater/' # training data
mod_dir='/home/ubuntu/data/ML/models/soilwater' # saved mdl
res_dir='/home/ubuntu/data/ML/results/soilwater'

'''
all cols 
utctime,swi2,evap,evap5d,evap15d,evap60d,evap100d,evapp,evapp5d,evapp15d,evapp60d,evapp100d,laihv-00,laihv-12,lailv-00,lailv-12,
ro,ro5d,ro15d,ro60d,ro100d,rsn-00,rsn-12,sd-00,sd-12,sf,skt-00,skt-12,slhf,sro,sro5d,sro15d,sro60d,sro100d,sshf,ssr,ssrd,
ssro,ssro5d,ssro15d,ssro60d,ssro100d,stl1-00,stl1-12,str,strd,swvl1-00,swvl1-12,swvl2-00,swvl2-12,swvl3-00,swvl3-12,swvl4-00,swvl4-12,
t2-00,t2-12,td2-00,td2-12,tp,tp5d,tp15d,tp60d,tp100d,u10-00,u10-12,v10-00,v10-12,
swi2clim,
lake_cover,cvh,cvl,lake_depth,land_cover,soiltype,urban_cover,tvh,tvl,
POINT_ID,TH_LAT,TH_LONG,DTM_height,DTM_slope,DTM_aspect,TCD,WAW,CorineLC,
clay_0-5cm,clay_100-200cm,clay_15-30cm,clay_30-60cm,clay_5-15cm,clay_60-100cm,
sand_0-5cm,sand_100-200cm,sand_15-30cm,sand_30-60cm,sand_5-15cm,sand_60-100cm,
silt_0-5cm,silt_100-200cm,silt_15-30cm,silt_30-60cm,silt_5-15cm,silt_60-100cm,
soc_0-5cm,soc_100-200cm,soc_15-30cm,soc_30-60cm,soc_5-15cm,soc_60-100cm
'''
### Read in 2D tabular training data
cols_own=['utctime','swi2','evap','evap15d',
'laihv-00','lailv-00',
'ro','ro15d','rsn-00','sd-00',
'slhf','sshf','ssr','ssrd',
'stl1-00','str','swvl2-00','t2-00','td2-00',
'tp','tp15d',
'swi2clim',
'lake_cover','cvh','cvl','lake_depth','land_cover','soiltype','urban_cover','tvh','tvl',
'TH_LAT','TH_LONG','DTM_height','DTM_slope','DTM_aspect',
'clay_0-5cm','clay_15-30cm','clay_5-15cm',
'sand_0-5cm','sand_15-30cm','sand_5-15cm',
'silt_0-5cm','silt_15-30cm','silt_5-15cm',
'soc_0-5cm','soc_15-30cm','soc_5-15cm',
]
fname=sys.argv[1] # training data csv filename
print(fname)
df=pd.read_csv(data_dir+fname,usecols=cols_own)

# add day of year to predictors
df['utctime']=pd.to_datetime(df['utctime'])
df['dayOfYear'] = df['utctime'].dt.dayofyear

# shift 24hsums data 1 day up: previous day sums to correct current dates (think if needed)
df[['tp','tp15d','ro','ro15d','evap','evap15d','slhf','sshf','ssrd','str','ssr']] = df[['tp','tp15d','ro','ro15d','evap','evap15d','slhf','sshf','ssrd','str','ssr']].shift(-1)
df = df[~(df['utctime'] == '2022-12-31')] # drop all last dates bc it is first date from next lucas point or nan

# drop negative swi2, NaN and -99999 values
s1=df.shape[0]
df=df.dropna(axis=0,how='any')
df=df[(df.DTM_slope !=-99999.000000) & (df.DTM_height !=-99999.000000) & (df.DTM_aspect !=-99999.000000)] 
df = df[df['swi2'] >= 0]
s2=df.shape[0]
print('From '+str(s1)+' dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')

# Split to train and test by years, KFold for best split (k=5)
test_y=[2019,2021]
train_y=[2015,2016,2017,2018,2020,2022]
print('test ',test_y,' train ',train_y)
train_stations,test_stations=pd.DataFrame(),pd.DataFrame()
for y in train_y:
        train_stations=pd.concat([train_stations,df[df['utctime'].dt.year == y]],ignore_index=True)
for y in test_y:
        test_stations=pd.concat([test_stations,df[df['utctime'].dt.year == y]],ignore_index=True)

# split data to predidctors (preds) and variable to be predicted (var)
preds=['evap','evap15d',
'laihv-00','lailv-00',
'ro','ro15d','rsn-00','sd-00',
'slhf','sshf','ssr','ssrd',
'stl1-00','str','swvl2-00','t2-00','td2-00',
'tp','tp15d',
'swi2clim',
'lake_cover','cvh','cvl','lake_depth','land_cover','soiltype','urban_cover','tvh','tvl',
'TH_LAT','TH_LONG','DTM_height','DTM_slope','DTM_aspect',
'clay_0-5cm','clay_15-30cm','clay_5-15cm',
'sand_0-5cm','sand_15-30cm','sand_5-15cm',
'silt_0-5cm','silt_15-30cm','silt_5-15cm',
'soc_0-5cm','soc_15-30cm','soc_5-15cm',
'dayOfYear'
]
var=['swi2']
preds_train=train_stations[preds] 
preds_test=test_stations[preds]
var_train=train_stations[var]
var_test=test_stations[var]
preds_train=preds_train.astype(float)

### XGBoost
# Define model hyperparameters (Optuna tuned)
nstm=645
lrte=0.067
max_depth=10
subsample=0.29
colsample_bytree=0.56
#colsample_bynode=1
num_parallel_tree=10
a=0.54

# initialize and tune model
xgbr=xgb.XGBRegressor(
            objective= 'reg:squarederror',#'count:poisson'
            n_estimators=nstm,
            learning_rate=lrte,
            max_depth=max_depth,
            alpha=a,#gamma=0.01
            num_parallel_tree=num_parallel_tree,
            n_jobs=64,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            #colsample_bynode=colsample_bynode,
            eval_metric='rmse', #'mae', 
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
xgbr.save_model(mod_dir+'/mdl_swi2_2015-2022_10000points-13.txt')

print("RMSE: %.5f" % (mse**(1/2.0)))
print("MAE: %.5f" % (mae))

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))