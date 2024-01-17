import os, time, random, warnings,sys
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings("ignore")
np.random.seed(42)

startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/soiltemp/' # training data
mod_dir='/home/ubuntu/data/ML/models/soiltemp' # saved mdl
optuna_dir='/home/ubuntu/data/ML/'
plot_dir='/home/ubuntu/ml-harvesterseasons/soil_temperature/plots/'


### Read in 2D tabular training data
cols_own=["utctime",'slhf','sshf','ssrd','strd','str','ssr','sktn','sktd-12',
          'sf','laihv-00','laihv-12','lailv-00','lailv-12','sd-00','sd-12',
          'rsn-00','rsn-12','stl1-00','stl1-12','swvl2-00','swvl2-12','t2-00',
          't2-12','td2-00','td2-12','u10-00','u10-12','v10-00','v10-12','ro5d',
          'sro5d','ssro5d','evapp5d','tp5d','Forest_T_Min','Forest_T_Max','Forest_T_Mean',
          'longitude','latitude','TCD','WAW','DTM_height','DTM_slope','DTM_aspect','sand_0_5cm_mean',
          'sand_5_15cm_mean','silt_0_5cm_mean','silt_5_15cm_mean','clay_0_5cm_mean','clay_5_15cm_mean',
          'soc_0_5cm_mean','soc_5_15cm_mean','meanT_warmestQ_5_15cm','meanT_warmestQ_0_5cm',
          'mean_diurnal_0_5cm',"clim_ts_value"
]
fname='train_data.csv' # training data csv filename
print(fname)
df=pd.read_csv(data_dir+fname,usecols=cols_own)

# remove outliers, take value between -42 and 42
df =df.loc[(df["clim_ts_value"]>= -42) & (df["clim_ts_value"]<= 42)]
        

# add day of year to predictors
df['utctime']=pd.to_datetime(df['utctime'])
df['dayOfYear'] = df['utctime'].dt.dayofyear


df=df.fillna(-999)

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
preds=['slhf','sshf','ssrd','strd','str','ssr','sktn','sktd-12',
        'sf','laihv-00','laihv-12','lailv-00','lailv-12','sd-00','sd-12',
        'rsn-00','rsn-12','stl1-00','stl1-12','swvl2-00','swvl2-12','t2-00',
        't2-12','td2-00','td2-12','u10-00','u10-12','v10-00','v10-12','ro5d',
        'sro5d','ssro5d','evapp5d','tp5d','Forest_T_Min','Forest_T_Max','Forest_T_Mean',
        'longitude','latitude','TCD','WAW','DTM_height','DTM_slope','DTM_aspect','sand_0_5cm_mean',
        'sand_5_15cm_mean','silt_0_5cm_mean','silt_5_15cm_mean','clay_0_5cm_mean','clay_5_15cm_mean',
        'meanT_warmestQ_5_15cm','meanT_warmestQ_0_5cm',
        'mean_diurnal_0_5cm','dayOfYear'
]
var=['clim_ts_value']
preds_train=train_stations[preds] 
preds_test=test_stations[preds]
var_train=train_stations[var]
var_test=test_stations[var]
preds_train=preds_train.astype(float)

### XGBoost
# Define model hyperparameters (Optuna tuned)
nstm=490
lrte=0.5340796435726778
max_depth=7
subsample=0.8838908494448574
colsample_bytree=0.1840117026068701
#colsample_bynode=1
num_parallel_tree=9
a=0.13545148279394106

# initialize and tune model
xgbr=xgb.XGBRegressor(
            objective= 'reg:squarederror',#'count:poisson'
            n_estimators=nstm,
            learning_rate=lrte,
            max_depth=max_depth,
            alpha=a,
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
# mae=mean_absolute_error(var_test,var_pred)
now = datetime.now()
date_time = now.strftime("%m%d%Y%H%M%S")


print("---Saving model---")
xgbr.save_model(f"{mod_dir}/xgbmodel_soiltemp_{date_time}.json")

print("RMSE: %.5f" % (np.sqrt(mse)))
# print("MAE: %.5f" % (mae))

plt.rcParams["figure.figsize"] = (18, 14)
plot_importance(xgbr,grid=False)
plt.savefig(plot_dir+'soil_temperature_importance.jpg',bbox_inches='tight'
            )

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))