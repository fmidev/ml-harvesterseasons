import os, time, random, warnings,sys
import pandas as pd
import numpy as np
#from lightgbm import LGBMRegressor as lgb
import lightgbm as lgb
from lightgbm import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
np.random.seed(42)
warnings.filterwarnings("ignore")
### lgb, for swi2

startTime=time.time()


data_dir='/home/ubuntu/data/ML/training-data/soiltemp/' # training data
mod_dir='/home/ubuntu/data/ML/models/soiltemp' # saved mdl
optuna_dir='/home/ubuntu/data/ML/'
plot_dir='/home/ubuntu/ml-harvesterseasons/soil_temperature/plots/'

### Read in 2D tabular training data

cols_own=["utctime","slhf","sshf",
        "ssrd","strd","str","ssr","skt","skt-00",
        "sktn","laihv-00",
        "lailv-00",
        "sd-00","rsn-00",
        "stl1-00","stl2-00","swvl2-00",
        "t2-00","td2-00","u10-00",
        "v10-00","ro","evapp","DTM_height","DTM_slope",
        "DTM_aspect","clim_ts_value"
]

fname='train_data_latest_additional.csv' # training data csv filename
print(fname)


df=pd.read_csv(data_dir+fname,usecols=cols_own)

# remove outliers, take value between -42 and 42
df =df.loc[(df["clim_ts_value"]>= -42) & (df["clim_ts_value"]<= 42)]

# add day of year to predictors
df['utctime']=pd.to_datetime(df['utctime'])
df['dayOfYear'] = df['utctime'].dt.dayofyear
print(df)


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

preds=["slhf","sshf",
        "ssrd","strd","str","ssr","skt","skt-00",
        "sktn","laihv-00",
        "lailv-00",
        "sd-00","rsn-00",
        "stl1-00","stl2-00","swvl2-00",
        "t2-00","td2-00","u10-00",
        "v10-00","ro","evapp","DTM_height","DTM_slope",
        "DTM_aspect",'dayOfYear'
        ]

var=['clim_ts_value']
preds_train=train_stations[preds] 
preds_test=test_stations[preds]
var_train=train_stations[var]
var_test=test_stations[var]
preds_train=preds_train.astype(float)

### LGMB
# Define model hyperparameters 
nstm=662
num_leaves=44
lrte=0.4227020551271515
feature_fraction=0.7316979668699045
reg_alpha=0.6487051405403162
max_depth=4
subsample=0.551480268886276
# colsample_bytree=0.7
nthread=9
n_boost_round= 1000
verbose=100

def rmse(preds, train_data):
    labels = train_data.get_label()
    MAE = mean_absolute_error(labels, preds)
    RMSE = math.sqrt(MAE)
    return 'RMSE', RMSE, False

def mae(preds, train_data):
    labels = train_data.get_label()
    MAE = mean_absolute_error(labels, preds)
    return 'MAE', MAE, False

# initialize and tune model
lgb_params = {'objective' :'regression',
              'num_leaves': num_leaves,
              'n_estimators':nstm,
              'learning_rate': lrte,
              'feature_fraction': feature_fraction,
              'max_depth': max_depth,
              'reg_alpha':reg_alpha,
              'n_jobs':64,
              'random_state':99,
              'subsample':subsample,
              'verbose': verbose,
              'num_boost_round': n_boost_round,
              'early_stopping_rounds': 50,
              'nthread':nthread}
# train model 
lgbtrain = lgb.Dataset(preds_train, label=var_train)

lgbval = lgb.Dataset(preds_test, label=var_test)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  #num_boost_round=lgb_params['num_boost_round'],
                  #early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=mae,
                  #verbose_eval=100
                  )

print(model)

# predict var and compare with test
var_pred=model.predict(preds_test,num_iteration=model.best_iteration)
# mse=mean_squared_error(var_test,var_pred)
mae=mean_absolute_error(var_test,var_pred)
#varOut['RR-pred']=var_pred.tolist()
#varOut.to_csv(res_dir+'predicted_results_194sta.csv')

# save model 
model.save_model(mod_dir+'/lgb_soiltemp_latest_mae.json',num_iteration=model.best_iteration)

# print("RMSE: %.5f" % (np.sqrt(mse)))
print("MAE: %.5f" % (mae))

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))

# plotting feature importance
plt.rcParams["figure.figsize"] = (6, 10)
plot_importance(model,grid=False)
plt.savefig(plot_dir+'soil_temperature_importance_lgb_mae.jpg',bbox_inches='tight')