import os,optuna,time,random,warnings
import sklearn.datasets
import sklearn.metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import lightgbm as lgb
import math
from sklearn.model_selection import GridSearchCV
import numpy as np
warnings.filterwarnings("ignore")
### XGBoost with Optuna hyperparameter tuning for swi2 ML model
# note: does not save trained mdl
startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/soiltemp/' # training data
# mod_dir='/home/ubuntu/data/ML/models/soiltemp' # saved mdl
# res_dir='/home/ubuntu/data/ML/results/soiltemp'
optuna_dir='/home/ubuntu/data/ML/' # optuna storage

os.chdir(optuna_dir)
print(os.getcwd())

def rmse(preds, train_data):
    labels = train_data.get_label()
    MSE = mean_squared_error(labels, preds)
    RMSE = math.sqrt(MSE)
    return 'RMSE', RMSE, False
### optuna objective & xgboost
def objective(trial):
    # hyperparameters
    lgb_params = {'objective' :'regression',
              'num_leaves': trial.suggest_int("num_leaves", 2, 50),
              'n_estimators':trial.suggest_int("n_estimators",50,1000),
              'learning_rate': trial.suggest_float("learning_rate",0.01,0.7),
              'feature_fraction': trial.suggest_float("feature_fraction",0.01,1.0),
              'max_depth': trial.suggest_int("max_depth",3,18),
              'reg_alpha':trial.suggest_float("reg_alpha", 0.000000001, 1.0),
              'n_jobs':64,
              'random_state':99,
              'subsample':trial.suggest_float("subsample",0.01,1),
              'verbose': 100,
              'num_boost_round': 10000,
              'early_stopping_rounds': 50,
              'nthread':trial.suggest_int("nthread", 1, 10)}

    bst=lgb.train(lgb_params,dtrain,feval=rmse,valid_sets=[dvalid])
    preds = bst.predict(valid_x)
    accuracy = np.sqrt(mean_squared_error(valid_y,preds))
    print("accuracy: "+str(accuracy))
    return accuracy


### Read in training data, split to preds and vars


cols_own=["utctime","slhf","sshf",
        "ssrd","strd","str","ssr","skt-00",
        "sktn","laihv-00",
        "lailv-00",
        "sd-00","rsn-00",
        "stl1-00","stl2-00","swvl2-00",
        "t2-00","td2-00","u10-00",
        "v10-00","ro","evapp","longitude","latitude","DTM_height","DTM_slope",
        "DTM_aspect","clim_ts_value"
]

fname = "train_data_latest.csv"
print(fname)
df=pd.read_csv(data_dir+fname,usecols=cols_own)

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
        "ssrd","strd","str","ssr","skt-00",
        "sktn","laihv-00",
        "lailv-00",
        "sd-00","rsn-00",
        "stl1-00","stl2-00","swvl2-00",
        "t2-00","td2-00","u10-00",
        "v10-00","ro","evapp","longitude","latitude","DTM_height","DTM_slope",
        "DTM_aspect",'dayOfYear'
]

var=['clim_ts_value']
train_x=train_stations[preds] 
valid_x=test_stations[preds]
train_y=train_stations[var]
valid_y=test_stations[var]
train_x=train_x.astype(float)

dtrain = lgb.Dataset(train_x, label=train_y)
dvalid = lgb.Dataset(valid_x,label=valid_y)
    
### Optuna trials
study = optuna.create_study(storage="sqlite:///MLexperiments.sqlite3",study_name="lgbm-soiltemp-rmse-latest-1",direction="minimize",load_if_exists=True)
study.optimize(objective, n_trials=100, timeout=432000)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))
