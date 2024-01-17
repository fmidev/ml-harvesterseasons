import os,optuna,time,random,warnings
import sklearn.datasets
import sklearn.metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import numpy as np
warnings.filterwarnings("ignore")
### XGBoost with Optuna hyperparameter tuning for soiltempearture ML model
# note: does not save trained mdl
startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/soiltemp/' # training data
# mod_dir='/home/ubuntu/data/ML/models/soiltemp' # saved mdl
# res_dir='/home/ubuntu/data/ML/results/soiltemp'
optuna_dir='/home/ubuntu/data/ML/' # optuna storage

os.chdir(optuna_dir)
print(os.getcwd())


### optuna objective & xgboost
def objective(trial):
    # hyperparameters
    param = {
        "objective":"reg:squarederror",
        "num_parallel_tree":trial.suggest_int("number_parallel_tree", 1, 10),
        "max_depth":trial.suggest_int("max_depth",3,18),
        "subsample":trial.suggest_float("subsample",0.01,1),
        "learning_rate":trial.suggest_float("learning_rate",0.01,0.7),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.01, 1.0),
        "n_estimators":trial.suggest_int("n_estimators",50,1000),
        "alpha":trial.suggest_float("alpha", 0.000000001, 1.0),
        "n_jobs":64,
        "random_state":99,
        #"early_stopping_rounds":50,
        "eval_metric":"rmse"
    }
    eval_set=[(valid_x,valid_y)]

    xgbr=xgb.XGBRegressor(**param)
    bst = xgbr.fit(train_x,train_y,eval_set=eval_set)
    preds = bst.predict(valid_x)
    pred_labels = np.rint(preds)
    accuracy = np.sqrt(mean_squared_error(valid_y,preds))
    print("accuracy: "+str(accuracy))
    return accuracy

### Read in training data, split to preds and varssoiltemp
cols_own=['utctime','slhf','sshf','ssrd','strd','str','ssr','sktn','sktd-12',
          'sf','laihv-00','laihv-12','lailv-00','lailv-12','sd-00','sd-12',
          'rsn-00','rsn-12','stl1-00','stl1-12','swvl2-00','swvl2-12','t2-00',
          't2-12','td2-00','td2-12','u10-00','u10-12','v10-00','v10-12','ro5d',
          'sro5d','ssro5d','evapp5d','tp5d','Forest_T_Min','Forest_T_Max','Forest_T_Mean',
          'longitude','latitude','TCD','WAW','DTM_height','DTM_slope','DTM_aspect','sand_0_5cm_mean',
          'sand_5_15cm_mean','silt_0_5cm_mean','silt_5_15cm_mean','clay_0_5cm_mean','clay_5_15cm_mean',
          'soc_0_5cm_mean','soc_5_15cm_mean','meanT_warmestQ_5_15cm','meanT_warmestQ_0_5cm',
          'mean_diurnal_0_5cm','clim_ts_value'
]
#fname='swi2_training_10000lucasPoints_2015-2022_all_fixed.csv'
fname='train_data.csv' # input training dataset

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
preds=['slhf','sshf','ssrd','strd','str','ssr',
          'sf','laihv-00','laihv-12','lailv-00','lailv-12','sd-00','sd-12','sktn','sktd-12',
          'rsn-00','rsn-12','stl1-00','stl1-12','swvl2-00','swvl2-12','t2-00',
          't2-12','td2-00','td2-12','u10-00','u10-12','v10-00','v10-12','ro5d',
          'sro5d','ssro5d','evapp5d','tp5d','Forest_T_Min','Forest_T_Max','Forest_T_Mean',
          'longitude','latitude','TCD','WAW','DTM_height','DTM_slope','DTM_aspect','sand_0_5cm_mean',
          'sand_5_15cm_mean','silt_0_5cm_mean','silt_5_15cm_mean','clay_0_5cm_mean','clay_5_15cm_mean',
          'soc_0_5cm_mean','soc_5_15cm_mean','meanT_warmestQ_5_15cm','meanT_warmestQ_0_5cm',
          'mean_diurnal_0_5cm','dayOfYear'
]
var=['clim_ts_value']
train_x=train_stations[preds] 
valid_x=test_stations[preds]
train_y=train_stations[var]
valid_y=test_stations[var]
train_x=train_x.astype(float)
    
### Optuna trials
study = optuna.create_study(storage="sqlite:///MLexperiments.sqlite3",study_name="xgb-soiltemp-rmse-1",direction="minimize",load_if_exists=True)
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
