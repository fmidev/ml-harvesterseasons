import os,optuna,time,random,warnings,sys
import sklearn.datasets
import sklearn.metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import numpy as np
warnings.filterwarnings("ignore")
### XGBoost with Optuna hyperparameter tuning for swi2 ML model
# note: does not save trained mdl
startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/soilwater/' # training data
mod_dir='/home/ubuntu/data/ML/models/soilwater' # saved mdl
res_dir='/home/ubuntu/data/ML/results/soilwater'
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
        "early_stopping_rounds":50,
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


### Read in training data, split to preds and vars
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
#fname='swi2_training_10000lucasPoints_2015-2022_all_fixed.csv'
fname=sys.argv[1]
print(fname)
df=pd.read_csv(data_dir+fname,usecols=cols_own)
        
# add day of year to predictors
df['utctime']=pd.to_datetime(df['utctime'])
df['dayOfYear'] = df['utctime'].dt.dayofyear
print(df)

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
train_x=train_stations[preds] 
valid_x=test_stations[preds]
train_y=train_stations[var]
valid_y=test_stations[var]
train_x=train_x.astype(float)
    
### Optuna trials
study = optuna.create_study(storage="sqlite:///MLexperiments.sqlite3",study_name="xgb-swi2-4",direction="minimize",load_if_exists=True)
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
