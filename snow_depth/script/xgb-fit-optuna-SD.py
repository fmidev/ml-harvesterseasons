import os,optuna,time,random,warnings,sys
import sklearn.datasets
import sklearn.metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import numpy as np
warnings.filterwarnings("ignore")
### XGBoost with Optuna hyperparameter tuning for snow depth ML model
# note: does not save trained mdl
startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/snowdepth/' # training data
mod_dir='/home/ubuntu/data/ML/models/snowdepth/' # saved mdl
res_dir='/home/ubuntu/data/ML/results/snowdepth/'

optuna_dir='/home/ubuntu/data/ML/' # optuna storage
os.chdir(optuna_dir) # move to dir with optuna storage
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

# Read in training data
#Do we need to include lat, lon? 'latitude','longitude'
# tp excluded
cols_own=[
    'utctime','pointID','slhf','sshf','ssrd',
    'strd','str','ssr','sf','laihv-00','lailv-00','sd-00','rsn-00',
    'stl1-00','swvl2-00','t2-00','td2-00','u10-00','v10-00',
    'ro','evap'
]

# CLMS columns: utctime, swe_clms, latitude, longitude, pointID
cols_clms=[
    'utctime','pointID','swe_clms'
]

# read in training data from era5l, CLMS and eobs
# note: eobs and clms do not always have full years time series, unlike era5l
fera5l='SD_era5l_training_9790stations_2006-2020_all.csv.gz' # (predictors) era5-L training data
fclms='SD_clms_swe_training_9790stations_2006-2020.csv' # (predictors) clms training data
feobs='SD_eobs_fmi_training_9790stations_2006-2020.csv.gz' # eobs (predictand) training data

print(fera5l)
print(fclms)
print(feobs)

df_era5l=pd.read_csv(data_dir+fera5l,usecols=cols_own)
print(df_era5l)
df_clms=pd.read_csv(data_dir+fclms,usecols=cols_clms)
print(df_clms)
df_eobs=pd.read_csv(data_dir+feobs)
print(df_eobs)
df_eobs.rename(columns={'DATE':'utctime','STAID':'pointID','LAT': 'latitude', 'LON': 'longitude'}, inplace=True)
# SD column: replace nan's with 0
df_eobs['SD']=df_eobs['SD'].replace(np.nan,0)

# merge SD eobs, clms and era5l data
df_eobs_clms=pd.merge(df_clms,df_eobs, on=['utctime','pointID'])
df=pd.merge(df_era5l,df_eobs_clms, on=['utctime','pointID'])
# note; lat,lon not used in merging, thats's why commented:
#df=pd.merge(df_era5l,df_eobs, on=['utctime','pointID','latitude','longitude'])
print(df)

# - - - - - - - RR example:
# read in training data from era5 and eobs
# note: eobs does not always have full 20 years time series, unlike era5
#fera5='RR_era5_training_6766stations_2000-2020_all.csv.gz' # era5 training data
#feobs='RR_eobs_training_6766stations_2000-2020_all.csv.gz' # eobs training data
#print(fera5)
#print(feobs)
#df_era5=pd.read_csv(data_dir+fera5,usecols=cols_own)
#print(df_era5)
#df_eobs=pd.read_csv(data_dir+feobs)
#print(df_eobs)
#df_eobs.rename(columns={'DATE':'utctime','STAID':'pointID','LAT': 'latitude', 'LON': 'longitude'}, inplace=True)
#df=pd.merge(df_era5,df_eobs, on=['utctime','pointID','latitude','longitude'])
#print(df)
# - - - - - - - end RR example

executionTime=(time.time()-startTime)
print('File read in, in minutes: %.2f'%(executionTime/60))

# drop NaN values
s1=df.shape[0]
df=df.dropna(axis=0,how='any')
#df=df[(df.HGHT !=-999)]  # is this needed? not used as a predictor now, HGHT from eobs(eca&d) data
#df = df[df['rsn-00'] !=99.999985]  # check if this is needed or not!
#df=df[(df.HGHT !=-99999) & (df.SLOPE !=-99999) & (df.ASPECT !=-99999)]  # from RR example
s2=df.shape[0]
print('From '+str(s1)+' dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')

df_eobs,df_era5l=[],[]
df_clms = []
df_eobs_clms = []

# add day of year to predictors
df['utctime']=pd.to_datetime(df['utctime'])
df['dayOfYear'] = df['utctime'].dt.dayofyear

print(df)

# Split to train and test by years, choose years, KFold for best split (k=5)
#Fold: 5 RMSE: 5.85
#Train:  [2006 2007 2008 2010 2011 2012 2013 2014 2015 2017 2019 2020] Test:  [2009 2016 2018]
test_y=[2009, 2016, 2018]
train_y=[2006, 2007, 2008, 2010, 2011, 2012, 2013, 2014, 2015, 2017, 2019, 2020]
print('test ',test_y,' train ',train_y)
train_stations,test_stations=pd.DataFrame(),pd.DataFrame()
for y in train_y:
        train_stations=pd.concat([train_stations,df[df['utctime'].dt.year == y]],ignore_index=True)
for y in test_y:
        test_stations=pd.concat([test_stations,df[df['utctime'].dt.year == y]],ignore_index=True)

# split data to predidctors (preds) and variable to be predicted (var)
# Note excluded here: 'latitude','longitude',pointID
# excluded tp
preds=[
    'slhf','sshf','ssrd','strd','str','ssr','sf',
    'laihv-00','lailv-00','sd-00','rsn-00',
    'stl1-00','swvl2-00','t2-00','td2-00','u10-00','v10-00',
    'ro','evap','swe_clms',
    'dayOfYear']
var=['SD'] # variable to be predicted
train_x=train_stations[preds] 
valid_x=test_stations[preds]
train_y=train_stations[var]
valid_y=test_stations[var]

df=[]

### Optuna trials
# Is this OK for storage? this example from RR
#study = optuna.create_study(storage="sqlite:///MLexperiments.sqlite3",study_name="xgb-RR-1",direction="minimize",load_if_exists=True)
study = optuna.create_study(storage="sqlite:///MLexperiments.sqlite3",study_name="xgb-snowdepth-rmse-9790stations-2",direction="minimize",load_if_exists=True)
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
