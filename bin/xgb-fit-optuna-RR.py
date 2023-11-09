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

data_dir='/home/ubuntu/data/ML/training-data/precipitation-harmonia/' # training data
mod_dir='/home/ubuntu/data/ML/models/precipitation-harmonia/' # saved mdl
res_dir='/home/ubuntu/data/ML/results/precipitation-harmonia/'

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
cols_own=[
    'utctime','latitude','longitude','pointID',
    'anor','evap','kx-00','lsm','mn2t-00','msl-00','mx2t-00',
    'q500-00','q700-00','q850-00','ro','rsn-00','sd-00','sdor',
    'sf','skt-00','slhf','slor','sshf','ssr','ssrd','strd',#'sst-00',
    't2m-00','t500-00','t700-00','t850-00','tcc-00','td2m-00','tp',
    'tsr','ttr','u10-00','u500-00','u700-00','u850-00','v10-00',
    'v500-00',#'v700-00',
    'v850-00','z',#'z500-00',
    'z700-00','z850-00'
]# read in training data from era5 and eobs
# note: eobs does not always have full 20 years time series, unlike era5
fera5='RR_era5_training_6766stations_2000-2020_all.csv.gz' # era5 training data
feobs='RR_eobs_training_6766stations_2000-2020_all.csv.gz' # eobs training data
print(fera5)
print(feobs)
df_era5=pd.read_csv(data_dir+fera5,usecols=cols_own)
print(df_era5)
df_eobs=pd.read_csv(data_dir+feobs)
print(df_eobs)
df_eobs.rename(columns={'DATE':'utctime','STAID':'pointID','LAT': 'latitude', 'LON': 'longitude'}, inplace=True)
df=pd.merge(df_era5,df_eobs, on=['utctime','pointID','latitude','longitude'])
print(df)

executionTime=(time.time()-startTime)
print('File read in, in minutes: %.2f'%(executionTime/60))

# drop NaN values
s1=df.shape[0]
df=df.dropna(axis=0,how='any')
#df=df[(df.HGHT !=-99999) & (df.SLOPE !=-99999) & (df.ASPECT !=-99999)] 
s2=df.shape[0]
print('From '+str(s1)+' dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')

df_eobs,df_era5=[],[]

# add day of year to predictors
df['utctime']=pd.to_datetime(df['utctime'])
df['dayOfYear'] = df['utctime'].dt.dayofyear

# Fix E-OBS station rainfall observations (rain > 12 mm: x1.4, other x1.1)
#df.loc[ df['RR'] >= 0.012, 'RR'] = df['RR']*1.4 
#df.loc[ df['RR'] < 0.012, 'RR'] = df['RR']*1.1

print(df)

# Split to train and test by years, choose years
test_y=[2003, 2009, 2011, 2015]
train_y=[2000, 2001, 2002, 2004, 2005, 2006, 2007, 2008, 2010, 2012, 2013, 2014, 2016, 2017, 2018, 2019, 2020]
print('test ',test_y,' train ',train_y)
train_stations,test_stations=pd.DataFrame(),pd.DataFrame()
for y in train_y:
        train_stations=pd.concat([train_stations,df[df['utctime'].dt.year == y]],ignore_index=True)
for y in test_y:
        test_stations=pd.concat([test_stations,df[df['utctime'].dt.year == y]],ignore_index=True)

# split data to predidctors (preds) and variable to be predicted (var)
preds=[
    'latitude','longitude',#'HGHT','SLOPE','ASPECT',
    #'TPI','TRI','DTW','DTR','DTL','DTO','FCH',
    'anor','evap','kx-00','lsm','mn2t-00','msl-00','mx2t-00',
    'q500-00','q700-00','q850-00','ro','rsn-00','sd-00','sdor',
    'sf','skt-00','slhf','slor','sshf','ssr','ssrd','strd',#'sst-00',
    't2m-00','t500-00','t700-00','t850-00','tcc-00','td2m-00','tp',
    'tsr','ttr','u10-00','u500-00','u700-00','u850-00','v10-00',
    'v500-00',#'v700-00',
    'v850-00','z',#'z500-00',
    'z700-00','z850-00',
    'dayOfYear'
]
var=['RR'] 
train_x=train_stations[preds] 
valid_x=test_stations[preds]
train_y=train_stations[var]
valid_y=test_stations[var]

df=[]

### Optuna trials
study = optuna.create_study(storage="sqlite:///MLexperiments.sqlite3",study_name="xgb-RR-1",direction="minimize",load_if_exists=True)
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