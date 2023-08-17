import os, time, random, warnings,sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings("ignore")
### XGBoost for testing SWI2 fitting 07/2023

startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/soilwater/' # training data
mod_dir='/home/ubuntu/data/ML/models/soilwater' # saved mdl
res_dir='/home/ubuntu/data/ML/results/soilwater'

### Read in 2D tabular training data
'''
all cols in training file
utctime,swi2,evap,evap5d,evap15d,evap60d,evap100d,evapp,evapp5d,evapp15d,evapp60d,evapp100d,
laihv-00,laihv-12,lailv-00,lailv-12,ro,ro5d,ro15d,ro60d,ro100d,rsn-00,rsn-12,sd-00,sd-12,sf,
skt-00,skt-12,slhf,sro,sro5d,sro15d,sro60d,sro100d,sshf,ssr,ssrd,ssro,ssro5d,ssro15d,ssro60d,
ssro100d,stl1-00,stl1-12,str,strd,swvl1-00,swvl1-12,swvl2-00,swvl2-12,swvl3-00,swvl3-12,swvl4-00,
swvl4-12,t2-00,t2-12,td2-00,td2-12,tp,tp5d,tp15d,tp60d,tp100d,u10-00,u10-12,v10-00,v10-12,
TH_LAT,TH_LONG,DTM_height,DTM_slope,DTM_aspect,TCD,WAW,CorineLC
'''

'''
first version teaching with
cols_own=['utctime','swi2','evap','evap5d','evap15d','evap60d','evap100d','tp','tp5d','tp15d','tp60d','tp100d','ro','ro5d','ro15d','ro60d','ro100d',
'lailv-00','lailv-12','laihv-00','laihv-12','swvl1-00','swvl1-12','swvl2-00','swvl2-12','swvl3-00','swvl3-12','swvl4-00','swvl4-12',
'u10-00','u10-12','v10-00','v10-12','rsn-00','rsn-12','sd-00','sd-12','stl1-00','stl1-12','t2-00','t2-12','td2-00','td2-12','sf','slhf','sshf',
'ssr','ssrd','str','strd','TH_LAT','TH_LONG','DTM_height','DTM_slope'
]
'''

cols_own=['utctime','swi2','evap','evap5d','evap15d','evap60d','evap100d','evapp','evapp5d','evapp15d','evapp60d','evapp100d',
'laihv-00','laihv-12','lailv-00','lailv-12','ro','ro5d','ro15d','ro60d','ro100d','rsn-00','rsn-12','sd-00','sd-12','sf',
'skt-00','skt-12','slhf','sro','sro5d','sro15d','sro60d','sro100d','sshf','ssr','ssrd','ssro','ssro5d','ssro15d','ssro60d',
'ssro100d','stl1-00','stl1-12','str','strd','swvl1-00','swvl1-12','swvl2-00','swvl2-12','swvl3-00','swvl3-12','swvl4-00',
'swvl4-12','t2-00','t2-12','td2-00','td2-12','tp','tp5d','tp15d','tp60d','tp100d','u10-00','u10-12','v10-00','v10-12',
'TH_LAT','TH_LONG','DTM_height','DTM_slope'
]
#fname='swi2_training_236lucasPoints_2015-2022-all.csv'
#fname='swi2_training_404lucasPoints_2015-2022-all.csv'
#fname='swi2_training_31801900_2015-2022_all.csv'
#fname='swi2_training_5000lucasPoints_2015-2022_all.csv'
point=sys.argv[1]
path1='swi2-training-63287points/'
fname='swi2_training_'+str(point)+'_2015-2022_all.csv'
print(fname)
df=pd.read_csv(data_dir+path1+fname,usecols=cols_own)
#print(df)
'''
df['RR']=df['RR']/10000.0 # units from x0.1mm to m
# Fix E-OBS station rainfall observations (rain > 12 mm: x1.4, other x1.1)
#df.loc[ df['RR'] >= 0.012, 'RR'] = df['RR']*1.4 
#df.loc[ df['RR'] < 0.012, 'RR'] = df['RR']*1.1
''' 

df['utctime']=pd.to_datetime(df['utctime'])
df['dayOfYear'] = df['utctime'].dt.dayofyear
#print(df)

# Split to train and test by years, choose years
test_y=[2018,2022]
train_y=[2015,2016,2017,2019,2020,2021]
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
#varOut=test_stations[['utctime','STAID','RR','tp-m']].copy()
print('test set dropped '+str(testnans1-testnans2)+' rows')
print('train set dropped '+str(trainnans1-trainnans2)+' rows')
train_stations=train_stations.drop(columns=['utctime'])
test_stations=test_stations.drop(columns=['utctime'])

# split data to predidctors (preds) and variable to be predicted (var)
''' first training preds
preds=['evap','evap5d','evap15d','evap60d','evap100d','tp','tp5d','tp15d','tp60d','tp100d','ro','ro5d','ro15d','ro60d','ro100d',
'lailv-00','lailv-12','laihv-00','laihv-12','swvl1-00','swvl1-12','swvl2-00','swvl2-12','swvl3-00','swvl3-12','swvl4-00','swvl4-12',
'u10-00','u10-12','v10-00','v10-12','rsn-00','rsn-12','sd-00','sd-12','stl1-00','stl1-12','t2-00','t2-12','td2-00','td2-12','sf','slhf','sshf',
'ssr','ssrd','str','strd','TH_LAT','TH_LONG','DTM_height','DTM_slope','dayOfYear'
]
'''
preds=['evap','evap5d','evap15d','evap60d','evap100d','evapp','evapp5d','evapp15d','evapp60d','evapp100d',
'laihv-00','laihv-12','lailv-00','lailv-12','ro','ro5d','ro15d','ro60d','ro100d','rsn-00','rsn-12','sd-00','sd-12','sf',
'skt-00','skt-12','slhf','sro','sro5d','sro15d','sro60d','sro100d','sshf','ssr','ssrd','ssro','ssro5d','ssro15d','ssro60d',
'ssro100d','stl1-00','stl1-12','str','strd','swvl1-00','swvl1-12','swvl2-00','swvl2-12','swvl3-00','swvl3-12','swvl4-00',
'swvl4-12','t2-00','t2-12','td2-00','td2-12','tp','tp5d','tp15d','tp60d','tp100d','u10-00','u10-12','v10-00','v10-12',
'TH_LAT','TH_LONG','DTM_height','DTM_slope',
'dayOfYear'
]
var=['swi2']
preds_train=train_stations[preds] 
preds_test=test_stations[preds]
var_train=train_stations[var]
var_test=test_stations[var]
preds_train=preds_train.astype(float)
#print(preds_train)

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
#varOut['RR-pred']=var_pred.tolist()
#varOut.to_csv(res_dir+'predicted_results_194sta.csv')

# save model 
#xgbr.save_model(mod_dir+'/mdl_swi2_2015-2022_5000points-1.txt')

print("RMSE: %.5f" % (mse**(1/2.0)))

# for some reason data for certain points is corrupted and I have no idea how, this is to check point by point which files actually work
# when running parallel, it saves working ones in the txt file and otherwise crashes with those that don't work
# one file appr. 0.16 min to run
with open('/home/ubuntu/data/ML/training-data/soilwater/workingPointsFromAll.txt', 'a') as file:
    file.write(point+'\n')

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))