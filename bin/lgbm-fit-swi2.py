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
warnings.filterwarnings("ignore")
### XGBoost for swi2

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
'TH_LAT','TH_LONG','DTM_height','DTM_slope','DTM_aspect'
]
#fname='swi2_training_236lucasPoints_2015-2022-all.csv'
#fname='swi2_training_404lucasPoints_2015-2022-all.csv'
#fname='swi2_training_31801900_2015-2022_all.csv'
#fname='swi2_training_236lucasPoints_2015-2022_all_FIXED.csv'
#fname='swi2_training_404lucasPoints_2015-2022_all_FIXED.csv'
#fname='swi2_training_4108lucasPoints_2015-2022_all.csv'
#fname='swi2_training_10000lucasPoints_2015-2022_all.csv'
#fname='swi2_training_10000lucasPoints_2015-2022_all_2.csv'
#fname='swi2_training_10000lucasPoints_2015-2022_all_2+.csv'
fname='swi2_training_10000lucasPoints_2015-2022_all_fixed.csv'
print(fname)
df=pd.read_csv(data_dir+fname,usecols=cols_own)

'''
df['RR']=df['RR']/10000.0 # units from x0.1mm to m
# Fix E-OBS station rainfall observations (rain > 12 mm: x1.4, other x1.1)
#df.loc[ df['RR'] >= 0.012, 'RR'] = df['RR']*1.4 
#df.loc[ df['RR'] < 0.012, 'RR'] = df['RR']*1.1
''' 

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

#varOut=test_stations[['utctime','STAID','RR','tp-m']].copy()

# split data to predidctors (preds) and variable to be predicted (var)
''' 
#first training preds
preds=['evap','evap5d','evap15d','evap60d','evap100d','tp','tp5d','tp15d','tp60d','tp100d','ro','ro5d','ro15d','ro60d','ro100d',
'lailv-00','lailv-12','laihv-00','laihv-12','swvl1-00','swvl1-12','swvl2-00','swvl2-12','swvl3-00','swvl3-12','swvl4-00','swvl4-12',
'u10-00','u10-12','v10-00','v10-12','rsn-00','rsn-12','sd-00','sd-12','stl1-00','stl1-12','t2-00','t2-12','td2-00','td2-12','sf','slhf','sshf',
'ssr','ssrd','str','strd','TH_LAT','TH_LONG','DTM_height','DTM_slope','dayOfYear'
]
'''
'''
#training with many pars
preds=['evap','evap5d','evap15d','evap60d','evap100d','evapp','evapp5d','evapp15d','evapp60d','evapp100d',
'laihv-00','laihv-12','lailv-00','lailv-12','ro','ro5d','ro15d','ro60d','ro100d','rsn-00','rsn-12','sd-00','sd-12','sf',
'skt-00','skt-12','slhf','sro','sro5d','sro15d','sro60d','sro100d','sshf','ssr','ssrd','ssro','ssro5d','ssro15d','ssro60d',
'ssro100d','stl1-00','stl1-12','str','strd','swvl1-00','swvl1-12','swvl2-00','swvl2-12','swvl3-00','swvl3-12','swvl4-00',
'swvl4-12','t2-00','t2-12','td2-00','td2-12','tp','tp5d','tp15d','tp60d','tp100d','u10-00','u10-12','v10-00','v10-12',
'TH_LAT','TH_LONG','DTM_height','DTM_slope',
'dayOfYear'
]'''
# training with relevant pars
preds=['evap','evap15d','evapp','evapp15d',
'laihv-00','laihv-12','lailv-00','lailv-12','ro','ro15d','rsn-00','rsn-12','sd-00','sd-12','sf',
'skt-00','skt-12','slhf','sro','sro15d','sshf','ssr','ssrd','ssro','ssro15d',
'stl1-00','stl1-12','str','strd','swvl2-00','swvl2-12','t2-00','t2-12','td2-00',
'td2-12','tp','tp15d','u10-00','u10-12','v10-00','v10-12',
'TH_LAT','TH_LONG','DTM_height','DTM_slope','DTM_aspect',
'dayOfYear'
]
var=['swi2']
preds_train=train_stations[preds] 
preds_test=test_stations[preds]
var_train=train_stations[var]
var_test=test_stations[var]
preds_train=preds_train.astype(float)

### LGMB
# Define model hyperparameters 
nstm=500
lrte=0.1
max_depth=7
subsample=0.7
colsample_bytree=0.7
colsample_bynode=1
num_parallel_tree=10

def rmse(preds, train_data):
    labels = train_data.get_label()
    MSE = mean_squared_error(labels, preds)
    RMSE = math.sqrt(MSE)
    return 'RMSE', RMSE, False

# initialize and tune model
lgb_params = {'objective' :'regression',
              'num_leaves': 10,
              'n_estimators':nstm,
              'learning_rate': lrte,
              'feature_fraction': 0.8,
              'max_depth': max_depth,
              'reg_alpha':0.01,
              'n_jobs':64,
              'random_state':99,
              'subsample':subsample,
              'colsample_bytree':colsample_bytree,
              'verbose': 0,
              'num_boost_round': 10000,
              'early_stopping_rounds': 50,
              'nthread':num_parallel_tree}
'''xgbr=xgb.XGBRegressor(
            objective= 'reg:squarederror',#'count:poisson'
            n_estimators=nstm,
            learning_rate=lrte,
            max_depth=max_depth,
            alpha=0.01,#gamma=0.01
            num_parallel_tree=num_parallel_tree,
            n_jobs=64,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bynode=colsample_bynode,
            eval_metric='rmse',
            random_state=99,
            early_stopping_rounds=50
            )'''

# train model 
lgbtrain = lgb.Dataset(data=preds_train, label=var_train, feature_name=cols_own)

lgbval = lgb.Dataset(data=preds_test, label=var_test, reference=lgbtrain, feature_name=cols_own)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=rmse,
                  verbose_eval=100)



##################
'''eval_set=[(preds_test,var_test)]
lgb.fit(
        preds_train,var_train,
        eval_set=eval_set)'''
print(model)

# predict var and compare with test
var_pred=model.predict(preds_test,num_iteration=model.best_iteration)
mse=mean_squared_error(var_test,var_pred)
#varOut['RR-pred']=var_pred.tolist()
#varOut.to_csv(res_dir+'predicted_results_194sta.csv')

# save model 
lgb.save_model(mod_dir+'/lgb_swi2_2015-2022_10000points-5.txt')

print("RMSE: %.5f" % (mse**(1/2.0)))

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))

# plotting feature importance
plot_importance(model)
plt.savefig('plot_importance_swi2_lgbm.jpg', height=.5,grid=None)