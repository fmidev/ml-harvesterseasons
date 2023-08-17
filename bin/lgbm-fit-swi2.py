import os, time, random, warnings,sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
'TH_LAT','TH_LONG','DTM_height','DTM_slope'
]
#fname='swi2_training_236lucasPoints_2015-2022-all.csv'
#fname='swi2_training_404lucasPoints_2015-2022-all.csv'
#fname='swi2_training_31801900_2015-2022_all.csv'
#fname='swi2_training_4108lucasPoints_2015-2022_all.csv'
fname='swi2_training_236lucasPoints_2015-2022_all_FIXED.csv'
#fname='swi2_training_404lucasPoints_2015-2022_all_FIXED.csv'
print(fname)
df=pd.read_csv(data_dir+fname,usecols=cols_own)

df['utctime']=pd.to_datetime(df['utctime'])
df['dayOfYear'] = df['utctime'].dt.dayofyear
print(df)

# drop negative swi2, NaN and -99999 values
s1=df.shape[0]
df=df.dropna(axis=0,how='any')
df=df[(df.DTM_slope !=-99999.000000) & (df.DTM_height !=-99999.000000)] 
df = df[df['swi2'] >= 0]
s2=df.shape[0]
print('From '+str(s1)+' dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')

# Split to train and test by years, choose years
test_y=[2018,2022]
train_y=[2015,2016,2017,2019,2020,2021]
#test_y=[2020,2022]
#train_y=[2015,2016,2017,2018,2019,2021]

print('test ',test_y,' train ',train_y)
train_stations,test_stations=pd.DataFrame(),pd.DataFrame()
for y in train_y:
        train_stations=pd.concat([train_stations,df[df['utctime'].dt.year == y]],ignore_index=True)
for y in test_y:
        test_stations=pd.concat([test_stations,df[df['utctime'].dt.year == y]],ignore_index=True)

#varOut=test_stations[['utctime','STAID','RR','tp-m']].copy()

# split data to predidctors (preds) and variable to be predicted (var)
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
'TH_LAT','TH_LONG','DTM_height','DTM_slope',
'dayOfYear'
]
var=['swi2']
preds_train=train_stations[preds] 
preds_test=test_stations[preds]
var_train=train_stations[var]
var_test=test_stations[var]
preds_train=preds_train.astype(float)

### LightGBM
'''params={
        'task':'train',
        'boosting':'gbdt',
        'objective':'regression',
        'num_leaves':7,
        'max_depth':10,
        'learning_rate':0.01,
        'verbosity':-1,
        'feature_fraction':, # similar to colsample_bytree
        'bagging_fraction':, # similar to subsample
        'bagging_freq':,
        'min_child_samples':,
        'lambda_l1': ,
        'lambda_l2': ,
        

}'''
eval_set=[(preds_test,var_test)]
model=LGBMRegressor(num_iterations=500,
                    metric='rmse')
model.fit(#params,
        preds_train,var_train,
        eval_set=eval_set)

# predict var and compare with test
var_pred=model.predict(preds_test)
mse=mean_squared_error(var_test,var_pred)
#varOut['RR-pred']=var_pred.tolist()
#varOut.to_csv(res_dir+'predicted_results_194sta.csv')

# save model 
#xgbr.save_model(mod_dir+'/mdl_swi2_2015-2022_4108points-3.txt')

print("RMSE: %.5f" % (mse**(1/2.0)))

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))