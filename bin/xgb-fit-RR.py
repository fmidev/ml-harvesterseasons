import time, warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings('ignore')
### XGBoost for downscaling and bias-adjusting precipitation (2024)

startTime=time.time()
file_path='RR_config.py'
with open(file_path, 'r') as file:
    exec(file.read())

predictand=predictand2
not_predictand=predictand1
print('Using '+predictand+' as predictand instead of '+not_predictand)

# Read in training data
df=pd.read_csv(fname,usecols=cols_own)
df.rename(columns={'DATE':'utctime'}, inplace=True)

# add day of year to predictors
df['utctime']=pd.to_datetime(df['utctime'])
df['dayOfYear'] = df['utctime'].dt.dayofyear

# add pressure change from previous day to predictors
df['Dmsl-00'] = df['msl-00'].diff()
df['Dmsl-12'] = df['msl-12'].diff()

# Weighted E-OBS station rainfall observations (rain > 12 mm: x1.4, other x1.1)
df.loc[ df['RR'] >= 0.012, 'WeightRR'] = df['RR']*1.4 
df.loc[ df['RR'] < 0.012, 'WeightRR'] = df['RR']*1.1

# drop NaN values and negative RR 
s1=df.shape[0]
df=df.replace(-99999.0,np.nan)
df=df.replace(-9999.0,np.nan)
df=df.dropna(axis=0,how='any')
df = df[df['RR'] >= 0]
s2=df.shape[0]
print('From '+str(s1)+' rows of data dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')
num_stations = df['STAID'].nunique()
print(f'Number of stations: {num_stations}')
headers=list(df) # list column headers

print(df)

'''
# Station IDS after processing NaN etc
stations=df.pointID.drop_duplicates().to_list()
file = open(data_dir+'training_stations_negRRfixed.txt','w')
for sta in stations:
	file.write(str(sta)+'\n')
file.close()
'''

# Split to train and test by years, KFold for best split (k=5)
print('test ',test_y,' train ',train_y)
train_stations,test_stations=pd.DataFrame(),pd.DataFrame()
for y in train_y:
        train_stations=pd.concat([train_stations,df[df['utctime'].dt.year == y]],ignore_index=True)
for y in test_y:
        test_stations=pd.concat([test_stations,df[df['utctime'].dt.year == y]],ignore_index=True)

# Split to predictors (preds) and predictand (var) data
var_headers=list(df[[predictand]])
preds_headers=list(df[headers].drop(['utctime','STAID',predictand,not_predictand], axis=1))
preds_train,preds_test=train_stations[preds_headers],test_stations[preds_headers] 
var_train,var_test=train_stations[var_headers],test_stations[var_headers]

df=None # free memory

# XGBoost
# initialize and tune model
xgbr=xgb.XGBRegressor(
            objective= 'reg:squarederror',
            n_estimators=nstm,
            learning_rate=lrte,
            max_depth=max_depth,
            alpha=al,#gamma=0.01
            num_parallel_tree=num_parallel_tree,
            n_jobs=64,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
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
mae=mean_absolute_error(var_test,var_pred)

# save model 
xgbr.save_model(mod_name)

print('RMSE: %.5f' % (mse**(1/2.0)*1000))
print('MAE: %.5f' % (mae*1000))

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))
print('create log...')
# create log
with open(logfile, 'w') as file:
    file.write('Log for '+name+' XGB run.\n')
    file.write('Using '+predictand+' as predictand instead of '+not_predictand+'\n')
    file.write('RMSE: %.5f\n' % (mse**(1 / 2.0) * 1000))
    file.write('MAE: %.5f\n' % (mae * 1000))
    file.write('From '+str(s1)+' rows of data dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %\n')
    file.write(f'Number of stations: {num_stations}\n')
    file.write('test years: '+test_y+', train years: '+train_y+'\n')
    file.write(f'Hyperparameters:\nnstm={nstm}\nlrte={lrte}\nmax_depth={max_depth}\nsubsample={subsample}\ncolsample_bytree={colsample_bytree}\nnum_parallel_tree={num_parallel_tree}\nal={al}\n')
    file.write('Execution time in minutes: %.2f\n'%(executionTime/60))    
print('...log finished')
