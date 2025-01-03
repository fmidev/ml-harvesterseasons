import time,warnings
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings("ignore")
### XGBoost for OCEANIDS

startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/OCEANIDS/' # training data
mod_dir='/home/ubuntu/data/ML/models/OCEANIDS' # saved mdl
res_dir='/home/ubuntu/data/ML/results/OCEANIDS'

'''
all cols 
utctime,latitude,longitude,FMISID,WS_PT1H_AVG,WG_PT1H_MAX,dayOfYear,hour,
utctime,lat-1,lon-1,lat-2,lon-2,lat-3,lon-3,lat-4,lon-4,e-1,e-2,e-3,e-4,
ewss-1,ewss-2,ewss-3,ewss-4,fg10-1,fg10-2,fg10-3,fg10-4,lsm-1,lsm-2,lsm-3,lsm-4,
msl-1,msl-2,msl-3,msl-4,nsss-1,nsss-2,nsss-3,nsss-4,slhf-1,slhf-2,slhf-3,slhf-4,
sshf-1,sshf-2,sshf-3,sshf-4,ssr-1,ssr-2,ssr-3,ssr-4,ssrd-1,ssrd-2,ssrd-3,ssrd-4,
str-1,str-2,str-3,str-4,strd-1,strd-2,strd-3,strd-4,t2-1,t2-2,t2-3,t2-4,
tcc-1,tcc-2,tcc-3,tcc-4,td2-1,td2-2,td2-3,td2-4,tlwc-1,tlwc-2,tlwc-3,tlwc-4,
tp-1,tp-2,tp-3,tp-4,tsea-1,tsea-2,tsea-3,tsea-4,u10-1,u10-2,u10-3,u10-4,
v10-1,v10-2,v10-3,v10-4
'''
### Read in 2D tabular training data
cols_own=['utctime','WS_PT1H_AVG','dayOfYear',#'hour',
#'lat-1','lon-1','lat-2','lon-2','lat-3','lon-3','lat-4','lon-4',
'e-1','e-2','e-3','e-4','ewss-1','ewss-2','ewss-3','ewss-4',
'fg10-1','fg10-2','fg10-3','fg10-4','lsm-1','lsm-2','lsm-3','lsm-4',
'msl-1','msl-2','msl-3','msl-4','nsss-1','nsss-2','nsss-3','nsss-4',
'slhf-1','slhf-2','slhf-3','slhf-4','sshf-1','sshf-2','sshf-3','sshf-4',
'ssr-1','ssr-2','ssr-3','ssr-4','ssrd-1','ssrd-2','ssrd-3','ssrd-4',
'str-1','str-2','str-3','str-4','strd-1','strd-2','strd-3','strd-4',
't2-1','t2-2','t2-3','t2-4','tcc-1','tcc-2','tcc-3','tcc-4',
'td2-1','td2-2','td2-3','td2-4','tlwc-1','tlwc-2','tlwc-3','tlwc-4',
'tp-1','tp-2','tp-3','tp-4','tsea-1','tsea-2','tsea-3','tsea-4',
'u10-1','u10-2','u10-3','u10-4','v10-1','v10-2','v10-3','v10-4'
]
fname = 'training_data_oceanids_Vuosaari-sf_2013-2023.csv' # training input data file
print(fname)
df=pd.read_csv(data_dir+fname,usecols=cols_own)

# drop NaN values
df=df.dropna(axis=1, how='all')
s1=df.shape[0]
df=df.dropna(axis=0,how='any')
s2=df.shape[0]
print('From '+str(s1)+' rows dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')
df['utctime']= pd.to_datetime(df['utctime'])
headers=list(df) # list column headers
#print(df)

# Split to train and test by years, KFold for best split (k=5)
test_y=[2016,2022]
train_y=[2013,2014,2015,2017,2018,2019,2020,2021,2023]
print('test ',test_y,' train ',train_y)
train_stations,test_stations=pd.DataFrame(),pd.DataFrame()
for y in train_y:
        train_stations=pd.concat([train_stations,df[df['utctime'].dt.year == y]],ignore_index=True)
for y in test_y:
        test_stations=pd.concat([test_stations,df[df['utctime'].dt.year == y]],ignore_index=True)

# Split to predictors (preds) and predictand (var) data
var_headers=list(df[['WS_PT1H_AVG']])
preds_headers=list(df[headers].drop(['utctime','WS_PT1H_AVG'], axis=1))
preds_train=train_stations[preds_headers] 
preds_test=test_stations[preds_headers]
var_train=train_stations[var_headers]
var_test=test_stations[var_headers]

### XGBoost
# Define model hyperparameters (Optuna tuned)
nstm=645
lrte=0.067
max_depth=10
subsample=0.29
colsample_bytree=0.56
#colsample_bynode=1
num_parallel_tree=10
a=0.54

# initialize and tune model
xgbr=xgb.XGBRegressor(
            objective= 'reg:squarederror',#'count:poisson'
            n_estimators=nstm,
            learning_rate=lrte,
            max_depth=max_depth,
            alpha=a,#gamma=0.01
            num_parallel_tree=num_parallel_tree,
            n_jobs=64,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            #colsample_bynode=colsample_bynode,
            eval_metric='rmse', #'mae', 
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
xgbr.save_model(mod_dir+'/mdl_WSPT1HAVG_2013-2023_sf_test.txt')

print("RMSE: %.5f" % (mse**(1/2.0)))
print("MAE: %.5f" % (mae))

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))