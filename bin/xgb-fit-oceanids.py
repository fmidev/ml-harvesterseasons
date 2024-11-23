import time,warnings
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
#from Vuosaari_151028_simple import *
#from Vuosaari_151028 import *
#from Vuosaari_151028_FGo import *
#from Raahe_101785_simple import *
#from Raahe_101785 import *
#from Raahe_101785_FGo import *
#from Rauma_101061_simple import *
#from Rauma_101061 import *
#from Rauma_101061_FGo import *
#from Malaga_000231 import *
#from Sitia_023330_ece3 import *
#from Cadiz_000415_ece3 import *
#from Ploumanach_011245_ece3 import *
#from Vuosaari_151028_cordex_ncc_dmi import *
#from Aktio_023317_cordex_ncc_smhi import *
#from Bremerhaven_004885 import *
#from Vuosaari_151028_cordex_ncc_smhi import *
#from Bremerhaven_004885_ece3 import *
#from Vuosaari_151028_cordex_cnrm_cnrm import *
from Bremerhaven_004885_ece3 import *

warnings.filterwarnings("ignore")
### XGBoost for OCEANIDS

startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/OCEANIDS/' # training data
mod_dir='/home/ubuntu/data/ML/models/OCEANIDS' # saved mdl
res_dir='/home/ubuntu/data/ML/results/OCEANIDS'

### Read in 2D tabular training data
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
print('test ',test_y,' train ',train_y)
train_stations,test_stations=pd.DataFrame(),pd.DataFrame()
for y in train_y:
        train_stations=pd.concat([train_stations,df[df['utctime'].dt.year == y]],ignore_index=True)
for y in test_y:
        test_stations=pd.concat([test_stations,df[df['utctime'].dt.year == y]],ignore_index=True)

# Split to predictors (preds) and predictand (var) data
var_headers=list(df[[pred]])
preds_headers=list(df[headers].drop(['utctime',pred], axis=1))
preds_train=train_stations[preds_headers] 
preds_test=test_stations[preds_headers]
var_train=train_stations[var_headers]
var_test=test_stations[var_headers]

### XGBoost
# Define model hyperparameters (Optuna tuned)
nstm=702
lrte=0.2394792064153895
max_depth=4
subsample=0.7932960772635375
colsample_bytree=0.3871330342069024
#colsample_bynode=1
num_parallel_tree=9
a=0.85970950766814

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
xgbr.save_model(mod_dir+'/'+mdl_name)

print("RMSE: %.5f" % (mse**(1/2.0)))
print("MAE: %.5f" % (mae))

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))