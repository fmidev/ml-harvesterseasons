import xgboost as xgb
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/OCEANIDS/' # training data
mdls_dir='/home/ubuntu/data/ML/models/OCEANIDS/' # saved mdl
res_dir='/home/ubuntu/data/ML/results/OCEANIDS/'

mod_fname='mdl_WSPT1HAVG_2013-2023_test.txt'

# read in predictors in the fitted model from training data file
fname = 'training_data_oceanids_Vuosaari_2013-2023.csv' # training input data file
df=pd.read_csv(data_dir+fname)
df=df.dropna(axis=1, how='all')
s1=df.shape[0]
df=df.dropna(axis=0,how='any')
s2=df.shape[0]
print('From '+str(s1)+' rows dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')
df['utctime']= pd.to_datetime(df['utctime'])
headers=list(df) # list column headers
preds=list(df[headers].drop(['utctime','utctime.1','WS_PT1H_AVG','latitude', 'longitude', 'FMISID', 'WG_PT1H_MAX','lat-1', 'lon-1', 'lat-2', 'lon-2', 'lat-3', 'lon-3', 'lat-4', 'lon-4'], axis=1))
print(preds)
## F-score
print("start fscore")
mdl=mdls_dir+mod_fname
models=[]
fitted_mdl=xgb.XGBRegressor()
fitted_mdl.load_model(mdl)
models.append(fitted_mdl)

all_scores=pd.DataFrame(columns=['Model','predictor','meangain'])
row=0
for i,mdl in enumerate(models):
    mdl.get_booster().feature_names = list(preds) # predictor column headers
    bst=mdl.get_booster() # get the underlying xgboost Booster of model
    gains=np.array(list(bst.get_score(importance_type='gain').values()))
    features=np.array(list(bst.get_fscore().keys()))
    '''
    get_fscore uses get_score with importance_type equal to weight
    weight: the number of times a feature is used to split the data across all trees
    gain: the average gain across all splits the feature is used in
    '''
    for feat,gain in zip(features,gains):
        all_scores.loc[row]=(i+1,feat,gain); row+=1
all_scores=all_scores.drop(columns=['Model'])
mean_scores=all_scores.groupby('predictor').mean().sort_values('meangain')
print(mean_scores)

f, ax = plt.subplots(1,1,figsize=(6, 10))
mean_scores.plot.barh(ax=ax, legend=False)
ax.set_xlabel('F score')
ax.set_title(mod_fname)
ax.set_xscale('log')
plt.tight_layout()
f.savefig(res_dir+'Fscore_WS_PT1H_AVG-test.png', dpi=200)
#plt.show()
plt.clf(); plt.close('all')

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))



