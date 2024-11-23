import xgboost as xgb # type: ignore
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from Raahe_101785 import *
#from Raahe_101785_simple import *
#from Raahe_101785_FGo import *
#from Vuosaari_151028 import *
#from Vuosaari_151028_simple import *
#from Vuosaari_151028_FGo import *
#from Rauma_101061 import *
#from Rauma_101061_simple import *
#from Rauma_101061_FGo import *
#from Malaga_000231 import * # type: ignore
#from Sitia_023330_ece3 import *
#from Cadiz_000415_ece3 import *
#from Ploumanach_011245_ece3 import *
from Bremerhaven_004885_ece3 import *

startTime=time.time()

def filter_points(df,lat,lon,nro,name):
    df0=df.copy()
    filter1 = df0['latitude'] == lat
    filter2 = df0['longitude'] == lon
    df0.where(filter1 & filter2, inplace=True)
    df0.columns=['lat-'+str(nro),'lon-'+str(nro),name+'-'+str(nro)] # change headers      
    df0=df0.dropna()
    return df0

data_dir='/home/ubuntu/data/ML/training-data/OCEANIDS/' # training data
mdls_dir='/home/ubuntu/data/ML/models/OCEANIDS/' # saved mdl
res_dir='/home/ubuntu/data/ML/results/OCEANIDS/'

# read in predictors in the fitted model from training data file
print(fname) # type: ignore
df=pd.read_csv(data_dir+fname,usecols=cols_own) # type: ignore
df=df.dropna(axis=1, how='all')
s1=df.shape[0]
df=df.dropna(axis=0,how='any')
s2=df.shape[0]
print('From '+str(s1)+' rows dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')

df['utctime']= pd.to_datetime(df['utctime'])
#headers=list(df) # list column headers
#preds=list(df[headers].drop(droplist, axis=1))
#print(preds)
preds=list(df.drop(['utctime',pred], axis=1)) # type: ignore
print(preds)

## F-score
print("start fscore")
mdl=mdls_dir+mdl_name # type: ignore
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
ax.set_title(mdl_name) # type: ignore
ax.set_xscale('log')
plt.tight_layout()
f.savefig(res_dir+fscorepic, dpi=200) # type: ignore
#plt.show()
plt.clf(); plt.close('all')

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))



