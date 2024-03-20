import xgboost as xgb
import glob, sklearn, time
#import shap
import functions as fcts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

startTime=time.time()

mdls_dir='/home/ubuntu/data/ML/models/snowdepth/' 
res_dir='/home/ubuntu/data/ML/results/snowdepth/figures/'

# xgb-fit without gridsearchCV
fname='mdl_SD_2006-2020_9790points-2.txt'
figname='Fscore_SD_2006-2020-9790points-3.png'

# Predictors in the fitted mdl
# tp left out (NaNs, data missing) 
preds=[
    'latitude','longitude',
    'slhf','sshf','ssrd','strd','str','ssr','sf',
    'laihv-00','lailv-00','sd-00','rsn-00',
    'stl1-00','swvl2-00','t2-00','td2-00','u10-00','v10-00',
    'ro','evap','swe_clms',
    'dayOfYear']
print(len(preds))

## F-score
print("start fscore")
mdl=mdls_dir+fname
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
ax.set_ylabel('predictor')  # y label added
ax.set_title(fname)
ax.set_xscale('log')
plt.tight_layout()
#f.savefig('Fscore.pdf')
f.savefig(res_dir+figname, dpi=200)
#plt.show()
plt.clf(); plt.close('all')

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))



