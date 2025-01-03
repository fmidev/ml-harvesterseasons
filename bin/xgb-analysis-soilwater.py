import xgboost as xgb
import time
#import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

startTime=time.time()

mdls_dir='/home/ubuntu/data/ML/models/soilwater/'
res_dir='/home/ubuntu/data/ML/results/soilwater/figures/'

# K-Fold fitted models
'''
fname='model-1000stations-era5params-*.txt' # pois: utctime
'''
# xgb-fit without gridsearchCV
#fname=sys.argv[1]
#fname = 'mdl_swi2_2015-2022_10000points-13.txt'
#fname='mdl_swi2_2015-2022_63287points-1.txt'
fname='mdl_swi2_2015-2022_10000points-noRunsums.txt'
# Predictors in the fitted mdl
preds=['evap',
       #'evap15d',
        'laihv-00','lailv-00',
        'ro',
        #'ro15d',
        'rsn-00','sd-00',
        'slhf','sshf','ssr','ssrd',
        'stl1-00','str','swvl2-00','t2-00','td2-00',
        'tp',
        #'tp15d',
        'swi2clim',
        'lake_cover','cvh','cvl','lake_depth','land_cover','soiltype','urban_cover','tvh','tvl',
        'TH_LAT','TH_LONG','DTM_height','DTM_slope','DTM_aspect',
        'clay_0-5cm','clay_15-30cm','clay_5-15cm',
        'sand_0-5cm','sand_15-30cm','sand_5-15cm',
        'silt_0-5cm','silt_15-30cm','silt_5-15cm',
        'soc_0-5cm','soc_15-30cm','soc_5-15cm',
        'dayOfYear'
]
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
ax.set_title(fname)
ax.set_xscale('log')
plt.tight_layout()
#f.savefig('Fscore.pdf')
f.savefig(res_dir+'Fscore_swi2-10000points-new.png', dpi=200)
#plt.show()
plt.clf(); plt.close('all')
'''
## SHAP analysis
# prepare X_test (same as for the trained mdl)
print("prepare x_test for shap")
test_years=[2003, 2009, 2011, 2015]
train_years=train_y=[2000, 2001, 2002, 2004, 2005, 2006, 2007, 2008, 2010, 2012, 2013, 2014, 2016, 2017, 2018, 2019, 2020, 2021]
cols_own=['utctime','t2m-K','td-K','msl-Pa','u10-ms','v10-ms','z-m','sdor-m','slor','anor-rad',
'tcc-0to1','sd-kgm2','rsn-kgm2','stl1-K','mx2t-K','mn2t-K',
'lsm-0to1','HGHT','LAT','LON','t850-K-00','t850-K-12','t700-K-00',
't700-K-12','t500-K-00','t500-K-12','q850-kgkg-00','q850-kgkg-12','q700-kgkg-00','q700-kgkg-12',
'q500-kgkg-00','q500-kgkg-12','u850-ms-00','u850-ms-12','u700-ms-00','u700-ms-12','u500-ms-00','u500-ms-12',
'v850-ms-00','v850-ms-12','v700-ms-00','v700-ms-12','v500-ms-00','v500-ms-12','z850-m-00','z850-m-12',
'z700-m-00','z700-m-12','z500-m-00','z500-m-12','kx-00','kx-12','tp-m','e-m','tsr-jm2','fg10-ms','slhf-Jm2',
'sshf-Jm2','ro-m','sf-m','dtw',
#slope,aspect,tpi,tri,
'dtr','dtl','dto','fch',
'RR'
]
df=pd.read_csv('/lustre/tmp/strahlen/mlbias/data/era5-SL-PL-24H_eobs_terrain_1995-2020_all.csv',usecols=cols_own)
# Fix E-OBS station rainfall observations (rain > 12 mm: x1.4, other x1.1)
df['RR']=df['RR']/10000.0 # units from x0.1mm to m
df.loc[ df['RR'] >= 0.012, 'RR'] = df['RR']*1.4
df.loc[ df['RR'] < 0.012, 'RR'] = df['RR']*1.1

df['utctime']=pd.to_datetime(df['utctime'])

test_stations=pd.DataFrame()
for y in test_years:
        test_stations=pd.concat([test_stations,df[df['utctime'].dt.year == y]],ignore_index=True)
test_stations=test_stations.dropna(axis=0,how='any')
test_stations=test_stations.drop(columns=['utctime'])
X_test=test_stations[preds]

print("start shap (warning: takes time)")
for i,mdl in enumerate(models):
    explainer=shap.TreeExplainer(mdl)
    shap_values=explainer.shap_values(X_test)
    data=pd.DataFrame(np.abs(shap_values).mean(axis=0),index=list(X_test.columns))
    data=data.sort_values(by=0,ascending=False)
    sorted_names,sorted_SHAP=np.flipud(list(data.index)),np.flipud(data.values.squeeze())

    all_scores2=pd.DataFrame(columns=["Model","Predictor variable","Mean SHAP"])
    i=0
    row=0
    for ft,sh in zip(sorted_names,sorted_SHAP):
        print(i+1,ft,sh)
        all_scores2.loc[row]=(i+1,ft,sh);row+=1

    sns.set_theme(style='whitegrid')
    f,ax=plt.subplots(1,1,figsize=(6,10))
    mean_scores=all_scores2.groupby('Predictor variable').mean()['Mean SHAP'].sort_values()
    mean_scores.plot.barh(ax=ax,width=0.3,legend=False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_title(fname)
    plt.tight_layout()
    f.savefig(res_dir+"SHAPanalysis195sta2000-2020.png",dpi=200)
    plt.clf(); plt.close('all')
'''
executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))



