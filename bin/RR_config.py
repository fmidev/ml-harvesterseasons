### Precipitation configuration file for XGBoost fitting
### xgb-fit-RR.py
data_dir='/home/ubuntu/data/ML/training-data/RR/' # training data path
fname=data_dir+'RR_eobs+era5_training_4582points_2000-2020.csv.gz' # training data file
mod_dir='/home/ubuntu/data/ML/models/RR/' # saved XGBoost models
name='RR_eobs+era5_XGB_2000-2020_4582-1'
mod_name=mod_dir+name+'.json' # model name
res_dir='/home/ubuntu/data/ML/results/RR/' # figures etc
logfile=res_dir+name+'.log' # log file
predictand1='RR'
predictand2='WeightRR'
cols_own=[
    'STAID','DATE','RR','LAT','LON',
    #'HGHT',
    'anor','e','ewss','fg10','kx-00','kx-12',
    'latitude','lsm','mn2t','msl-00','msl-12',
    'mx2t','nsss','q500-00','q500-12','q700-00',
    'q700-12','q850-00','sdor','slhf','slor','ssr',
    'ssrd','sshf','str','strd','swvl1','swvl2',
    'swvl3','swvl4','t2-00','t2-12','t500-00',
    't500-12','t700-00','t700-12','t850-00','t850-12',
    'tcc-00','tcc-12','tclw','tcwv','tp','td2-00','td2-12','ttr',
    'u10-00','u10-12','v10-00','v10-12','u500-00','u500-12',
    'u700-00','u700-12','u850-00','u850-12','v500-00','v500-12',
    'v700-00','v700-12','v850-00','v850-12','z','z500-00','z500-12',
    'z700-00','z700-12','z850-00','z850-12'
]
test_y=[2003, 2009, 2011, 2015]
train_y=[2000, 2001, 2002, 2004, 2005, 2006, 2007, 2008, 2010, 2012, 2013, 2014, 2016, 2017, 2018, 2019, 2020]
# XGBoost hyperparameters
nstm=857
lrte=0.044
max_depth=9
subsample=0.96
colsample_bytree=0.37
num_parallel_tree=2
al=0.72
