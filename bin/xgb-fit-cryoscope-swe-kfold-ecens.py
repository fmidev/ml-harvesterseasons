import os,time,warnings,sys,json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import xgboost as xgb
import numpy as np
warnings.filterwarnings("ignore")
# Leave-one-season-out k-fold CV for SWE XGBoost from saved optuna params (NO optuna, NO shap/plots)
# variant: EC-ENS (full features); runs Scandinavia / Alps / both; metrics only
# each fold: one season = validation, all other data in the domain = training
# works for both reg:squarederror and reg:tweedie params files
# resumes: folds already in the results csv are skipped
# usage: python kfold_swe_ecens.py <optuna_tag> [Scandinavia|Alps|both]

startTime = time.time()

variant='ec-ens'
optuna_tag=sys.argv[1]
domains=[sys.argv[2]] if len(sys.argv)>2 else ['Scandinavia','Alps','both']

# --- fold definition ---
SPLIT_BY='snowyear'      # 'snowyear' = Sep..Aug, each winter kept whole (label = start year, 2018 -> 2018-09..2019-08)
                         # 'calyear'  = Jan..Dec, same split as fit_swe_ecens.py (cuts winters in two)
SNOW_START_MONTH=9
MIN_DAYS=200             # skip partial seasons at the ends of the record as validation folds

data_dir='/home/ubuntu/data/cryoscope/snowmodel/'
mod_dir=f'{data_dir}models'
res_dir=f'{data_dir}results'
os.makedirs(res_dir,exist_ok=True)

SEED=99; N_JOBS=64
DEFAULT_N_ESTIMATORS=500
XGB_KEYS=['objective','eval_metric','tree_method','num_parallel_tree','max_depth',
          'min_child_weight','gamma','subsample','colsample_bytree','learning_rate',
          'reg_alpha','reg_lambda','max_delta_step','max_leaves','grow_policy',
          'booster','sampling_method','max_bin',
          'tweedie_variance_power']

# EC-ENS: full feature set (must match optuna_swe_ecens.py / fit_swe_ecens.py)
cols_used=['utctime','latitude','longitude','SWE','height','slope','aspect',#'fch_2021',
           'tag','tcf',
           'swe_mean','swe_max','swe_min',
           'snowdepth_mean','snowdepth_max','snowdepth_min',
           'e',#'es',
           'pev','ro','sf','slhf','smlt','sro','sshf','ssr','ssrd','ssro','str','strd',
           'tp','asn_00',
           'laihv_00',
           'lailv_00',
           'lict_00',
           'rsn_00',
           'sde_00',
           'skt_00',
           'sp_00',
           'src_00',
           'stl1_00',
           'swvl1_00',
           'swvl2_00',
           't2_00',
           'td2_00',
           'tsn_00',
           'u10_00',
           'v10_00',
           'q500_00','t500_00','u500_00','v500_00','z500_00',
           'q700_00','u700_00','t700_00','v700_00','z700_00',
           'q850_00','u850_00','t850_00','v850_00','z850_00',
           ]
input=f'{data_dir}training_all_scandi_alps_ext.csv.gz'
#input=f'{data_dir}sample1000_scandi_alps.csv.gz'   # testing
df_input=pd.read_csv(input,usecols=cols_used)

df_input['SWE']=df_input['SWE'].clip(lower=0.0)
df_input['utctime']=pd.to_datetime(df_input['utctime'])
df_input['doy_cos']=np.cos(2*np.pi*df_input['utctime'].dt.dayofyear/365.25)
df_input['doy_sin']=np.sin(2*np.pi*df_input['utctime'].dt.dayofyear/365.25)
f32cols=df_input.select_dtypes(include='float').columns
df_input[f32cols]=df_input[f32cols].astype(np.float32)

yr=df_input['utctime'].dt.year; mo=df_input['utctime'].dt.month
if SPLIT_BY=='snowyear': df_input['fold']=np.where(mo>=SNOW_START_MONTH,yr,yr-1)
else:                    df_input['fold']=yr

target='SWE'; predictors_drop=['utctime','tag','fold']
features=[c for c in df_input.columns if c!=target and c not in predictors_drop]
print(str(len(features))+' features | split by',SPLIT_BY)

# arrays once; folds index into these
X_all=df_input[features].to_numpy(np.float32)
y_all=df_input[target].to_numpy(np.float32)
tag_all=df_input['tag'].to_numpy()
fold_all=df_input['fold'].to_numpy()

res_path=f'{res_dir}/kfold_metrics_{variant}_{SPLIT_BY}_{optuna_tag}.csv'
done=set()
if os.path.exists(res_path):
    prev=pd.read_csv(res_path); done=set(zip(prev['domain'],prev['valid_season']))
    print('resuming,',len(done),'folds already done ->',res_path)

def metrics(yv,pred):
    snow=yv>0
    m={'rmse_all':float(np.sqrt(mean_squared_error(yv,pred))),'mae_all':float(mean_absolute_error(yv,pred)),
       'bias':float(np.mean(pred-yv)),'r2_all':float(r2_score(yv,pred))}
    if snow.sum()>1:
        m.update({'rmse_snowon':float(np.sqrt(mean_squared_error(yv[snow],pred[snow]))),
                  'mae_snowon':float(mean_absolute_error(yv[snow],pred[snow])),
                  'r2_snowon':float(r2_score(yv[snow],pred[snow]))})
    else: m.update({'rmse_snowon':np.nan,'mae_snowon':np.nan,'r2_snowon':np.nan})
    m['frac_pred_lt1mm_obs0']=float((pred[~snow]<1).mean()) if (~snow).any() else np.nan
    m['frac_pred_neg']=float((pred<0).mean())
    return m

# ======================================================================
for domain in domains:
    print('\n'+'='*70); print('DOMAIN:',domain,'| variant:',variant)

    hp_path=f'{mod_dir}/hyperparams_swe_{variant}_{domain}_{optuna_tag}.json'
    if not os.path.exists(hp_path):
        print('  no params file',hp_path,'- skipping'); continue
    with open(hp_path) as f: hp=json.load(f)
    params={k:hp[k] for k in XGB_KEYS if k in hp}
    params.update({'n_jobs':N_JOBS,'random_state':SEED})
    if   'n_estimators_earlystop' in hp: n_estimators=int(hp['n_estimators_earlystop'])+1
    elif 'n_estimators' in hp:           n_estimators=int(hp['n_estimators'])
    else:                                n_estimators=DEFAULT_N_ESTIMATORS
    params['n_estimators']=n_estimators
    print('  objective =',params.get('objective','reg:squarederror'),'| n_estimators =',n_estimators)

    dmask=np.ones(len(y_all),bool) if domain=='both' else (tag_all==domain)
    if not dmask.any(): print('  no rows for',domain,'- skipping'); continue
    days=df_input.loc[dmask].groupby('fold')['utctime'].nunique()
    seasons=[int(s) for s,n in days.items() if n>=MIN_DAYS]
    print('  seasons (days):',days.to_dict()); print('  validation folds:',seasons)

    for vs in seasons:
        if (domain,vs) in done: print(f'  season {vs}: already done'); continue
        foldStart=time.time()
        vmask=dmask&(fold_all==vs); tmask=dmask&(fold_all!=vs)
        Xt,yt=X_all[tmask],y_all[tmask]; Xv,yv=X_all[vmask],y_all[vmask]

        model=xgb.XGBRegressor(**params)
        model.fit(Xt,yt,verbose=False)
        pred=model.predict(Xv)

        row={'domain':domain,'valid_season':vs,'split_by':SPLIT_BY,'n_train':int(tmask.sum()),'n_valid':int(vmask.sum()),
             'valid_swe_mean':float(yv.mean()),'valid_swe_max':float(yv.max()),'train_swe_max':float(yt.max()),
             'valid_frac0':float((yv==0).mean())}
        row.update(metrics(yv,pred))
        if domain=='both':
            for dom in np.unique(tag_all[vmask]):
                m=(tag_all[vmask]==dom)
                row[f'rmse_{dom}']=float(np.sqrt(mean_squared_error(yv[m],pred[m])))
                row[f'r2_{dom}']=float(r2_score(yv[m],pred[m]))
        row['fit_min']=round((time.time()-foldStart)/60,2)

        print(f"  season {vs}: RMSE={row['rmse_all']:.2f} MAE={row['mae_all']:.2f} bias={row['bias']:.2f} "
              f"R2={row['r2_all']:.3f} | snow-on RMSE={row['rmse_snowon']:.2f} R2={row['r2_snowon']:.3f} "
              f"| valid max {row['valid_swe_max']:.0f} (train max {row['train_swe_max']:.0f}) | {row['fit_min']} min")
        # append per fold so a crash keeps finished folds
        pd.DataFrame([row]).to_csv(res_path,mode='a',header=not os.path.exists(res_path),index=False)

# ======================================================================
# summary over folds
# ======================================================================
if os.path.exists(res_path):
    res=pd.read_csv(res_path)
    cols=['rmse_all','mae_all','bias','r2_all','rmse_snowon','mae_snowon','r2_snowon']
    rows=[]
    print('\n'+'='*70); print('SUMMARY (',SPLIT_BY,')')
    for domain,g in res.groupby('domain'):
        g=g.sort_values('valid_season')
        best=g.loc[g['r2_all'].idxmax()]; worst=g.loc[g['r2_all'].idxmin()]
        print(f'\n  {domain}: {len(g)} folds')
        print(g[['valid_season','rmse_all','mae_all','bias','r2_all','rmse_snowon','r2_snowon','valid_swe_max']
               ].to_string(index=False,float_format=lambda v:f'{v:.3f}'))
        for c in cols:
            print(f'    {c:<12} mean {g[c].mean():9.3f}  std {g[c].std():8.3f}  min {g[c].min():9.3f}  max {g[c].max():9.3f}')
        print(f"    best season (R2): {int(best['valid_season'])} ({best['r2_all']:.3f}) | "
              f"worst: {int(worst['valid_season'])} ({worst['r2_all']:.3f})")
        s={'domain':domain,'n_folds':len(g),'best_season':int(best['valid_season']),'worst_season':int(worst['valid_season'])}
        for c in cols: s.update({f'{c}_mean':g[c].mean(),f'{c}_std':g[c].std(),f'{c}_min':g[c].min(),f'{c}_max':g[c].max()})
        rows.append(s)
    sum_path=f'{res_dir}/kfold_summary_{variant}_{SPLIT_BY}_{optuna_tag}.csv'
    pd.DataFrame(rows).to_csv(sum_path,index=False); print('\n  summary ->',sum_path)

print('\nTotal execution time in minutes: %.2f'%((time.time()-startTime)/60))