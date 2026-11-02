import os,time,warnings,sys,json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless server
import matplotlib.pyplot as plt
import shap
warnings.filterwarnings("ignore")
# Fit SWE XGBoost from saved optuna params (NO optuna), variant: SF (reduced features)
# split by SNOW SEASON (1 Sep - 31 Aug), label = start year: 2018 -> 2018-09-01..2019-08-31 (winter 2018/19)
# runs Scandinavia / Alps / both; reads hyperparams_swe_sf_<domain>_<tag>.json
# works for both reg:squarederror and reg:tweedie params files
# usage: python fit_swe_sf.py <optuna_tag>

startTime = time.time()

variant='sf'              # <<< which feature-set variant; labels params loaded + outputs
optuna_tag=sys.argv[1]    # which saved params to load + output namespace

data_dir='/home/ubuntu/data/cryoscope/snowmodel/'
mod_dir=f'{data_dir}models'   # saved mdl + hyperparam json
res_dir=f'{data_dir}results'
os.makedirs(mod_dir,exist_ok=True); os.makedirs(res_dir,exist_ok=True)

SEED=99; N_JOBS=64; VERBOSE_ROUNDS=100
DEFAULT_N_ESTIMATORS=500   # only used if a params file has no saved tree count
# xgboost keys to pull out of the saved json (everything else in it is metadata)
XGB_KEYS=['objective','eval_metric','tree_method','num_parallel_tree','max_depth',
          'min_child_weight','gamma','subsample','colsample_bytree','learning_rate',
          'reg_alpha','reg_lambda','max_delta_step','max_leaves','grow_policy',
          'booster','sampling_method','max_bin',
          'tweedie_variance_power']   # needed for reg:tweedie, otherwise xgboost default 1.5 is used

# SF: reduced feature set (must match optuna_swe_sf.py)
cols_used=['utctime','latitude','longitude','SWE','height','slope','aspect',#'fch_2021',
           'tag','tcf',
           'swe_mean',#'swe_max','swe_min',
           'snowdepth_mean',#'snowdepth_max','snowdepth_min',
           #'es','pev','smlt','asn_00','asn_12','fal_00','fal_12','licd_00','licd_12','lict_00','lict_12','tsn_00','tsn_12',
           #'tp',
           'sf',#'e','slhf','sshf','ssr','ssrd','str','strd',
           #'sro','ro',
           'ssro',
           #'laihv_00',#'laihv_12',
           #'lailv_00',#'lailv_12',
           'rsn_00',#'rsn_12','sd_00','sd_12',
           'sde_00',#'sde_12',
           #'skt_00','skt_12',#'snowc_00','snowc_12',
           #'sp_00',#'sp_12','src_00','src_12',
           'stl1_00',#'stl1_12',
           #'swvl1_00',#'swvl1_12',
           #'swvl2_00','swvl2_12',#'swvl3_00','swvl3_12','swvl4_00','swvl4_12',
           #'t2_00',#'t2_12',
           #'td2_00',#'td2_12',
           #'u10_00',#'u10_12',
           #'v10_00',#'v10_12',
           #'q500_00',#'q500_12',
           #'t500_00',#'t500_12',
           #'u500_00',#'u500_12',
           #'v500_00',#'v500_12',
           #'z500_00',#'z500_12',
           'q700_00',#'q700_12',
           #'u700_00',#'u700_12',
           't700_00',#'t700_12',
           #'v700_00',#'v700_12',
           'z700_00',#'z700_12',
           #'q850_00',#'q850_12',
           #'u850_00',#'u850_12',
           #'t850_00',#'t850_12',
           #'v850_00',#'v850_12',
           #'z850_00',#'z850_12',
           #'sktmax_max','sktmin_min',
           # 'tmax_max','tmin_min'
           ]

input=f'{data_dir}training_all_scandi_alps_ext.csv.gz'
#input=f'{data_dir}sample1000_scandi_alps.csv.gz'   # testing
df_input=pd.read_csv(input,usecols=cols_used)

df_input['SWE']=df_input['SWE'].clip(lower=0.0)  # negative SWE -> 0 (tweedie requires target >= 0)
df_input['utctime']=pd.to_datetime(df_input['utctime'])
df_input['doy_cos']=np.cos(2*np.pi*df_input['utctime'].dt.dayofyear/365.25)
df_input['doy_sin']=np.sin(2*np.pi*df_input['utctime'].dt.dayofyear/365.25)

'''# ---- lapse-rate features: true lapse rates in K/km ----
def lapse_rate(t_up,t_low,z_up,z_low):
    """(t_up - t_low) / layer thickness in km; z is geopotential (m2/s2)."""
    thick_km=(df_input[z_up]-df_input[z_low])/9.80665/1000.0
    return (df_input[t_up]-df_input[t_low])/thick_km, thick_km
 
df_input['gamma700850'],thick=lapse_rate('t700_00','t850_00','z700_00','z850_00')
df_input['gamma500700'],thick=lapse_rate('t500_00','t700_00','z500_00','z700_00')
del thick
# t850, t500, z850 and z500 only served the lapse rates
df_input.drop(columns=['t850_00','t500_00','z850_00','z500_00'], inplace=True)
'''

# snow season: Sep-Dec belong to that year's season, Jan-Aug to the previous year's season
SNOW_START_MONTH=9
yr=df_input['utctime'].dt.year; mo=df_input['utctime'].dt.month
df_input['season']=np.where(mo>=SNOW_START_MONTH,yr,yr-1)
# cast every float column to float32 to cut memory; utctime + tag untouched
f32cols=df_input.select_dtypes(include='float').columns
df_input[f32cols]=df_input[f32cols].astype(np.float32)
print('tags present:',df_input['tag'].unique())
print('seasons present:',sorted(df_input['season'].unique().tolist()))

valid_s=[2018]   # validation snow season(s): 2018 = winter 2018/19; all other seasons are training
target=['SWE']; predictors_drop=['utctime','tag','season']
features=[c for c in df_input.columns if c not in target and c not in predictors_drop]
print(str(len(features))+' features | split by snow season, valid:',valid_s)
domains=['Scandinavia','Alps','both']


# ======================================================================
# loop over the three domains
# ======================================================================
for domain in domains:
    print('\n'+'='*70); print('DOMAIN:',domain,'| variant:',variant); domStart=time.time()

    # ---- load best params saved by the optuna script for this variant+domain ----
    hp_path=f'{mod_dir}/hyperparams_swe_{variant}_{domain}_{optuna_tag}.json'
    if not os.path.exists(hp_path):
        print('  no params file',hp_path,'- skipping'); continue
    with open(hp_path) as f: hp=json.load(f)
    params={k:hp[k] for k in XGB_KEYS if k in hp}
    params.update({'n_jobs':N_JOBS,'random_state':SEED})
    params.setdefault('eval_metric','rmse')   # learning curve below reads rmse
    objective=params.get('objective','reg:squarederror')
    log_link=objective in ('reg:tweedie','reg:gamma','count:poisson')
    if objective=='reg:tweedie' and 'tweedie_variance_power' not in params:
        print('  WARNING: reg:tweedie but no tweedie_variance_power in params file; xgboost default 1.5 used')
    # tree count: use the optuna best_iteration (0-based -> +1); no new early stopping
    if 'n_estimators_earlystop' in hp:   n_estimators=int(hp['n_estimators_earlystop'])+1
    elif 'n_estimators' in hp:           n_estimators=int(hp['n_estimators'])
    else:                                n_estimators=DEFAULT_N_ESTIMATORS
    params['n_estimators']=n_estimators
    print('  loaded params:',{k:params[k] for k in params if k not in ('n_jobs','random_state')})
    if 'features' in hp and list(hp['features'])!=features:
        print('  WARNING: feature list differs from the saved run; check column changes')

    # ---- filter domain + split by snow season ----
    df_dom=df_input if domain=='both' else df_input[df_input['tag']==domain]
    if len(df_dom)==0: print('  no rows for',domain,'- skipping'); continue
    vm=df_dom['season'].isin(valid_s).to_numpy()
    train_set=df_dom[~vm].reset_index(drop=True); valid_set=df_dom[vm].reset_index(drop=True)
    if len(train_set)==0 or len(valid_set)==0: print('  empty split - skipping'); continue
    train_seasons=sorted(train_set['season'].unique().tolist())
    print('  train seasons:',train_seasons,'| valid seasons:',valid_s)
    target_train=train_set[target]; features_train=train_set[features].astype(np.float32)
    target_valid=valid_set[target]; features_valid=valid_set[features].astype(np.float32)
    print('  train rows',len(features_train),' valid rows',len(features_valid))

    # ---- EDA: train vs valid ----
    yt=train_set['SWE'].to_numpy(); yv=valid_set['SWE'].to_numpy()
    print('  rows per tag, train:',train_set['tag'].value_counts().to_dict())
    print('  rows per tag, valid:',valid_set['tag'].value_counts().to_dict())
    print('  SWE stats           train        valid')
    for name,fn in [('mean',np.mean),('median',np.median),('p95',lambda a:np.percentile(a,95)),('max',np.max)]:
        print(f'    {name:<8} {fn(yt):12.3f} {fn(yv):12.3f}')
    print(f'    frac0    {(yt==0).mean():12.3f} {(yv==0).mean():12.3f}')
    print(f'    snowon   {yt[yt>0].mean():12.3f} {yv[yv>0].mean():12.3f}')
    print(f'    valid SWE max {yv.max():.1f} vs train SWE max {yt.max():.1f}'
          + ('  <-- valid exceeds train range' if yv.max()>yt.max() else ''))
    print('    months train:',sorted(train_set['utctime'].dt.month.unique().tolist()),
          '| valid:',sorted(valid_set['utctime'].dt.month.unique().tolist()))

    plt.figure(figsize=(7,4))
    plt.hist(yt[yt>0],bins=60,alpha=0.5,density=True,label='train')
    plt.hist(yv[yv>0],bins=60,alpha=0.5,density=True,label='valid')
    plt.yscale('log'); plt.xlabel('SWE (snow-on only)'); plt.ylabel('density (log)')
    plt.legend(); plt.title(f'SWE distribution {variant} {domain}'); plt.tight_layout()
    plt.savefig(f'{res_dir}/eda_swe_hist_{variant}_{domain}_{optuna_tag}.png',dpi=120); plt.close()

    mth_t=train_set.groupby(train_set['utctime'].dt.month)['SWE'].mean()
    mth_v=valid_set.groupby(valid_set['utctime'].dt.month)['SWE'].mean()
    plt.figure(figsize=(7,4))
    plt.plot(mth_t.index,mth_t.values,'-o',label='train'); plt.plot(mth_v.index,mth_v.values,'-o',label='valid')
    plt.xlabel('month'); plt.ylabel('mean SWE'); plt.legend()
    plt.title(f'seasonal cycle {variant} {domain}'); plt.tight_layout()
    plt.savefig(f'{res_dir}/eda_swe_seasonal_{variant}_{domain}_{optuna_tag}.png',dpi=120); plt.close()

    # ---- fit model with loaded params (fixed tree count, no early stopping) ----
    print('  fitting with n_estimators =',n_estimators,'| objective =',objective)
    final_model=xgb.XGBRegressor(**params)
    final_model.fit(features_train,target_train,eval_set=[(features_valid,target_valid)],verbose=VERBOSE_ROUNDS)

    # ---- learning curve (valid metric per boosting round) ----
    ev=final_model.evals_result()['validation_0']
    metric_name='rmse' if 'rmse' in ev else list(ev.keys())[0]
    curve=ev[metric_name]
    plt.figure(figsize=(10,5)); plt.plot(range(len(curve)),curve,linewidth=1.2)
    plt.xlabel('boosting round'); plt.ylabel(f'valid {metric_name}'); plt.grid(True,alpha=0.3)
    plt.title(f'learning curve ({variant} {domain}, {optuna_tag})'); plt.tight_layout()
    lc=f'{res_dir}/learning_curve_{variant}_{domain}_{optuna_tag}'
    plt.savefig(lc+'.png',dpi=200); plt.savefig(lc+'.svg'); plt.close()

    # ---- metrics: RMSE, MAE, bias, R2 (all + snow-on) ----
    pred=final_model.predict(features_valid); yv=valid_set['SWE'].to_numpy()
    rmse_all=np.sqrt(mean_squared_error(yv,pred)); mae_all=mean_absolute_error(yv,pred)
    bias=float(np.mean(pred-yv)); r2_all=r2_score(yv,pred)
    mask=yv>0
    rmse_snow=np.sqrt(mean_squared_error(yv[mask],pred[mask])); mae_snow=mean_absolute_error(yv[mask],pred[mask])
    r2_snow=r2_score(yv[mask],pred[mask])
    frac_small_nosnow=float((pred[~mask]<1).mean()) if (~mask).any() else float('nan')
    frac_neg=float((pred<0).mean())
    print(f'  VALID  RMSE={rmse_all:.3f}  MAE={mae_all:.3f}  bias={bias:.3f}  R2={r2_all:.3f}  (all rows)')
    print(f'  VALID  RMSE={rmse_snow:.3f}  MAE={mae_snow:.3f}  R2={r2_snow:.3f}          (snow-on rows only)')
    print(f'  VALID  frac pred<1mm where obs=0: {frac_small_nosnow:.3f} | frac pred<0: {frac_neg:.4f}')
    metrics_domain={}
    for dom in valid_set['tag'].unique():
        m=(valid_set['tag']==dom).to_numpy()
        metrics_domain[dom]=float(np.sqrt(mean_squared_error(yv[m],pred[m])))
        print(f'  VALID  RMSE={metrics_domain[dom]:.3f}  ({dom}, {m.sum()} rows)')

    # obs vs pred scatter
    n=min(30000,len(yv)); idx=np.random.RandomState(SEED).choice(len(yv),n,replace=False)
    plt.figure(figsize=(5,5)); plt.scatter(yv[idx],pred[idx],s=3,alpha=0.3)
    lim=max(yv.max(),pred.max()); plt.plot([0,lim],[0,lim],'k--',lw=1)
    plt.xlabel('observed SWE'); plt.ylabel('predicted SWE')
    plt.title(f'valid obs vs pred ({variant} {domain}, {optuna_tag})'); plt.tight_layout()
    plt.savefig(f'{res_dir}/valid_scatter_{variant}_{domain}_{optuna_tag}.png',dpi=120); plt.close()

    # ---- save model + fitted params/metrics ----
    model_path=f'{mod_dir}/mdl_swe_{variant}_{domain}_{optuna_tag}.json'
    final_model.save_model(model_path); print('  saved model ->',model_path)
    out={k:params[k] for k in XGB_KEYS if k in params}
    out.update({"variant":variant,"domain":domain,"n_estimators_used":int(n_estimators),
                "n_features":len(features),"features":features,
                "split_by":"snowseason","snow_start_month":SNOW_START_MONTH,
                "train_seasons":train_seasons,"valid_seasons":valid_s,
                "valid_rmse_all":float(rmse_all),"valid_mae_all":float(mae_all),"valid_r2_all":float(r2_all),"valid_bias":bias,
                "valid_rmse_snowon":float(rmse_snow),"valid_mae_snowon":float(mae_snow),"valid_r2_snowon":float(r2_snow),
                "valid_frac_pred_lt1mm_obs0":frac_small_nosnow,"valid_frac_pred_neg":frac_neg,
                "valid_rmse_by_domain":metrics_domain,"input_file":os.path.basename(input),
                "params_from":os.path.basename(hp_path),"fit_without_optuna":True,"early_stopping_used":False,"optuna_tag":optuna_tag})
    fit_path=f'{mod_dir}/hyperparams_swe_fitted_{variant}_{domain}_{optuna_tag}.json'
    with open(fit_path,'w') as f: json.dump(out,f,indent=2)
    print('  saved fitted params ->',fit_path)

    # gain importance
    gain=final_model.get_booster().get_score(importance_type='gain')
    pd.DataFrame([(k,gain.get(k,0.0)) for k in features],columns=['feature','gain']
                ).sort_values('gain',ascending=False).to_csv(
                f'{res_dir}/gain_importance_{variant}_{domain}_{optuna_tag}.csv',index=False)

    # ---- SHAP (TreeExplainer) on a sample of valid ----
    # with a log-link objective (reg:tweedie) SHAP values are in log(SWE) margin units, not mm
    shap_unit='log-SWE margin' if log_link else 'SWE mm'
    ns=min(20000,len(features_valid)); shap_X=features_valid.sample(n=ns,random_state=SEED)
    explainer=shap.TreeExplainer(final_model); shap_values=explainer.shap_values(shap_X)
    shap.summary_plot(shap_values,shap_X,max_display=25,show=False)
    plt.title(f'SHAP ({shap_unit}) {variant} {domain}')
    plt.tight_layout(); plt.savefig(f'{res_dir}/shap_beeswarm_{variant}_{domain}_{optuna_tag}.png',dpi=120); plt.close()
    shap.summary_plot(shap_values,shap_X,plot_type='bar',max_display=25,show=False)
    plt.title(f'mean |SHAP| ({shap_unit}) {variant} {domain}')
    plt.tight_layout(); plt.savefig(f'{res_dir}/shap_bar_{variant}_{domain}_{optuna_tag}.png',dpi=120); plt.close()
    shap_imp=pd.DataFrame({'feature':features,'mean_abs_shap':np.abs(shap_values).mean(axis=0)}
                         ).sort_values('mean_abs_shap',ascending=False)
    shap_imp.to_csv(f'{res_dir}/shap_importance_{variant}_{domain}_{optuna_tag}.csv',index=False)
    for feat in shap_imp['feature'].head(4):
        shap.dependence_plot(feat,shap_values,shap_X,show=False)
        plt.title(f'SHAP dependence ({shap_unit}): {feat}')
        plt.tight_layout(); plt.savefig(f'{res_dir}/shap_dep_{feat}_{variant}_{domain}_{optuna_tag}.png',dpi=120); plt.close()
    print('  SHAP + plots ->',f'{res_dir}/*_{variant}_{domain}_{optuna_tag}.*')
    print('  domain time (min): %.2f'%((time.time()-domStart)/60))

print('\nAll domains done. Total execution time in minutes: %.2f'%((time.time()-startTime)/60))