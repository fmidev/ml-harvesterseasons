import os,optuna,time,warnings,sys,json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless server
import matplotlib.pyplot as plt
import shap
warnings.filterwarnings("ignore")
# XGBoost + Optuna tuning for SWE for Alps and Scandinavia (20 000 training locations)
# Objective: reg:tweedie (log link -> predictions always > 0, handles many zeros)
# split by SNOW SEASON (1 Sep - 31 Aug), label = start year: 2018 -> 2018-09-01..2019-08-31 (winter 2018/19)
# MULTI-OBJECTIVE tuning: minimize valid RMSE AND maximize valid KGE (KGE penalizes too small variability -> high SWE)
# optuna returns a Pareto front instead of one best trial; final trial is picked with RMSE_TOL below
# NOTE: use a NEW optuna_tag, an old single-objective study with the same name must not be loaded

startTime = time.time()

variant='cryoscope-SWE'
optuna_tag=sys.argv[1]

data_dir='/home/ubuntu/data/cryoscope/snowmodel/'
mod_dir=f'{data_dir}models'
res_dir=f'{data_dir}results'
optuna_dir='/home/ubuntu/data/ML/' # optuna storage
os.makedirs(mod_dir,exist_ok=True)
os.makedirs(res_dir,exist_ok=True)
os.chdir(optuna_dir)

N_TRIALS=50            # number of optuna trials
N_ESTIMATORS_CAP=1500  # fixed upper bound; early stopping picks the real count
EARLY_STOP=50
SEED=99
VERBOSE_ROUNDS=100     # xgboost prints valid RMSE every this many boosting rounds (0/False to silence)
OBJECTIVE='reg:tweedie'
RMSE_TOL=0.05          # final trial: highest KGE on the Pareto front with RMSE at most 5 % above the front's best RMSE
print('optuna version:',optuna.__version__)  # multi-objective TPE needs optuna >= 3.0
DROP_LATLON=False
SD_CHANGE_DAYS=8       # window (days) for the sd_00 change feature: sd_00(t) minus sd_00(t-8)

# Kling-Gupta efficiency: correlation, variability ratio and bias ratio in one score (1 = perfect)
def kge(o,p):
    o=np.asarray(o,dtype=np.float64).ravel(); p=np.asarray(p,dtype=np.float64).ravel()
    r=np.corrcoef(o,p)[0,1]; a=p.std()/o.std(); b=p.mean()/o.mean()
    return 1-np.sqrt((r-1)**2+(a-1)**2+(b-1)**2)

# Read in training data
cols_used=['utctime','latitude','longitude',
           'SWE','height','slope','aspect',
           'tag','tcf',#'swe_mean',#'swe_max',
           #'snowdepth_mean',
           'tp','sf','slhf','str','ssr','sshf',
           'ro',
           'rsn_00',
           'sd_00', # snow depth m of water equivalent
           'stl1_00',
           't2_00','tmax_max', # these used in FIN multi, below this for a model performance check with all features (64)
            #'td2_00',#'td2_12',
           #'u10_00',#'u10_12',
           #'v10_00',#'v10_12',
           #'q500_00',#'q500_12',
           #'t500_00',#'t500_12',
           #'u500_00',#'u500_12',
           #'v500_00',#'v500_12',
           #'z500_00',#'z500_12',
           #'q700_00',#'q700_12',
           #'u700_00',#'u700_12',
           #'t700_00',#'t700_12',
           #'v700_00',#'v700_12',
           #'z700_00',#'z700_12',
           #'q850_00',#'q850_12',
           #'u850_00',#'u850_12',
           #'t850_00',#'t850_12',
           #'v850_00',#'v850_12',
          # 'z850_00',#'z850_12',
          # 'tmin_min',
          # 'es','pev','smlt','e',
          #  'asn_00',#'asn_12','fal_00','fal_12','licd_00','licd_12','lict_00','lict_12',
          #  'tsn_00',#'tsn_12',
          # 'snowdepth_max','snowdepth_min',
          # 'swe_min',
          #  'skt_00',#'skt_12',#'snowc_00','snowc_12',
          # 'sp_00',#'sp_12',
          # 'src_00',#'src_12',
          # 'ssro','sro',
          # 'laihv_00','lailv_00',
          # 'swvl1_00','swvl2_00',
           ]



input=f'{data_dir}training_all_scandi_alps_ext.csv.gz'
df_input=pd.read_csv(input,usecols=cols_used)

# change negative SWE values to 0.0 (tweedie requires target >= 0)
df_input['SWE'] = df_input['SWE'].clip(lower=0.0)
#df_input['swe_mean'] = df_input['swe_mean'].clip(lower=0.0) # just in case
#df_input['swe_max'] = df_input['swe_max'].clip(lower=0.0)

# add day of year cosine and sine as columns
df_input['utctime'] = pd.to_datetime(df_input['utctime'])
df_input['doy_cos'] = np.cos(2 * np.pi * df_input['utctime'].dt.dayofyear / 365.25)
df_input['doy_sin'] = np.sin(2 * np.pi * df_input['utctime'].dt.dayofyear / 365.25)

# lag features at the SAME location, computed on the continuous series BEFORE the snow-season split:
#   sd_00_lag1: previous day's sd_00; NaN if that day is missing
#   sd_00_d8:   sd_00(t) minus sd_00(t-8), the 8-day net change (build-up > 0, melt < 0); NaN if day t-8 is not exactly 8 rows back
df_input['date']=df_input['utctime'].dt.normalize()
df_input=df_input.sort_values(['latitude','longitude','date']).reset_index(drop=True)
loc=df_input.groupby(['latitude','longitude'],sort=False)
prev_sde=loc['sd_00'].shift(1)
prev_date=loc['date'].shift(1)
df_input['sd_00_lag1']=prev_sde.where(df_input['date']-prev_date==pd.Timedelta(days=1))
sd_k=loc['sd_00'].shift(SD_CHANGE_DAYS)
date_k=loc['date'].shift(SD_CHANGE_DAYS)
df_input[f'sd_00_d{SD_CHANGE_DAYS}']=(df_input['sd_00']-sd_k).where(df_input['date']-date_k==pd.Timedelta(days=SD_CHANGE_DAYS))
df_input=df_input.drop(columns='date')
del loc,prev_sde,prev_date,sd_k,date_k
print('duplicate location-days:',int(df_input.duplicated(['latitude','longitude','utctime']).sum()))
print('sd_00_lag1 NaN fraction:',round(float(df_input['sd_00_lag1'].isna().mean()),4))
print(f'sd_00_d{SD_CHANGE_DAYS} NaN fraction:',round(float(df_input[f'sd_00_d{SD_CHANGE_DAYS}'].isna().mean()),4))

# snow season: Sep-Dec belong to that year's season, Jan-Aug to the previous year's season
SNOW_START_MONTH=9
yr=df_input['utctime'].dt.year; mo=df_input['utctime'].dt.month
df_input['season']=np.where(mo>=SNOW_START_MONTH,yr,yr-1)
# cast every float column to float32 to cut memory; utctime + tag untouched
f32cols=df_input.select_dtypes(include='float').columns
df_input[f32cols]=df_input[f32cols].astype(np.float32)
print(df_input)

# split by snow season (same across all domains): valid_s is validation, every other season is training
# 2018 = low-snow winter 2018/19, 2021 = high-snow winter 2021/22 -> tuning sees both ends
valid_s=[2018,2021]

# feature list is identical across domains (only the rows differ)
target=['SWE'] # snow water equivalent from Snowcap
predictors_drop=['utctime','tag','season']  # datetime, domain tag and season label never go in as features
# latitude and longitude stay in the data (sorting, per-location lag, duplicate check) but are not features:
# the model has to learn SWE from terrain and weather, not from where a point is

if DROP_LATLON:
    predictors_drop+=['latitude','longitude']
features=[c for c in df_input.columns if c not in target and c not in predictors_drop]
print(str(len(features))+' features | split by snow season, valid:',valid_s)
print('features:',features)
assert not DROP_LATLON or not {'latitude','longitude'}&set(features), 'latitude / longitude still in features'

# Run only for both version (tag column holds 'Scandinavia' / 'Alps')
domains=['both']

# Optuna objective (reads the current domain's train/valid globals set in the loop)
# returns TWO values: valid RMSE (minimize) and valid KGE (maximize)
def objective(trial):
    param = {
        "objective":OBJECTIVE,
        "tweedie_variance_power":trial.suggest_float("tweedie_variance_power", 1.1, 1.9),
        "eval_metric":"rmse",
        "tree_method":"hist",
        "n_estimators":N_ESTIMATORS_CAP,
        "early_stopping_rounds":EARLY_STOP,
        "num_parallel_tree":1,   # fixed: final retrain uses 1, so tune with 1 too
        "max_depth":trial.suggest_int("max_depth", 3, 8),
        "min_child_weight":trial.suggest_int("min_child_weight", 1, 30),
        "gamma":trial.suggest_float("gamma", 0.0, 5.0),
        "subsample":trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "learning_rate":trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        "reg_alpha":trial.suggest_float("reg_alpha", 1e-9, 1.0, log=True),
        "reg_lambda":trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "n_jobs":64,
        "random_state":SEED,
    }
    print(f'  [{variant}/{domain}] starting trial {trial.number} ...')
    eval_set=[(features_valid,target_valid)]
    xgbr=xgb.XGBRegressor(**param)
    xgbr.fit(features_train,target_train,eval_set=eval_set,verbose=VERBOSE_ROUNDS)
    preds=xgbr.predict(features_valid)
    # Tweedie (log link) can overflow to inf/NaN; a NaN return marks the trial failed and the study continues
    if not np.isfinite(preds).all():
        print(f'  [{variant}/{domain}] trial {trial.number}: non-finite predictions, marked failed')
        return float('nan'), float('nan')
    rmse=np.sqrt(mean_squared_error(target_valid,preds))
    k=kge(target_valid.to_numpy(),preds)
    trial.set_user_attr('best_iteration',int(xgbr.best_iteration))
    trial.set_user_attr('std_ratio',float(np.std(preds)/np.std(target_valid.to_numpy()))) # < 1 = too flat, high SWE underestimated
    print(f'  [{variant}/{domain}] trial {trial.number} done: valid RMSE={rmse:.4f}, KGE={k:.4f}, best_iter={xgbr.best_iteration}'
          + ('  <-- near n_estimators cap' if xgbr.best_iteration>=N_ESTIMATORS_CAP-EARLY_STOP else ''))
    return rmse, k


# loop over the chosen domains
for domain in domains:
    print('\n'+'='*70)
    print('DOMAIN:',domain,'| variant:',variant,'| objective:',OBJECTIVE)
    domStart=time.time()

    # filter rows for this domain
    if domain=='both':
        df_dom=df_input
    else:
        df_dom=df_input[df_input['tag']==domain]
    if len(df_dom)==0:
        print('  no rows for',domain,'(check the tag values printed above) - skipping')
        continue

    # split to validation and train by snow season
    vm=df_dom['season'].isin(valid_s).to_numpy()
    train_set=df_dom[~vm].reset_index(drop=True)
    valid_set=df_dom[vm].reset_index(drop=True)
    if len(train_set)==0 or len(valid_set)==0:
        print('  empty train or valid split for',domain,'- skipping')
        continue
    train_seasons=sorted(train_set['season'].unique().tolist())
    print('  train seasons:',train_seasons,'| valid seasons:',valid_s)

    # target/features (these globals are what objective() reads); tune on the FULL df
    target_train=train_set[target]
    features_train=train_set[features].astype(np.float32)
    target_valid=valid_set[target]
    features_valid=valid_set[features].astype(np.float32)
    print('  train rows',len(features_train),' valid rows',len(features_valid),'(tuning on full train)')

    # ---- EDA on this domain's train / valid ----
    print('  --- EDA: train vs valid ---')
    yt=train_set['SWE'].to_numpy(); yv=valid_set['SWE'].to_numpy()
    print('  rows per tag, train:', train_set['tag'].value_counts().to_dict())
    print('  rows per tag, valid:', valid_set['tag'].value_counts().to_dict())
    print('  SWE stats           train        valid')
    for name,fn in [('mean',np.mean),('median',np.median),('p95',lambda a:np.percentile(a,95)),('max',np.max)]:
        print(f'    {name:<8} {fn(yt):12.3f} {fn(yv):12.3f}')
    print(f'    frac0    {(yt==0).mean():12.3f} {(yv==0).mean():12.3f}')
    print(f'    snowon   {yt[yt>0].mean():12.3f} {yv[yv>0].mean():12.3f}')
    print(f'    valid SWE max {yv.max():.1f} vs train SWE max {yt.max():.1f}'
          + ('  <-- valid exceeds train range' if yv.max()>yt.max() else ''))
    print('    months train:', sorted(train_set['utctime'].dt.month.unique().tolist()))
    print('    months valid:', sorted(valid_set['utctime'].dt.month.unique().tolist()))
    na=features_train.isna().mean().sort_values(ascending=False)
    print('    top-5 NaN-fraction features (train):', na.head(5).round(3).to_dict())

    plt.figure(figsize=(7,4))
    plt.hist(yt[yt>0],bins=60,alpha=0.5,density=True,label='train')
    plt.hist(yv[yv>0],bins=60,alpha=0.5,density=True,label='valid')
    plt.yscale('log'); plt.xlabel('SWE (snow-on only)'); plt.ylabel('density (log)')
    plt.legend(); plt.title(f'SWE distribution {variant} {domain}, train vs valid'); plt.tight_layout()
    plt.savefig(f'{res_dir}/eda_swe_hist_{variant}_{domain}_{optuna_tag}.png',dpi=120); plt.close()

    mth_t=train_set.groupby(train_set['utctime'].dt.month)['SWE'].mean()
    mth_v=valid_set.groupby(valid_set['utctime'].dt.month)['SWE'].mean()
    plt.figure(figsize=(7,4))
    plt.plot(mth_t.index,mth_t.values,'-o',label='train')
    plt.plot(mth_v.index,mth_v.values,'-o',label='valid')
    plt.xlabel('month'); plt.ylabel('mean SWE'); plt.legend()
    plt.title(f'seasonal cycle {variant} {domain}, train vs valid'); plt.tight_layout()
    plt.savefig(f'{res_dir}/eda_swe_seasonal_{variant}_{domain}_{optuna_tag}.png',dpi=120); plt.close()

    # ---- run study for this domain (multi-objective: RMSE down, KGE up) ----
    study = optuna.create_study(storage="sqlite:///MLexperiments.sqlite3",
                                study_name="xgb-swe-"+variant+"-"+domain+"-"+optuna_tag,
                                directions=["minimize","maximize"],load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, timeout=432000)

    print('  finished trials:', len(study.trials))
    # Pareto front: trials where no other trial has both lower RMSE and higher KGE
    pareto=sorted(study.best_trials,key=lambda t:t.values[0])
    if not pareto:
        print('  no completed trials for',domain,'- skipping')
        continue
    print('  Pareto front (trial, valid RMSE, valid KGE, std ratio):')
    for t in pareto:
        print(f'     {t.number:4d}  {t.values[0]:10.4f}  {t.values[1]:8.4f}  {t.user_attrs.get("std_ratio",np.nan):6.3f}')
    pd.DataFrame([{'trial':t.number,'valid_rmse':t.values[0],'valid_kge':t.values[1],
                   'std_ratio':t.user_attrs.get('std_ratio',np.nan),'best_iteration':t.user_attrs.get('best_iteration'),**t.params}
                  for t in pareto]).to_csv(f'{res_dir}/optuna_pareto_{variant}_{domain}_{optuna_tag}.csv',index=False)

    # pick final trial: highest KGE among Pareto trials with RMSE <= best RMSE * (1+RMSE_TOL)
    rmse_min=pareto[0].values[0]
    ok=[t for t in pareto if t.values[0]<=rmse_min*(1+RMSE_TOL)]
    trial=max(ok,key=lambda t:t.values[1])
    print(f'  chosen trial {trial.number}: valid RMSE={trial.values[0]:.4f} (front best {rmse_min:.4f}), KGE={trial.values[1]:.4f}  (RMSE_TOL={RMSE_TOL})')
    print('  best params:')
    for key, value in trial.params.items():
        print('     {}: {}'.format(key, value))

    # ---- optuna plots: Pareto front + history of both objectives ----
    done=[t for t in study.trials if t.values is not None]
    t_num=[t.number for t in done]; t_rmse=[t.values[0] for t in done]; t_kge=[t.values[1] for t in done]
    fig,axs=plt.subplots(1,2,figsize=(16,5))
    ax=axs[0]
    ax.scatter(t_rmse,t_kge,s=15,alpha=0.5,label='trials')
    ax.plot([t.values[0] for t in pareto],[t.values[1] for t in pareto],'r-o',ms=4,lw=1,label='Pareto front')
    ax.scatter([trial.values[0]],[trial.values[1]],s=120,facecolors='none',edgecolors='k',lw=2,label=f'chosen (trial {trial.number})')
    ax.axvline(rmse_min*(1+RMSE_TOL),color='gray',ls=':',lw=1,label=f'RMSE_TOL {RMSE_TOL}')
    ax.set_xlabel('valid RMSE'); ax.set_ylabel('valid KGE'); ax.set_title(f'Pareto front ({variant} {domain}, {optuna_tag})')
    ax.legend(); ax.grid(True,alpha=0.3)
    ax=axs[1]
    ax.scatter(t_num,t_rmse,s=15,alpha=0.6,color='tab:blue',label='valid RMSE')
    ax.plot(t_num,pd.Series(t_rmse).cummin(),color='tab:blue',lw=1.5)
    ax.set_xlabel('trial number'); ax.set_ylabel('valid RMSE',color='tab:blue')
    ax2=ax.twinx()
    ax2.scatter(t_num,t_kge,s=15,alpha=0.6,color='tab:orange',label='valid KGE')
    ax2.plot(t_num,pd.Series(t_kge).cummax(),color='tab:orange',lw=1.5)
    ax2.set_ylabel('valid KGE',color='tab:orange')
    ax.set_title(f'Optuna optimization history ({variant} {domain}, {optuna_tag}); lines = best so far')
    ax.grid(True,alpha=0.3); plt.tight_layout()
    hist_file=f'{res_dir}/optuna_history_{variant}_{domain}_{optuna_tag}'
    plt.savefig(hist_file+'.png',dpi=200); plt.savefig(hist_file+'.svg'); plt.close()
    print('  optuna Pareto + history ->',hist_file+'.png/.svg')

    # ---- retrain final model with chosen params ON FULL TRAIN ----
    best=dict(trial.params)   # includes tweedie_variance_power
    best.update({"objective":OBJECTIVE,"eval_metric":"rmse","tree_method":"hist",
                 "num_parallel_tree":1,"n_estimators":N_ESTIMATORS_CAP,
                 "early_stopping_rounds":EARLY_STOP,"n_jobs":64,"random_state":SEED})
    final_model=xgb.XGBRegressor(**best)
    final_model.fit(features_train,target_train,eval_set=[(features_valid,target_valid)],verbose=VERBOSE_ROUNDS)
    best_iter=int(final_model.best_iteration)
    print('  best_iteration (actual trees used):',best_iter)
    if best_iter>=N_ESTIMATORS_CAP-EARLY_STOP:
        print('  WARNING: best_iteration near N_ESTIMATORS_CAP, consider raising the cap')

    # ---- metrics on valid (optimistic: valid drove tuning + early stopping) ----
    pred=final_model.predict(features_valid)
    yv=valid_set['SWE'].to_numpy()
    rmse_all=np.sqrt(mean_squared_error(yv,pred)); mae_all=mean_absolute_error(yv,pred)
    bias=float(np.mean(pred-yv)); r2_all=r2_score(yv,pred); kge_all=kge(yv,pred)
    mask=yv>0
    rmse_snow=np.sqrt(mean_squared_error(yv[mask],pred[mask])); mae_snow=mean_absolute_error(yv[mask],pred[mask])
    r2_snow=r2_score(yv[mask],pred[mask])
    frac_small_nosnow=float((pred[~mask]<1).mean()) if (~mask).any() else float('nan')
    print(f'  VALID  RMSE={rmse_all:.3f}  MAE={mae_all:.3f}  bias={bias:.3f}  R2={r2_all:.3f}  KGE={kge_all:.3f}  (all rows)')
    print(f'  VALID  RMSE={rmse_snow:.3f}  MAE={mae_snow:.3f}  R2={r2_snow:.3f}          (snow-on rows only)')
    print(f'  VALID  frac pred<1mm where obs=0: {frac_small_nosnow:.3f}')
    metrics_domain={}
    for dom in valid_set['tag'].unique():
        m=(valid_set['tag']==dom).to_numpy()
        r=np.sqrt(mean_squared_error(yv[m],pred[m])); k=kge(yv[m],pred[m])
        metrics_domain[dom]={'rmse':float(r),'kge':float(k)}
        print(f'  VALID  RMSE={r:.3f}  KGE={k:.3f}  ({dom}, {m.sum()} rows)')

    # obs vs pred scatter
    n=min(30000,len(yv)); idx=np.random.RandomState(SEED).choice(len(yv),n,replace=False)
    plt.figure(figsize=(5,5))
    plt.scatter(yv[idx],pred[idx],s=3,alpha=0.3)
    lim=max(yv.max(),pred.max())
    plt.plot([0,lim],[0,lim],'k--',lw=1)
    plt.xlabel('observed SWE'); plt.ylabel('predicted SWE')
    plt.title(f'valid obs vs pred ({variant} {domain}, {optuna_tag})'); plt.tight_layout()
    plt.savefig(f'{res_dir}/valid_scatter_{variant}_{domain}_{optuna_tag}.png',dpi=120); plt.close()

    # ---- save model + hyperparameters + context ----
    model_path=f'{mod_dir}/mdl_swe_{variant}_{domain}_{optuna_tag}.json'
    final_model.save_model(model_path)
    print('  saved model ->',model_path)

    hp_out=dict(trial.params)   # includes tweedie_variance_power
    hp_out.update({"objective":OBJECTIVE,"eval_metric":"rmse","tree_method":"hist",
                   "num_parallel_tree":1,"variant":variant,"domain":domain,"n_estimators_earlystop":best_iter,
                   "n_features":len(features),"features":features,"drop_latlon":DROP_LATLON,
                   "sd_change_days":SD_CHANGE_DAYS,
                   "split_by":"snowseason","snow_start_month":SNOW_START_MONTH,
                   "train_seasons":train_seasons,"valid_seasons":valid_s,
                   "tuning":"multi-objective (min valid RMSE, max valid KGE)","rmse_tol":RMSE_TOL,
                   "n_trials":N_TRIALS,
                   "chosen_trial":trial.number,"chosen_trial_rmse":trial.values[0],"chosen_trial_kge":trial.values[1],
                   "pareto_size":len(pareto),
                   "valid_rmse_all":float(rmse_all),"valid_mae_all":float(mae_all),"valid_r2_all":float(r2_all),
                   "valid_kge_all":float(kge_all),
                   "valid_bias":bias,"valid_rmse_snowon":float(rmse_snow),
                   "valid_mae_snowon":float(mae_snow),"valid_r2_snowon":float(r2_snow),
                   "valid_frac_pred_lt1mm_obs0":frac_small_nosnow,
                   "valid_metrics_by_domain":metrics_domain,
                   "input_file":os.path.basename(input),
                   "study_name":"xgb-swe-"+variant+"-"+domain+"-"+optuna_tag})
    with open(f'{mod_dir}/hyperparams_swe_{variant}_{domain}_{optuna_tag}.json','w') as f:
        json.dump(hp_out,f,indent=2)
    print('  saved hyperparams ->',f'{mod_dir}/hyperparams_swe_{variant}_{domain}_{optuna_tag}.json')

    # built-in gain importance
    gain=final_model.get_booster().get_score(importance_type='gain')
    pd.DataFrame([(k,gain.get(k,0.0)) for k in features],columns=['feature','gain']
                ).sort_values('gain',ascending=False
                ).to_csv(f'{res_dir}/gain_importance_{variant}_{domain}_{optuna_tag}.csv',index=False)

    # ---- SHAP (TreeExplainer) on a sample of valid ----
    # NOTE: with reg:tweedie (log link) SHAP values are in log(SWE) margin units, not mm;
    # contributions combine multiplicatively on the SWE scale
    ns=min(20000,len(features_valid))
    shap_X=features_valid.sample(n=ns,random_state=SEED)
    explainer=shap.TreeExplainer(final_model)
    shap_values=explainer.shap_values(shap_X)

    shap.summary_plot(shap_values,shap_X,max_display=25,show=False)
    plt.title(f'SHAP (log-SWE margin) {variant} {domain}')
    plt.tight_layout(); plt.savefig(f'{res_dir}/shap_beeswarm_{variant}_{domain}_{optuna_tag}.png',dpi=120); plt.close()

    shap.summary_plot(shap_values,shap_X,plot_type='bar',max_display=25,show=False)
    plt.title(f'mean |SHAP| (log-SWE margin) {variant} {domain}')
    plt.tight_layout(); plt.savefig(f'{res_dir}/shap_bar_{variant}_{domain}_{optuna_tag}.png',dpi=120); plt.close()

    shap_imp=pd.DataFrame({'feature':features,
                           'mean_abs_shap':np.abs(shap_values).mean(axis=0)}
                         ).sort_values('mean_abs_shap',ascending=False)
    shap_imp.to_csv(f'{res_dir}/shap_importance_{variant}_{domain}_{optuna_tag}.csv',index=False)

    for feat in shap_imp['feature'].head(4):
        shap.dependence_plot(feat,shap_values,shap_X,show=False)
        plt.title(f'SHAP dependence (log-SWE margin): {feat}')
        plt.tight_layout()
        plt.savefig(f'{res_dir}/shap_dep_{feat}_{variant}_{domain}_{optuna_tag}.png',dpi=120); plt.close()
    print('  SHAP + plots ->',f'{res_dir}/*_{variant}_{domain}_{optuna_tag}.*')
    print('  domain time (min): %.2f'%((time.time()-domStart)/60))

executionTime=(time.time()-startTime)
print('\nAll domains done. Total execution time in minutes: %.2f'%(executionTime/60))