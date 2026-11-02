import os,sys,json,time,glob,warnings,traceback
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')  # headless server
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, explained_variance_score
import shap

# Independent validation of the SWE XGBoost models trained by the multi-objective tuning script
# usage: python xgb-validate-cryo-swe.py <optuna_tag> [<optuna_tag> ...]   one or more trained models (same tags as the training runs)
#        python xgb-validate-cryo-swe.py all                              every trained model in models/
#        python xgb-validate-cryo-swe.py missing                          only trained models without a finished validation
# the validation data is read once (union of the columns all chosen models need) and every model is validated on the same rows
# one model per tag (domain 'both'), predicts every validation row (all locations, all years); scores for Scandinavia, Alps and both together
# outputs go to validationresults_<optuna_tag>/, one folder per model; the Streamlit app finds them and offers a model selection

warnings.filterwarnings('ignore')
startTime=time.time()

def tlog(msg):
    print(f'[{(time.time()-startTime)/60:6.1f} min] {msg}',flush=True)

USAGE=('usage: python xgb-validate-cryo-swe.py <optuna_tag> [<optuna_tag> ...]\n'
       '       python xgb-validate-cryo-swe.py all       every trained model in models/\n'
       '       python xgb-validate-cryo-swe.py missing   only trained models without a finished validation')

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
variant='cryoscope-SWE'   # same as in the training script
train_domain='both'       # the training script only trains the combined model

base_dir='/home/ubuntu/data/cryoscope/snowmodel/'
data_dir=f'{base_dir}validationdata/'                       # validation data
mod_dir=f'{base_dir}models/'                                # trained models

input_file=f'{data_dir}validation_all_scandi_alps_ext.csv.gz'
# DEM files hold dist_to_train_m (distance of each validation point to nearest training point)
dem_files={'Scandinavia':f'{data_dir}validation_dem+fch_merged_fin25_10000_fscs1_scandi.csv.gz',
           'Alps':f'{data_dir}validation_dem+fch_merged_fin2_10000_fscs1_alps.csv.gz'}
mname='both'          # model label in the output file names (bothmodel_Scandinavia etc.), same for every tag
zero_thr=0.1        # mm, predictions below this are set to 0 (tweedie predictions are always > 0)
snow_thr=1.0          # mm SWE, snow / no snow threshold (detection scores, onset, melt-out, duration)
peak_min=10.0         # mm, location seasons with lower observed peak SWE are left out of timing scores
min_season_days=300   # location seasons with fewer days are left out of snow season metrics
err_max=150.0         # mm, fixed colour scale for RMSE maps (same in every domain and run), values above shown red
me_max=75.0           # mm, fixed symmetric colour scale for ME maps
run_shap=True
shap_n=5000           # rows sampled for SHAP
save_predictions=True   # writes all predictions to SWE_xgboost_validation_<model>model.csv.gz (large, slow; used by the app's Validation explorer)

def model_stem(tag): return f'{variant}_{train_domain}_{tag}'
def out_dir(tag): return f'{base_dir}validationresults_{tag}/'
model_prefix=f'mdl_swe_{variant}_{train_domain}_'

def trained_tags():
    return sorted(os.path.basename(p)[len(model_prefix):-len('.json')] for p in glob.glob(f'{mod_dir}{model_prefix}*.json'))

def validated(tag):
    # finished validation: run_info.json says complete; older folders without the flag count as done when the SHAP csv (written last) exists
    try:
        with open(f'{out_dir(tag)}run_info.json') as f: info=json.load(f)
    except Exception:
        info={}
    return bool(info.get('complete',os.path.exists(f'{out_dir(tag)}shap_importance_{mname}model.csv')))

# ---------------------------------------------------------------
# Which models
# ---------------------------------------------------------------
args=sys.argv[1:]
if not args: raise SystemExit(USAGE)
avail=trained_tags()
tlog(f'trained models in {mod_dir}: {avail}')
if args==['all']:
    tags=avail
elif args==['missing']:
    tags=[t for t in avail if not validated(t)]
else:
    tags=list(dict.fromkeys(args))
    unknown=[t for t in tags if t not in avail]
    if unknown: raise SystemExit(f'no trained model for {unknown}; trained models: {avail}')
if not tags: raise SystemExit('nothing to validate (every trained model already has a finished validation)')
tlog(f'validating: {tags}')

# ---------------------------------------------------------------
# Load models; feature list (and order) comes from the model files
# ---------------------------------------------------------------
header=pd.read_csv(input_file,nrows=0).columns
def file_col(c):
    # column in the validation file for model column c (the model may use 'sd_00' where the file has 'sd'); None if missing
    if c in header: return c
    if c.endswith('_00') and c[:-3] in header: return c[:-3]
    return None
derived={'doy_cos','doy_sin'}
def needed(feats):
    # columns to read for these features: lag features need their base column, doy features are computed
    return sorted({f[:-5] if f.endswith('_lag1') else f for f in feats if f not in derived})

M={}; skipped={}
for tag in tags:
    stem=model_stem(tag)
    mf=f'{mod_dir}mdl_swe_{stem}.json'; hf=f'{mod_dir}hyperparams_swe_{stem}.json'
    try:
        m=xgb.XGBRegressor(); m.load_model(mf)
    except Exception as e:
        skipped[tag]=f'model not loaded: {e}'; continue
    fn=m.get_booster().feature_names
    if fn is None:
        skipped[tag]='model file has no stored feature names'; continue
    hp={}
    if os.path.exists(hf):
        with open(hf) as f: hp=json.load(f)
    else:
        tlog(f'{tag}: WARNING no hyperparams file {hf}, season split info will be empty')
    if hp.get('features') and list(hp['features'])!=list(fn):
        tlog(f"{tag}: WARNING feature list in hyperparams differs from the model file: only in json {sorted(set(hp['features'])-set(fn))}, only in model {sorted(set(fn)-set(hp['features']))}")
    miss=[c for c in needed(fn) if file_col(c) is None]
    if miss:
        skipped[tag]=f'columns missing in validation file: {miss}'; continue
    M[tag]=dict(stem=stem,model=m,feats=list(fn),hp=hp,model_file=mf,hp_file=hf)
    tlog(f'{tag}: {len(fn)} features: {list(fn)}')
for t,why in skipped.items(): tlog(f'SKIPPED {t}: {why}')
if not M: raise SystemExit('no model can be validated')

# ---------------------------------------------------------------
# Read and preprocess exactly as in training (union of the columns every model needs)
# ---------------------------------------------------------------
all_feats=sorted(set().union(*(v['feats'] for v in M.values())))
lag_feats=[f for f in all_feats if f.endswith('_lag1')]
base_cols=['utctime','latitude','longitude','SWE','tag','height']
# baselines: SNOWCAP climatology (swe_mean) and ERA5-Land SWE (sd, m water equivalent; read as sd_00)
extra=[c for c in ['swe_mean','sd_00'] if file_col(c)]
cols=sorted(set(base_cols)|set(needed(all_feats))|set(extra))
missing=[c for c in cols if file_col(c) is None]
if missing: raise SystemExit(f'missing columns in validation file: {missing}')
rename={file_col(c):c for c in cols if file_col(c)!=c}
tlog(f'renaming: {rename}')

tlog(f'reading {input_file} ({len(cols)} columns)')
df=pd.read_csv(input_file,usecols=[file_col(c) for c in cols]).rename(columns=rename)
# negative values to 0 as in training (target, and the climatology features)
df['SWE']=df['SWE'].clip(lower=0.0)
for c in ['swe_mean','swe_max']:
    if c in df: df[c]=df[c].clip(lower=0.0)
df['utctime']=pd.to_datetime(df['utctime'])
doy=df['utctime'].dt.dayofyear
df['doy_cos']=np.cos(2*np.pi*doy/365.25)
df['doy_sin']=np.sin(2*np.pi*doy/365.25)

# lag-1: previous day's value at the SAME location; NaN if that day is missing
tlog('building lag features')
df['date']=df['utctime'].dt.normalize()
df=df.sort_values(['latitude','longitude','date']).reset_index(drop=True)
g=df.groupby(['latitude','longitude'],sort=False)
prev_date=g['date'].shift(1)
consecutive=(df['date']-prev_date)==pd.Timedelta(days=1)
for f in lag_feats:
    df[f]=g[f[:-5]].shift(1).where(consecutive)
df=df.drop(columns='date')
del g,prev_date,consecutive
print('duplicate location-days:',int(df.duplicated(['latitude','longitude','utctime']).sum()))
for f in lag_feats: print(f'{f} NaN fraction:',round(float(df[f].isna().mean()),4))

# snow year (September to August) and day of snow year (0 = 1 September)
yr=df['utctime'].dt.year; mo=df['utctime'].dt.month
df['season']=np.where(mo>=9,yr,yr-1).astype(np.int16)
starts={s:pd.Timestamp(f'{s}-09-01') for s in df['season'].unique()}
df['dos']=(df['utctime']-df['season'].map(starts)).dt.days.astype(np.int16)
del yr,mo

# baseline predictions
baselines=[]
if 'swe_mean' in df:
    df['SWE_climatology']=df['swe_mean'].clip(lower=0.0); baselines.append('SWE_climatology')
if 'sd_00' in df:
    df['SWE_era5l']=(df['sd_00']*1000.0).clip(lower=0.0); baselines.append('SWE_era5l')  # m -> mm
tlog(f'baselines: {baselines}')

# distance to nearest training point
dist=[]
for t,f in dem_files.items():
    if os.path.exists(f) and 'dist_to_train_m' in pd.read_csv(f,nrows=0).columns:
        dist.append(pd.read_csv(f,usecols=['latitude','longitude','dist_to_train_m']))
    else:
        tlog(f'no dist_to_train_m for {t} ({f})')
if dist:
    dd=pd.concat(dist).drop_duplicates(['latitude','longitude'])
    df=df.merge(dd,on=['latitude','longitude'],how='left')
    tlog(f"dist_to_train_m found for {df['dist_to_train_m'].notna().mean()*100:.1f} % of rows")

f32=df.select_dtypes(include='float').columns.difference(['latitude','longitude'])
df[f32]=df[f32].astype(np.float32)
df=df.rename(columns={'SWE':'SWE_observed'})
tlog('validation data ready')
print(df.groupby('tag')['utctime'].agg(['min','max','count']))
for b in baselines:
    print(df.groupby('tag')[['SWE_observed',b]].mean())  # sanity check of baseline units

# ---------------------------------------------------------------
# Scores
# ---------------------------------------------------------------
def scores(o,p,spearman=True):
    o=np.asarray(o,dtype=np.float64); p=np.asarray(p,dtype=np.float64)
    ok=np.isfinite(o)&np.isfinite(p); o=o[ok]; p=p[ok]
    n=len(o)
    if n<2: return {'n':n}
    e=p-o
    me=e.mean(); rmse=np.sqrt(np.mean(e**2))
    so,sp=o.std(),p.std()
    r=np.corrcoef(o,p)[0,1] if so>0 and sp>0 else np.nan
    alpha=sp/so if so>0 else np.nan                    # variability ratio
    beta=p.mean()/o.mean() if o.mean()>0 else np.nan    # bias ratio
    obs_s=o>=snow_thr; pred_s=p>=snow_thr
    hits=np.sum(obs_s&pred_s); miss=np.sum(obs_s&~pred_s)
    fa=np.sum(~obs_s&pred_s); cn=np.sum(~obs_s&~pred_s)
    acc=(hits+cn)/n
    pe=((hits+miss)*(hits+fa)+(cn+miss)*(cn+fa))/n**2  # accuracy expected by chance
    return {'n':n,'obs_mean':o.mean(),'pred_mean':p.mean(),
            'ME':me,'MAE':np.mean(np.abs(e)),'MedAE':np.median(np.abs(e)),
            'P95_AE':np.percentile(np.abs(e),95),
            'RMSE':rmse,'ubRMSE':np.sqrt(max(rmse**2-me**2,0.0)),
            'nRMSE':rmse/o.mean() if o.mean()>0 else np.nan,
            'PBIAS_%':100*e.sum()/o.sum() if o.sum()>0 else np.nan,
            'R2':r2_score(o,p),'explained_var':explained_variance_score(o,p),
            'pearson_r':r,
            'spearman_r':pd.Series(o).corr(pd.Series(p),method='spearman') if spearman else np.nan,
            'std_ratio':alpha,'KGE':1-np.sqrt((r-1)**2+(alpha-1)**2+(beta-1)**2),
            'POD':hits/(hits+miss) if hits+miss else np.nan,
            'FAR':fa/(hits+fa) if hits+fa else np.nan,
            'CSI':hits/(hits+miss+fa) if hits+miss+fa else np.nan,
            'accuracy':acc,'HSS':(acc-pe)/(1-pe) if pe<1 else np.nan}

def bname(b): return b.replace('SWE_','')
short=('n','obs_mean','pred_mean','ME','MAE','RMSE','ubRMSE','R2','explained_var','KGE')

def model_scores(ds,spearman=True):
    # XGBoost scores plus skill against each baseline on exactly the same rows (baseline may be NaN at some locations)
    s=scores(ds['SWE_observed'],ds['SWE_xgboost'],spearman=spearman)
    for b in baselines:
        ok=ds[b].notna()
        sb=scores(ds['SWE_observed'][ok],ds[b][ok],spearman=False)
        sm=scores(ds['SWE_observed'][ok],ds['SWE_xgboost'][ok],spearman=False)
        s[f'n_{bname(b)}']=sb.get('n',0)
        s[f'RMSE_{bname(b)}']=sb.get('RMSE',np.nan)
        s[f'skill_MSE_vs_{bname(b)}']=1-sm['RMSE']**2/sb['RMSE']**2 if sb.get('RMSE',0)>0 else np.nan
        s[f'skill_MAE_vs_{bname(b)}']=1-sm['MAE']/sb['MAE'] if sb.get('MAE',0)>0 else np.nan
    return s

# ---------------------------------------------------------------
# Analyses and plots (write to the current model's res_dir / fig_dir, set in the model loop below)
# ---------------------------------------------------------------
season_months=[9,10,11,12,1,2,3,4,5,6,7,8]

def plot_scatter(d,label):
    o,p=d['SWE_observed'],d['SWE_xgboost']
    s=scores(o,p,spearman=False); lim=np.nanpercentile(np.r_[o,p],99.9)
    fig,ax=plt.subplots(figsize=(6,6))
    hb=ax.hexbin(o,p,gridsize=120,bins='log',mincnt=1,cmap='viridis',extent=(0,lim,0,lim))
    ax.plot([0,lim],[0,lim],'r--',lw=1)
    ax.set_xlim(0,lim); ax.set_ylim(0,lim)
    ax.set_xlabel('SWE observed (mm)'); ax.set_ylabel('SWE XGBoost (mm)')
    ax.set_title(f'{tag}: {label}')
    ax.text(0.03,0.97,f"n={s['n']}\nRMSE={s['RMSE']:.1f} mm\nubRMSE={s['ubRMSE']:.1f} mm\nME={s['ME']:.1f} mm\nMAE={s['MAE']:.1f} mm\nR2={s['R2']:.3f}\nKGE={s['KGE']:.3f}",
            transform=ax.transAxes,va='top',fontsize=9,bbox=dict(fc='white',alpha=0.8))
    fig.colorbar(hb,ax=ax,label='count (log)')
    fig.tight_layout(); fig.savefig(f'{fig_dir}scatter_{label}.png',dpi=150); plt.close(fig)

def per_location(d):
    e=d['SWE_xgboost']-d['SWE_observed']
    tmp=pd.DataFrame({'ME':e,'err2':e**2,'obs_mean':d['SWE_observed'],'pred_mean':d['SWE_xgboost']})
    for b in baselines: tmp[f'err2_{bname(b)}']=(d[b]-d['SWE_observed'])**2
    keys=[d['latitude'],d['longitude']]
    g=tmp.groupby(keys)
    t=g.mean(); t['n']=g.size()
    t['RMSE']=np.sqrt(t.pop('err2'))
    for b in baselines:
        k=bname(b); t[f'RMSE_{k}']=np.sqrt(t.pop(f'err2_{k}'))
        t[f'skill_MSE_vs_{k}']=1-t['RMSE']**2/t[f'RMSE_{k}']**2
    t['r']=d.groupby(['latitude','longitude'])[['SWE_observed','SWE_xgboost']].apply(
        lambda x: x['SWE_observed'].corr(x['SWE_xgboost']))
    t['height']=d.groupby(keys)['height'].first()
    if 'dist_to_train_m' in d: t['dist_to_train_m']=d.groupby(keys)['dist_to_train_m'].first()
    return t.replace([np.inf,-np.inf],np.nan).reset_index()

def plot_maps(pl,label):
    specs=[('ME','RdBu_r','sym','mean error (mm)'),('RMSE','viridis','pos','RMSE (mm)'),('r','magma','r','Pearson r')]
    if 'skill_MSE_vs_climatology' in pl: specs.append(('skill_MSE_vs_climatology','RdBu','skill','MSE skill vs climatology'))
    fig,axs=plt.subplots(2,2,figsize=(14,12)); axs=axs.ravel()
    asp=1/np.cos(np.deg2rad(pl['latitude'].mean()))
    for ax,(c,cm,mode,lab) in zip(axs,specs):
        ext='neither'
        if mode=='sym': vmin,vmax=-me_max,me_max; ext='both'
        elif mode=='pos':
            vmin,vmax=0,err_max; ext='max'
            cm=plt.get_cmap(cm).copy(); cm.set_over('red')   # values above err_max in red
        elif mode=='r': vmin,vmax=max(0,np.nanpercentile(pl[c],2)),1
        else: vmin,vmax=-1,1
        sc=ax.scatter(pl['longitude'],pl['latitude'],c=pl[c],cmap=cm,vmin=vmin,vmax=vmax,s=4,lw=0)
        ax.set_aspect(asp); ax.set_title(f'{tag} {label}: {lab}'+(f', {int((pl[c]>err_max).sum())} locations above {err_max:g} mm' if mode=='pos' else ''))
        ax.set_xlabel('longitude'); ax.set_ylabel('latitude')
        fig.colorbar(sc,ax=ax,shrink=0.8,label=lab,extend=ext)
    for ax in axs[len(specs):]: ax.axis('off')
    fig.tight_layout(); fig.savefig(f'{fig_dir}maps_{label}.png',dpi=150); plt.close(fig)

def plot_timeseries(d,label):
    cols=['SWE_observed','SWE_xgboost']+baselines
    ts=d.groupby(d['utctime'].dt.normalize())[cols].mean()
    fig,ax=plt.subplots(figsize=(14,4))
    for c,lab,ls in zip(cols,['observed','XGBoost']+[bname(b) for b in baselines],['-','-',':','--']):
        ax.plot(ts.index,ts[c],lw=1,ls=ls,label=lab)
    ax.set_ylabel('domain mean SWE (mm)'); ax.set_title(f'{tag}: {label}'); ax.legend()
    fig.tight_layout(); fig.savefig(f'{fig_dir}timeseries_{label}.png',dpi=150); plt.close(fig)

def monthly(d):
    m=d.groupby(d['utctime'].dt.month).apply(
        lambda x: pd.Series({k:v for k,v in scores(x['SWE_observed'],x['SWE_xgboost'],spearman=False).items() if k in short}))
    return m.reindex([x for x in season_months if x in m.index])

def plot_monthly(mt,label):
    x=np.arange(len(mt)); w=0.27
    fig,ax=plt.subplots(figsize=(10,4))
    ax.bar(x-w,mt['RMSE'],w,label='RMSE'); ax.bar(x,mt['ubRMSE'],w,label='ubRMSE'); ax.bar(x+w,mt['ME'],w,label='ME')
    ax.axhline(0,color='k',lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(mt.index.astype(int))
    ax.set_xlabel('month'); ax.set_ylabel('mm'); ax.set_title(f'{tag}: {label}'); ax.legend()
    fig.tight_layout(); fig.savefig(f'{fig_dir}monthly_{label}.png',dpi=150); plt.close(fig)

def plot_elevation(d,label):
    bins=np.arange(0,d['height'].max()+250,250)
    e=d['SWE_xgboost']-d['SWE_observed']
    b=d.assign(err=e,err2=e**2,band=pd.cut(d['height'],bins)).groupby('band',observed=True).agg(
        ME=('err','mean'),MSE=('err2','mean'),obs=('SWE_observed','mean'),n=('err','size'))
    b=b[b['n']>=100]
    mid=[iv.mid for iv in b.index]
    fig,ax=plt.subplots(figsize=(6,6))
    ax.plot(np.sqrt(b['MSE']),mid,'o-',label='RMSE')
    ax.plot(b['ME'],mid,'s-',label='ME')
    ax.plot(b['obs'],mid,'^--',label='mean observed SWE')
    ax.axvline(0,color='k',lw=0.5)
    ax.set_xlabel('mm'); ax.set_ylabel('elevation (m)'); ax.set_title(f'{tag}: {label}'); ax.legend()
    fig.tight_layout(); fig.savefig(f'{fig_dir}elevation_{label}.png',dpi=150); plt.close(fig)

def distance_analysis(d,pl,label):
    # errors binned by distance to nearest training point (10 bins with equal numbers of locations)
    if 'dist_to_train_m' not in d or pl['dist_to_train_m'].notna().sum()<20: return None
    edges=np.unique(np.nanquantile(pl['dist_to_train_m'],np.linspace(0,1,11)))
    if len(edges)<3: return None
    band=pd.cut(d['dist_to_train_m'],edges,include_lowest=True)
    e=d['SWE_xgboost']-d['SWE_observed']
    snow=(d['SWE_observed']>0)|(d['SWE_xgboost']>0)
    tmp=pd.DataFrame({'ME':e,'MAE':e.abs(),'err2':e**2,'err2_snow':(e**2).where(snow),'mean_dist_km':d['dist_to_train_m']/1000})
    for b in baselines: tmp[f'err2_{bname(b)}']=(d[b]-d['SWE_observed'])**2
    g=tmp.groupby(band,observed=True)
    t=g.mean(); t['n']=g.size()
    t['RMSE']=np.sqrt(t.pop('err2')); t['RMSE_snow_rows']=np.sqrt(t.pop('err2_snow'))
    for b in baselines: t[f'RMSE_{bname(b)}']=np.sqrt(t.pop(f'err2_{bname(b)}'))
    t.index=t.index.astype(str); t.index.name='dist_bin_m'
    t=t.reset_index()
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(t['mean_dist_km'],t['RMSE'],'o-',label='RMSE XGBoost')
    ax.plot(t['mean_dist_km'],t['RMSE_snow_rows'],'s-',label='RMSE XGBoost, snow rows')
    ax.plot(t['mean_dist_km'],t['ME'],'^-',label='ME XGBoost')
    for b in baselines: ax.plot(t['mean_dist_km'],t[f'RMSE_{bname(b)}'],'--',label=f'RMSE {bname(b)}')
    ax.axhline(0,color='k',lw=0.5)
    ax.set_xlabel('mean distance to nearest training point (km)'); ax.set_ylabel('mm')
    ax.set_title(f'{tag}: {label}'); ax.legend()
    fig.tight_layout(); fig.savefig(f'{fig_dir}distance_{label}.png',dpi=150); plt.close(fig)
    return t

def snowyear_scores(d):
    out=[]
    if baselines: d=d[d[baselines].notna().all(axis=1)]  # same rows for every predictor
    for s,ds in d.groupby('season'):
        for c in ['SWE_xgboost']+baselines:
            sc=scores(ds['SWE_observed'],ds[c],spearman=False)
            out.append({'season':f'{s}-{s+1}','predictor':bname(c),**{k:sc.get(k,np.nan) for k in short}})
    return pd.DataFrame(out)

def plot_snowyear(sy,label):
    fig,axs=plt.subplots(1,3,figsize=(21,4))
    obs=sy.drop_duplicates('season')  # observed mean is the same for every predictor
    axs[2].plot(obs['season'],obs['obs_mean'],'ko-',lw=2,label='observed')
    for c,gp in sy.groupby('predictor',sort=False):
        axs[0].plot(gp['season'],gp['RMSE'],'o-',label=c)
        axs[1].plot(gp['season'],gp['ME'],'o-',label=c)
        axs[2].plot(gp['season'],gp['pred_mean'],'o--',label=c)
    axs[0].set_ylabel('RMSE (mm)'); axs[1].set_ylabel('ME (mm)'); axs[1].axhline(0,color='k',lw=0.5)
    axs[2].set_ylabel('mean SWE (mm)')
    for ax in axs:
        ax.set_xlabel('snow year'); ax.tick_params(axis='x',rotation=45); ax.legend(); ax.set_title(f'{tag}: {label}')
    fig.tight_layout(); fig.savefig(f'{fig_dir}snowyear_{label}.png',dpi=150); plt.close(fig)

def season_table(d,cols):
    # per location and snow year: peak SWE, peak timing, onset, melt-out, snow cover duration
    keys=[d['latitude'],d['longitude'],d['season']]
    parts=[d.groupby(keys,sort=False).size().rename('n_days')]
    for c in cols:
        v=d[c].fillna(-1)  # -1 only so idxmax works; locations without data are set to NaN below
        snow=v>=snow_thr
        tmp=pd.DataFrame({'v':v,'snow':snow,'dos_snow':d['dos'].where(snow),'valid':d[c].notna()})
        g=tmp.groupby(keys,sort=False)
        a=g.agg(peak=('v','max'),onset=('dos_snow','min'),meltout=('dos_snow','max'),duration=('snow','sum'),valid=('valid','sum'))
        a['peak_dos']=d['dos'].loc[g['v'].idxmax().values].values
        nodata=a.pop('valid')<min_season_days  # e.g. locations where the climatology is missing
        a.loc[nodata,:]=np.nan
        a.columns=[f'{x}_{c}' for x in a.columns]
        parts.append(a)
    t=pd.concat(parts,axis=1).reset_index()
    return t[t['n_days']>=min_season_days].reset_index(drop=True)

def season_scores(st,c):
    o='SWE_observed'
    st=st[st[f'peak_{c}'].notna()]  # only location seasons where this predictor has data
    e=st[f'peak_{c}']-st[f'peak_{o}']
    r={'predictor':bname(c),'n_location_seasons':len(st),
       'peak_ME':e.mean(),'peak_MAE':e.abs().mean(),'peak_RMSE':np.sqrt((e**2).mean()),
       'peak_r':st[f'peak_{c}'].corr(st[f'peak_{o}']),
       'peak_PBIAS_%':100*e.sum()/st[f'peak_{o}'].sum() if st[f'peak_{o}'].sum()>0 else np.nan}
    e=st[f'duration_{c}']-st[f'duration_{o}']
    r['duration_ME_days']=e.mean(); r['duration_MAE_days']=e.abs().mean()
    r['duration_r']=st[f'duration_{c}'].corr(st[f'duration_{o}'])
    snowy=st[st[f'peak_{o}']>=peak_min]
    r['n_snowy_location_seasons']=len(snowy)
    for m in ['peak_dos','onset','meltout']:
        e=(snowy[f'{m}_{c}']-snowy[f'{m}_{o}']).dropna()
        r[f'{m}_ME_days']=e.mean(); r[f'{m}_MAE_days']=e.abs().mean(); r[f'{m}_MedAE_days']=e.abs().median()
        r[f'{m}_n']=len(e)
    return r

def plot_snowseason(st,label):
    o='SWE_observed'
    fig,axs=plt.subplots(1,4,figsize=(24,5.5))
    ax=axs[0]
    lim=np.nanpercentile(np.r_[st[f'peak_{o}'],st['peak_SWE_xgboost']],99.5)
    ax.scatter(st[f'peak_{o}'],st['peak_SWE_xgboost'],s=2,alpha=0.3,lw=0)
    ax.plot([0,lim],[0,lim],'r--',lw=1); ax.set_xlim(0,lim); ax.set_ylim(0,lim)
    ax.set_xlabel('observed peak SWE (mm)'); ax.set_ylabel('XGBoost peak SWE (mm)')
    ax.set_title(f"{tag} {label}: peak SWE, r={st['peak_SWE_xgboost'].corr(st[f'peak_{o}']):.3f}")
    snowy=st[st[f'peak_{o}']>=peak_min]
    for ax,m,lab in zip(axs[1:],['peak_dos','onset','meltout'],['peak timing','snow onset','melt-out']):
        for c in ['SWE_xgboost']+baselines:
            e=(snowy[f'{m}_{c}']-snowy[f'{m}_{o}']).dropna()
            if len(e)==0: continue
            ax.hist(e.clip(-90,90),bins=np.arange(-90,93,3),histtype='step',lw=1.5,
                    label=f'{bname(c)} (MAE {e.abs().mean():.1f} d)')
        ax.axvline(0,color='k',lw=0.5)
        ax.set_xlabel(f'{lab} error, predicted minus observed (days)'); ax.set_ylabel('location seasons')
        ax.set_title(f'{label}: {lab}'); ax.legend()
    fig.tight_layout(); fig.savefig(f'{fig_dir}snowseason_{label}.png',dpi=150); plt.close(fig)

def plot_shap(model,Xs,label):
    # reg:tweedie (log link): SHAP values are in log(SWE) margin units, not mm
    sv=shap.TreeExplainer(model).shap_values(Xs)
    shap.summary_plot(sv,Xs,show=False,max_display=25)
    plt.title(f'{tag} {label} (log-SWE margin)'); plt.tight_layout()
    plt.savefig(f'{fig_dir}shap_{label}.png',dpi=150,bbox_inches='tight'); plt.close('all')
    pd.DataFrame({'feature':Xs.columns,'mean_abs_shap':np.abs(sv).mean(axis=0)}).sort_values(
        'mean_abs_shap',ascending=False).to_csv(f'{res_dir}shap_importance_{label}.csv',index=False)

def save_tables():
    # written after every label so a crash loses nothing
    if rows: pd.DataFrame(rows).to_csv(f'{res_dir}validation_metrics.csv',index=False)
    if monthly_tables: pd.concat(monthly_tables).to_csv(f'{res_dir}validation_metrics_monthly.csv',index=False)
    if snowyear_tables: pd.concat(snowyear_tables).to_csv(f'{res_dir}validation_metrics_snowyear.csv',index=False)
    if snowseason_rows: pd.DataFrame(snowseason_rows).to_csv(f'{res_dir}validation_metrics_snowseason.csv',index=False)
    if distance_tables: pd.concat(distance_tables).to_csv(f'{res_dir}validation_metrics_distance.csv',index=False)

def write_run_info(info):
    with open(f'{res_dir}run_info.json','w') as f: json.dump(info,f,indent=2)

# ---------------------------------------------------------------
# Validate each model: predict, score, plot; one folder per model
# ---------------------------------------------------------------
keep=['utctime','latitude','longitude','tag','height','season','dos','SWE_observed']+baselines+(['dist_to_train_m'] if 'dist_to_train_m' in df else [])
summary=[]
for i,(tag,mi) in enumerate(M.items(),1):
    t_model=time.time()
    res_dir=out_dir(tag); fig_dir=f'{res_dir}figs/'
    os.makedirs(fig_dir,exist_ok=True)
    rows=[]; monthly_tables=[]; snowyear_tables=[]; snowseason_rows=[]; distance_tables=[]
    model=mi['model']; feats=mi['feats']; hp=mi['hp']
    tlog(f'========== model {i}/{len(M)}: {tag} ({len(feats)} features) -> {res_dir} ==========')
    # run info first, so the Streamlit app can link this validation to its training run; complete=True at the end
    info={'training_stem':mi['stem'],'optuna_tag':tag,'variant':variant,'train_domain':train_domain,
          'model_file':mi['model_file'],'hyperparams_file':mi['hp_file'],'input_file':os.path.basename(input_file),
          'n_features':len(feats),'features':feats,
          'train_seasons':list(hp.get('train_seasons',[])),'valid_seasons':list(hp.get('valid_seasons',[])),
          'baselines':baselines,'complete':False,
          'written':pd.Timestamp.now(tz='Europe/Helsinki').strftime('%Y-%m-%d %H:%M')}
    write_run_info(info)
    try:
        X=df[feats].astype(np.float32)
        tlog(f'{tag}: predicting SWE for {len(X)} rows, {df.groupby(["latitude","longitude"]).ngroups} locations')
        out=df[keep].copy()
        pred=model.predict(X)
        out['SWE_xgboost']=np.where(pred<zero_thr,0.0,pred).astype(np.float32)  # below 0.001 mm (incl. negatives) -> 0
        tlog(f'{tag}: set {np.sum(pred<zero_thr)} predictions below {zero_thr} mm to 0 ({np.sum(pred<0)} were negative)')
        Xs=X.sample(min(shap_n,len(X)),random_state=42) if run_shap else None
        del X,pred
        if save_predictions:
            fout=f'{res_dir}SWE_xgboost_validation_{mname}model.csv.gz'
            tlog(f'{tag}: saving predictions to {fout}')
            out.to_csv(fout,index=False)

        # each domain separately (Scandinavia, Alps), then all validation rows together
        baseline_done=set()
        for t in sorted(out['tag'].unique())+['all']:
            d=out if t=='all' else out[out['tag']==t]
            label=f'{mname}model_{t}'
            print(f'--- {tag} {label} ---')

            # overall scores: all rows, and only rows where observed or predicted SWE > 0
            for subset in ['all_rows','snow_rows']:
                ds=d if subset=='all_rows' else d[(d['SWE_observed']>0)|(d['SWE_xgboost']>0)]
                s=model_scores(ds)
                rows.append({'model':mname,'domain':t,'subset':subset,**s})
                tlog(f"{tag} {label} {subset}: RMSE={s['RMSE']:.2f} ubRMSE={s['ubRMSE']:.2f} ME={s['ME']:.2f} MAE={s['MAE']:.2f} mm, R2={s['R2']:.3f}, KGE={s['KGE']:.3f}"
                     +''.join(f", MSE skill vs {bname(b)}={s[f'skill_MSE_vs_{bname(b)}']:.3f}" for b in baselines))
            # baselines scored on their own, once per domain (identical in every model folder)
            if t not in baseline_done:
                for b in baselines:
                    for subset in ['all_rows','snow_rows']:
                        ds=d if subset=='all_rows' else d[(d['SWE_observed']>0)|(d[b]>0)]
                        rows.append({'model':f'baseline_{bname(b)}','domain':t,'subset':subset,**scores(ds['SWE_observed'],ds[b])})
                baseline_done.add(t)

            tlog(f'{tag} {label}: scatter, monthly and snow year scores')
            plot_scatter(d,label)
            mt=monthly(d); plot_monthly(mt,label)
            monthly_tables.append(mt.assign(model=mname,domain=t).reset_index().rename(columns={'utctime':'month'}))
            sy=snowyear_scores(d); plot_snowyear(sy,label)
            snowyear_tables.append(sy.assign(model=mname,domain=t))

            if t!='all':
                tlog(f'{tag} {label}: per location scores, maps, time series, elevation and distance')
                pl=per_location(d)
                pl.to_csv(f'{res_dir}per_location_{label}.csv',index=False)
                plot_maps(pl,label)
                plot_timeseries(d,label)
                plot_elevation(d,label)
                dt=distance_analysis(d,pl,label)
                if dt is not None: distance_tables.append(dt.assign(model=mname,domain=t))
                else: tlog(f'{tag} {label}: no distance information, skipped')
                tlog(f'{tag} {label}: snow season metrics (peak SWE, timing, onset, melt-out, duration)')
                st=season_table(d,['SWE_observed','SWE_xgboost']+baselines)
                st.to_csv(f'{res_dir}snowseason_{label}.csv',index=False)
                for c in ['SWE_xgboost']+baselines:
                    snowseason_rows.append({'model':mname,'domain':t,**season_scores(st,c)})
                plot_snowseason(st,label)
            save_tables()
            tlog(f'{tag} {label}: done')
        if run_shap:
            tlog(f'{tag}: SHAP summary on {len(Xs)} sampled rows')
            plot_shap(model,Xs,f'{mname}model')
        del out
        save_tables()
        info.update({'complete':True,'minutes':round((time.time()-t_model)/60,1),
                     'written':pd.Timestamp.now(tz='Europe/Helsinki').strftime('%Y-%m-%d %H:%M')})
        write_run_info(info)
        r=pd.DataFrame(rows)
        r=r[(r['model']==mname)&(r['subset']=='all_rows')].set_index('domain')
        summary.append({'tag':tag,'features':len(feats),'status':'ok',
                        **{f'{k} {dom}':r.loc[dom,k] for dom in r.index for k in ('RMSE','KGE')},
                        'minutes':info['minutes']})
        tlog(f'{tag}: validation finished in {info["minutes"]} min')
    except Exception as e:
        traceback.print_exc()
        tlog(f'{tag}: FAILED ({e}), continuing with the next model')
        summary.append({'tag':tag,'features':len(feats),'status':f'failed: {e}'})

# ---------------------------------------------------------------
# Summary of all models validated in this call
# ---------------------------------------------------------------
pd.set_option('display.width',250); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',200)
for t,why in skipped.items(): summary.append({'tag':t,'status':f'skipped: {why}'})
print('\nindependent validation, all rows (XGBoost):')
print(pd.DataFrame(summary).round(3).to_string(index=False))
tlog(f'done; results in {base_dir}validationresults_<tag>/')