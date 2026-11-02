# CryoSCOPE SWE model dashboard: training runs (Optuna, SHAP, EDA), independent validation, validation explorer
# and SWE forecast predictions validated against HKL
# reads the outputs of the tuning script (models/, results/), the validation script (validationresults_<optuna_tag>/, one folder per model)
# and the ECXENS vs HKL forecast validation script (validation/<model>/<start date>/<prediction>/, copied from the forecast server)
# run: streamlit run swe-streamlit-app.py --server.port 8501 --server.address 0.0.0.0
import os,glob,json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# logos: snowcap.png (square), fmi.png (wide), cryoscope.png (footer banner with the funding statement)
LOGO_DIR=os.environ.get('LOGO_DIR','/home/ubuntu/data/cryoscope')
def logo(name):
    p=os.path.join(LOGO_DIR,name); return p if os.path.exists(p) else None

st.set_page_config(page_title='ML SWE',page_icon='❄️',layout='wide')

# CryoSCOPE logo at the top left of the sidebar, above the CryoSCOPE SWE title
if logo('cryo.png'):
    if hasattr(st,'logo'):
        try: st.logo(logo('cryo.png'),size='large')
        except TypeError: st.logo(logo('cryo.png'))   # streamlit < 1.39 has no size
    else:
        st.sidebar.image(logo('cryo.png'),width=180)

# FMI and SNOWCAP logos in the upper right corner of every page, before the page content
_hdr=st.columns([11,2.4,0.7])   # FMI column about as wide as its logo, so the two logos sit close together
if logo('fmi.png'): _hdr[1].image(logo('fmi.png'),width=260)
if logo('snowcap.png'): _hdr[2].image(logo('snowcap.png'),width=70)

SEASON_MONTHS=[9,10,11,12,1,2,3,4,5,6,7,8]
MONTH_NAMES={1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
PRED_STYLE={'SWE_observed':('observed','black','solid'),'SWE_xgboost':('XGBoost','#1f77b4','solid'),
            'SWE_climatology':('climatology','#2ca02c','dot'),'SWE_era5l':('ERA5-Land','#ff7f0e','dash')}
PEAK_MIN=10.0  # mm, same as peak_min in the validation script
ERR_MAX=150.0  # mm, fixed colour scale for RMSE, ubRMSE, MAE maps (all predictors): 0..ERR_MAX, values above shown red
ME_MAX=75.0    # mm, fixed symmetric colour scale for ME and difference maps: -ME_MAX..+ME_MAX

# SWE predictions page: prediction folders written by the forecast validation script, shown in this order (any others after them)
FC_ORDER=['ensmean','member01','wettest','driest']
# layout: validation/<model>/<start date>/<prediction>/, e.g. validation/FINmulti/20230210/wettest/swe_metrics.csv
#   <model>      model edition (optuna_tag) that made the ECXENS forecast, any name works (also models not in EDITION_NAMES);
#                'model' in <start date>/run_info.json overrides the folder name
#   <start date> forecast base date, YYYYMMDD (also YYYYMMDDTHHMMSS, YYYYMMDDHH or YYYY-MM-DD); 'base_time' in run_info.json
#                overrides it, and a folder name that is no date falls back to the most common (date minus lead day) in the scores
#   <prediction> ensmean, member01, wettest, driest, ... each with swe_metrics.csv and the png figures
# older layouts still work: validation/<model>/<prediction>/ (start date from the scores) and validation/<prediction>/ (model FC_DEFAULT_MODEL)
FC_DEFAULT_MODEL='FINmulti'
FC_SKIP_DIRS={'app_cache','tmp','__pycache__'}   # folders under validation/ that are never models, start dates or predictions
# names shown when a folder has no pred_info.json; member1, member01, ... are named 'ECXENS member N' automatically
FC_NAMES={'ensmean':'ECXENS ens mean','wettest':'ECXENS wettest member','driest':'ECXENS driest member'}
MILD_BIAS=50   # mm, same as MILD_BIAS in the forecast validation script (soft bias colours up to this, strong beyond)
# member_ranking.csv columns -> label
MEMBER_STATS={'mean':'domain mean SWE, all lead days','mean_last_day':'domain mean SWE, last lead day','p99':'99th percentile SWE','max':'max SWE'}
# scores in swe_metrics.csv -> (label, unit); 'snow_' versions use only pixels where forecast or HKL has at least 1 mm
FC_METRICS={'bias':('bias (forecast minus HKL)','mm'),'mae':('MAE','mm'),'rmse':('RMSE','mm'),'r':('correlation r',''),'kge':('KGE','')}

# training locations figures in the snowmodel directory: area -> file name prefix; shown on the Training run page
TRAIN_LOC_FIGS={'Scandinavia':'dem+fch_merged_fch25','Alps':'dem+fch_merged_alps_fch2'}
# figure suffix -> caption, in the order shown (the polar plot is left out on purpose)
TRAIN_LOC_KINDS=[('map_height','elevation'),('map_slope','slope'),('map_aspect','aspect'),
                 ('map_fch_2021','forest canopy height 2021'),('hist','distributions')]

# validation locations maps in the snowmodel directory: area -> file name; shown on the Validation page
VAL_LOC_FIGS={'Scandinavia':'validation_dem+fch_merged_fin25_10000_fscs1_scandi_map.png',
              'Alps':'validation_dem+fch_merged_fin2_10000_fscs1_alps_map.png'}

# model editions = optuna_tag of the training run -> name shown in the app; unknown tags are shown as they are
EDITION_NAMES={'FINmulti':'FINmulti','MANYfeatures':'MANYfeatures'}

# feature descriptions for the Features tab: name -> (group, description, units)
# units are ECMWF native units unless the fetch converted them; edit freely, unknown features still show up as 'other'
FEATURE_INFO={
    'latitude':('location','latitude','°N'),
    'longitude':('location','longitude','°E'),
    'height':('terrain','elevation','m'),
    'slope':('terrain','terrain slope','°'),
    'aspect':('terrain','terrain aspect','°'),
    'tcf':('land cover','tree cover fraction','0..1'),
    'laihv':('land cover','leaf area index, high vegetation','m² m⁻²'),
    'lailv':('land cover','leaf area index, low vegetation','m² m⁻²'),
    'swe_mean':('SNOWCAP climatology','SWE climatology, mean','mm'),
    'swe_max':('SNOWCAP climatology','SWE climatology, max','mm'),
    'swe_min':('SNOWCAP climatology','SWE climatology, min','mm'),
    'snowdepth_mean':('SNOWCAP climatology','snow depth climatology, mean',''),
    'snowdepth_max':('SNOWCAP climatology','snow depth climatology, max',''),
    'snowdepth_min':('SNOWCAP climatology','snow depth climatology, min',''),
    'tp':('forcing: precipitation','total precipitation','m'),
    'sf':('forcing: precipitation','snowfall','m w.e.'),
    'slhf':('forcing: energy','surface latent heat flux','J m⁻²'),
    'sshf':('forcing: energy','surface sensible heat flux','J m⁻²'),
    'ssr':('forcing: energy','surface net solar radiation','J m⁻²'),
    'str':('forcing: energy','surface net thermal radiation','J m⁻²'),
    't2':('forcing: temperature','2 m temperature','K'),
    'td2':('forcing: temperature','2 m dewpoint temperature','K'),
    'tmax_max':('forcing: temperature','daily maximum 2 m temperature','K'),
    'tmin_min':('forcing: temperature','daily minimum 2 m temperature','K'),
    'skt':('forcing: temperature','skin temperature','K'),
    'stl1':('forcing: temperature','soil temperature, level 1','K'),
    'u10':('forcing: wind & pressure','10 m wind, eastward','m s⁻¹'),
    'v10':('forcing: wind & pressure','10 m wind, northward','m s⁻¹'),
    'sp':('forcing: wind & pressure','surface pressure','Pa'),
    'ro':('reanalysis snow & hydrology','runoff','m'),
    'sro':('reanalysis snow & hydrology','surface runoff','m'),
    'ssro':('reanalysis snow & hydrology','sub-surface runoff','m'),
    'rsn':('reanalysis snow & hydrology','snow density','kg m⁻³'),
    'sd':('reanalysis snow & hydrology','snow depth water equivalent','m w.e.'),
    'smlt':('reanalysis snow & hydrology','snowmelt','m w.e.'),
    'es':('reanalysis snow & hydrology','snow evaporation','m w.e.'),
    'e':('reanalysis snow & hydrology','total evaporation','m w.e.'),
    'pev':('reanalysis snow & hydrology','potential evaporation','m'),
    'asn':('reanalysis snow & hydrology','snow albedo','0..1'),
    'tsn':('reanalysis snow & hydrology','temperature of snow layer','K'),
    'src':('reanalysis snow & hydrology','skin reservoir content','m w.e.'),
    'swvl1':('reanalysis snow & hydrology','volumetric soil water, layer 1','m³ m⁻³'),
    'swvl2':('reanalysis snow & hydrology','volumetric soil water, layer 2','m³ m⁻³'),
    'doy_cos':('time','day of year, cosine',''),
    'doy_sin':('time','day of year, sine',''),
}
# pressure level features q500, t700, z850, ...: one entry per variable instead of 15 dict lines
PLEV_VARS={'q':('specific humidity','kg kg⁻¹'),'t':('temperature','K'),'u':('wind, eastward','m s⁻¹'),
           'v':('wind, northward','m s⁻¹'),'z':('geopotential','m² s⁻²')}
def feature_info(f):
    if f in FEATURE_INFO: return FEATURE_INFO[f]
    if len(f)==4 and f[0] in PLEV_VARS and f[1:] in ('500','700','850'):
        d,u=PLEV_VARS[f[0]]; return 'forcing: upper air',f'{d}, {f[1:]} hPa',u
    if f.endswith('_lag1'):
        g,d,u=feature_info(f[:-5]); return g,f'{d}, previous day (lag 1)',u
    if f.endswith('_00'):
        g,d,u=feature_info(f[:-3]); return g,f'{d}, 00 UTC',u
    return 'other','',''

def edition_name(tag): return EDITION_NAMES.get(tag,tag)
def edition_label(tag,runs):
    n=next((r.get('n_features') for r in runs if r['optuna_tag']==tag),None)
    return f'{edition_name(tag)} ({n} features)' if n else edition_name(tag)

# ---------------------------------------------------------------
# Sidebar: navigation and paths
# ---------------------------------------------------------------
st.sidebar.title('❄️ CryoSCOPE ML SWE')
page=st.sidebar.radio('page',['Overview','Training run','Validation','Validation explorer','SWE predictions'])
with st.sidebar.expander('data paths',expanded=False):
    base=st.text_input('snowmodel directory',os.environ.get('SNOWMODEL_DIR','/home/ubuntu/data/cryoscope/snowmodel/'))
    optuna_db=st.text_input('optuna sqlite',os.environ.get('OPTUNA_DB','/home/ubuntu/data/ML/MLexperiments.sqlite3'))
    PRED_DIR=st.text_input('SWE prediction validation directory',os.environ.get('SWE_PRED_DIR',os.path.join(base,'validation')))

def read_json(p):
    try:
        with open(p) as f: return json.load(f)
    except Exception: return {}

# validation folders written by the validation script (validationresults_<optuna_tag>/), newest first
val_dirs=sorted((os.path.basename(p) for p in glob.glob(os.path.join(base,'validationresults*')) if os.path.isdir(p)),
                key=lambda d:-os.path.getmtime(os.path.join(base,d)))
def val_dir_tag(d):
    # model tag of a validation folder: optuna_tag in run_info.json, else the folder name after validationresults_
    tag=read_json(os.path.join(base,d,'run_info.json')).get('optuna_tag')
    if tag: return tag
    return d[len('validationresults_'):] if d.startswith('validationresults_') else d
def val_complete(d):
    # unfinished runs have complete=False in run_info.json; older folders without the flag count as finished
    return read_json(os.path.join(base,d,'run_info.json')).get('complete',True)
# validated model editions: tag -> folder (newest folder wins if two hold the same tag); folders without scores are left out
VAL_BY_TAG={}
for _d in val_dirs:
    if os.path.exists(os.path.join(base,_d,'validation_metrics.csv')): VAL_BY_TAG.setdefault(val_dir_tag(_d),_d)
def val_dir_for(stem):
    # validation folder whose run_info.json points to this training run
    for d in val_dirs:
        if read_json(os.path.join(base,d,'run_info.json')).get('training_stem')==stem: return d
    return None
MOD_DIR=os.path.join(base,'models'); RES_DIR=os.path.join(base,'results')
VAL_DIR=FIG_DIR=None   # set by pick_val_model() on the Validation and Validation explorer pages
if st.sidebar.button('reload files'):
    st.cache_data.clear(); st.cache_resource.clear()

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def mtime(p): return os.path.getmtime(p) if os.path.exists(p) else 0.0
def stamp(p): return pd.Timestamp(mtime(p),unit='s').tz_localize('UTC').tz_convert('Europe/Helsinki').strftime('%Y-%m-%d %H:%M')

@st.cache_data(show_spinner=False)
def _read_csv(p,mt): return pd.read_csv(p)
def read_csv(p): return _read_csv(p,mtime(p)) if os.path.exists(p) else None

def stretch(fn,*a,**k):
    # newer streamlit uses width='stretch', older use_container_width=True
    try: return fn(*a,width='stretch',**k)
    except Exception: return fn(*a,use_container_width=True,**k)
def img(p,caption=None):
    if os.path.exists(p): stretch(st.image,p,caption=caption)
    else: st.info(f'not found: {os.path.basename(p)}')
def chart(fig,**k): return stretch(st.plotly_chart,fig,**k)
def table(df,**k): return stretch(st.dataframe,df,**k)
def fmt(v,f):
    try: return f.format(v) if v is not None and pd.notna(v) else 'n/a'
    except (ValueError,TypeError): return str(v)

def quick_scores(o,p):
    o=np.asarray(o,float); p=np.asarray(p,float); ok=np.isfinite(o)&np.isfinite(p); o,p=o[ok],p[ok]
    if len(o)<2: return {'n':len(o)}
    e=p-o; so,sp=o.std(),p.std()
    r=np.corrcoef(o,p)[0,1] if so>0 and sp>0 else np.nan
    a=sp/so if so>0 else np.nan; b=p.mean()/o.mean() if o.mean()>0 else np.nan
    return {'n':len(o),'RMSE':np.sqrt(np.mean(e**2)),'ME':e.mean(),'MAE':np.abs(e).mean(),'r':r,
            'KGE':1-np.sqrt((r-1)**2+(a-1)**2+(b-1)**2),'peak obs':o.max(),'peak pred':p.max()}

def is_err(col): return col=='RMSE' or col.startswith('RMSE_') or col in ('ubRMSE','MAE')

def err_scale(err_max):
    # viridis from 0 to err_max, red above; the range ends 10 % above err_max so the red band shows in the colour bar
    top=err_max*1.1; f=err_max/top
    vir=px.colors.sample_colorscale('Viridis',[i/10 for i in range(11)])
    return [[f*i/10,c] for i,c in enumerate(vir)]+[[f+1e-6,'red'],[1.0,'red']],(0,top)

def color_args(s,col,err_max=ERR_MAX):
    v=pd.Series(s).replace([np.inf,-np.inf],np.nan).dropna()
    if v.empty: return 'Viridis',None
    if is_err(col): return err_scale(err_max)   # same limits for every predictor, domain and run
    if col.startswith('skill'): return 'RdBu',(-1,1)
    if col=='ME' or 'minus' in col: return 'RdBu_r',(-ME_MAX,ME_MAX)
    if col in ('r','pearson_r'): return 'Magma',(max(0.0,float(np.nanpercentile(v,2))),1)
    if col in ('height','dist_to_train_m'): return 'Viridis',(float(v.min()),float(v.max()))
    return 'Viridis',(0,float(np.nanpercentile(v,98)) or 1.0)

def map_fig(d,col,hover=None,height=600,title=None,rng=None,cscale=None,err_max=ERR_MAX):
    cs,cr=color_args(d[col],col,err_max)
    kw=dict(lat='latitude',lon='longitude',color=col,color_continuous_scale=cscale or cs,range_color=rng or cr,
            hover_data=hover,custom_data=['latitude','longitude'],height=height,title=title,
            zoom=3 if d['longitude'].max()-d['longitude'].min()>12 else 4.5,
            center=dict(lat=float(d['latitude'].mean()),lon=float(d['longitude'].mean())))
    try: fig=px.scatter_map(d,map_style='carto-positron',**kw)
    except AttributeError: fig=px.scatter_mapbox(d,mapbox_style='carto-positron',**kw)  # plotly < 5.24
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(margin=dict(l=0,r=0,t=40 if title else 10,b=0))
    if is_err(col) and rng is None and cscale is None:
        tv=list(np.linspace(0,err_max,4))
        fig.update_coloraxes(colorbar=dict(tickvals=tv+[err_max*1.05],ticktext=[f'{x:g}' for x in tv]+[f'>{err_max:g}']))
    return fig

def click_latlon(p):
    # lat/lon of a clicked map point; streamlit returns lat/lon directly for map traces,
    # customdata can be a list, a dict or missing (e.g. the red 'selected' marker), so try both
    lat,lon=p.get('lat'),p.get('lon')
    if lat is not None and lon is not None:
        try: return float(lat),float(lon)
        except (TypeError,ValueError): pass
    cd=p.get('customdata')
    if isinstance(cd,dict): cd=list(cd.values())
    if isinstance(cd,(list,tuple)) and len(cd)>=2:
        try: return float(cd[0]),float(cd[1])
        except (TypeError,ValueError): pass
    return None

def hbar(d,x,title,n):
    d=d.sort_values(x,ascending=False).head(n).iloc[::-1]
    fig=px.bar(d,x=x,y='feature',orientation='h',title=title,height=22*len(d)+120)
    fig.update_layout(margin=dict(l=0,r=10,t=40,b=0),yaxis_title=None)
    return fig

def train_vs_indep(h,vm):
    # tuning validation (held-out seasons at training locations, drove tuning and early stopping) vs independent validation locations
    v=vm[(~vm['model'].astype(str).str.startswith('baseline_'))&(vm['subset']=='all_rows')]
    bydom=h.get('valid_metrics_by_domain') or {}
    rows=[]
    for _,r in v.iterrows():
        dom=r['domain']
        tr=({'rmse':h.get('valid_rmse_all'),'kge':h.get('valid_kge_all')} if dom=='all' else bydom.get(dom,{}))
        rows.append({'domain':dom,'RMSE tuning valid':tr.get('rmse'),'RMSE independent':r.get('RMSE'),
                     'KGE tuning valid':tr.get('kge'),'KGE independent':r.get('KGE'),
                     'R² independent':r.get('R2'),'ME independent':r.get('ME')})
    return pd.DataFrame(rows)

def run_features(h,g=None,s=None):
    # feature list in model column order; older hyperparams files without 'features' fall back to the importance csv
    if h.get('features'): return list(h['features'])
    for d in (g,s):
        if d is not None and 'feature' in d: return d['feature'].tolist()
    return []

def feature_table(feats,g,s):
    t=pd.DataFrame({'#':np.arange(1,len(feats)+1),'feature':feats})
    info=[feature_info(f) for f in feats]
    t['group']=[i[0] for i in info]; t['description']=[i[1] for i in info]; t['units']=[i[2] for i in info]
    for d,col,lab in [(g,'gain','gain'),(s,'mean_abs_shap','shap')]:
        if d is None or col not in d: continue
        t=t.merge(d[['feature',col]],on='feature',how='left')
        tot=t[col].sum()
        t[f'{lab}_%']=100*t[col]/tot if tot>0 else np.nan
        t[f'{lab}_rank']=t[col].rank(ascending=False,method='min').astype('Int64')
    return t

# ---------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------
@st.cache_data(ttl=60,show_spinner=False)
def training_runs(mod_dir):
    out=[]
    for p in glob.glob(os.path.join(mod_dir,'hyperparams_swe_*.json')):
        try:
            with open(p) as f: h=json.load(f)
        except Exception: continue
        stem=os.path.basename(p)[len('hyperparams_swe_'):-len('.json')]  # variant_domain_optunatag
        parts=stem.split('_')
        h.setdefault('variant',parts[0]); h.setdefault('domain',parts[1] if len(parts)>1 else '')
        h['optuna_tag']=stem[len(h['variant'])+len(h['domain'])+2:]
        h['stem']=stem; h['modified']=stamp(p); h['_mtime']=mtime(p); h['_path']=p
        out.append(h)
    return sorted(out,key=lambda h:-h['_mtime'])

def use_val(d):
    # point the Validation pages to one validation folder
    global VAL_DIR,FIG_DIR
    VAL_DIR=os.path.join(base,d); FIG_DIR=os.path.join(VAL_DIR,'figs')

def pick_val_model():
    # model edition selection for the Validation and Validation explorer pages; shares the choice with the Training run page
    runs=training_runs(MOD_DIR)
    trained=list(dict.fromkeys(r['optuna_tag'] for r in runs))   # newest training run first
    tags=[t for t in trained if t in VAL_BY_TAG]+sorted(t for t in VAL_BY_TAG if t not in trained)
    todo=[t for t in trained if t not in VAL_BY_TAG]
    if not tags:
        st.warning(f'no validationresults_<tag>/ folders with validation_metrics.csv in {base}')
        if todo: st.code('python xgb-validate-cryo-swe.py missing',language='bash')
        return None
    keep=st.session_state.get('edition_keep')
    tag=st.radio('model edition',tags,index=tags.index(keep) if keep in tags else 0,horizontal=True,
                 format_func=lambda t:edition_label(t,runs)+('' if val_complete(VAL_BY_TAG[t]) else ' (running)'))
    st.session_state['edition_keep']=tag
    use_val(VAL_BY_TAG[tag])
    if todo:
        st.caption('not validated yet: '+', '.join(edition_label(t,runs) for t in todo)+
                   '. ')
    return tag

@st.cache_data(show_spinner='reading optuna study')
def load_study(db,name,mt):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        t=optuna.load_study(study_name=name,storage=f'sqlite:///{db}').trials_dataframe()
        t=t[t['state']=='COMPLETE']
        return t if 'values_0' in t else None
    except Exception:
        return None

@st.cache_resource(show_spinner='loading predictions (first load converts the csv.gz to parquet)')
def load_pred(path,mt):
    # parquet cache next to the predictions, in <validation folder>/app_cache/, so every model has its own
    pq=os.path.join(os.path.dirname(path),'app_cache',os.path.basename(path).replace('.csv.gz','.parquet'))
    if os.path.exists(pq) and mtime(pq)>=mt:
        return pd.read_parquet(pq)
    df=pd.read_csv(path,parse_dates=['utctime'])
    f64=df.select_dtypes('float64').columns.difference(['latitude','longitude'])
    df[f64]=df[f64].astype(np.float32)
    df['tag']=df['tag'].astype('category')
    try:
        os.makedirs(os.path.dirname(pq),exist_ok=True); df.to_parquet(pq,index=False)
    except Exception as e:
        print('parquet cache not written:',e)
    return df

@st.cache_resource(show_spinner=False)
def loc_index(path,mt):
    return load_pred(path,mt).groupby(['latitude','longitude'],sort=False).indices

@st.cache_data(show_spinner='computing per location scores')
def loc_table(path,mt):
    df=load_pred(path,mt)
    e=df['SWE_xgboost']-df['SWE_observed']
    agg=dict(tag=('tag','first'),height=('height','first'),n=('err','size'),obs_mean=('SWE_observed','mean'),
             obs_max=('SWE_observed','max'),pred_mean=('SWE_xgboost','mean'),ME=('err','mean'),MSE=('err2','mean'))
    if 'dist_to_train_m' in df: agg['dist_to_train_m']=('dist_to_train_m','first')
    t=df.assign(err=e,err2=e**2).groupby(['latitude','longitude'],sort=False).agg(**agg).reset_index()
    t['RMSE']=np.sqrt(t.pop('MSE')); t['tag']=t['tag'].astype(str)
    return t

@st.cache_data(show_spinner='computing daily domain means')
def domain_mean(path,mt):
    df=load_pred(path,mt); cols=[c for c in PRED_STYLE if c in df]
    return df.groupby([df['utctime'].dt.normalize().rename('date'),df['tag'].astype(str).rename('tag')])[cols].mean().reset_index()

def nearest(lt,lat,lon):
    d=(lt['latitude']-lat)**2+((lt['longitude']-lon)*np.cos(np.deg2rad(lat)))**2
    r=lt.loc[d.idxmin()]
    return float(r['latitude']),float(r['longitude'])

def fc_sort_key(p):
    # prediction folders in FC_ORDER first (ensmean, member01, wettest, driest), any others after them by name
    lab=os.path.basename(p.rstrip('/'))
    return (FC_ORDER.index(lab) if lab in FC_ORDER else len(FC_ORDER),lab)

def fc_subdirs(p):
    try:
        return sorted(d for d in os.listdir(p) if os.path.isdir(os.path.join(p,d)) and not d.startswith('.') and d not in FC_SKIP_DIRS)
    except OSError:
        return []

def fc_pred_dirs(p):
    # prediction folders directly under p (those with swe_metrics.csv), in FC_ORDER
    return sorted((os.path.join(p,d) for d in fc_subdirs(p) if os.path.exists(os.path.join(p,d,'swe_metrics.csv'))),key=fc_sort_key)

def fc_date_from_name(name):
    for f in ('%Y%m%d','%Y%m%dT%H%M%S','%Y%m%d%H','%Y-%m-%d'):
        try: return pd.to_datetime(name,format=f)
        except (ValueError,TypeError): pass
    return None

def fc_base_from_scores(pred):
    # most common (date minus lead day) in one prediction's scores
    try:
        d=pd.read_csv(os.path.join(pred,'swe_metrics.csv'),usecols=['date','lead_day'])
        m=(pd.to_datetime(d['date'])-pd.to_timedelta(d['lead_day'],unit='D')).mode()
        return m.iloc[0] if len(m) else None
    except Exception:
        return None

@st.cache_data(ttl=60,show_spinner=False)
def fc_runs(pdir,known=()):
    # forecast validation runs: validation/<model>/<start date>/<prediction>/swe_metrics.csv; one run = one model and start date.
    # newest start date first
    found=[]   # (run folder, model from the folder name)
    for m in fc_subdirs(pdir):
        mdir=os.path.join(pdir,m)
        if fc_pred_dirs(mdir): found.append((mdir,m))                  # old layout: validation/<model>/<prediction>/
        for d in fc_subdirs(mdir):
            if fc_pred_dirs(os.path.join(mdir,d)): found.append((os.path.join(mdir,d),m))
    if fc_pred_dirs(pdir): found.append((pdir,FC_DEFAULT_MODEL))      # oldest layout: validation/<prediction>/
    out=[]
    for run,model in found:
        preds=fc_pred_dirs(run)
        info=read_json(os.path.join(run,'run_info.json'))
        bt=None
        if info.get('base_time'):
            try: bt=pd.Timestamp(info['base_time'])
            except (ValueError,TypeError): bt=None
        if bt is None and run!=pdir: bt=fc_date_from_name(os.path.basename(run))
        if bt is None: bt=fc_base_from_scores(preds[0])
        out.append({'run':run,'preds':preds,'base_time':bt,'model':info.get('model') or model})
    return sorted(out,key=lambda r:r['base_time'] if r['base_time'] is not None else pd.Timestamp('1900-01-01'),reverse=True)

def fc_rel(run):
    return os.path.relpath(run,PRED_DIR) if os.path.normpath(run)!=os.path.normpath(PRED_DIR) else os.path.basename(PRED_DIR.rstrip('/'))

def fc_run_label(r):
    bt=r['base_time']
    when=(f"{bt:%Y-%m-%d}"+(f" {bt:%H} UTC" if bt.hour else '')) if bt is not None else 'start date unknown'
    return f"{when}  ({fc_rel(r['run'])}, {len(r['preds'])} predictions)"

def fc_name(pred_dir):
    # shown name: pred_info.json written by the validation script (e.g. 'ECXENS member 23 (wettest)'),
    # then FC_NAMES, then member<N> (member1, member01, ...) -> 'ECXENS member N'
    name=read_json(os.path.join(pred_dir,'pred_info.json')).get('name')
    if name: return name
    lab=os.path.basename(pred_dir.rstrip('/'))
    if lab in FC_NAMES: return FC_NAMES[lab]
    if lab.startswith('member') and lab[6:].isdigit(): return f'ECXENS member {int(lab[6:])}'
    return lab

def fc_metrics(preds):
    # daily scores of every prediction in the run, with 'pred' (folder) and 'name' (shown) columns
    parts=[]
    for p in preds:
        d=read_csv(os.path.join(p,'swe_metrics.csv'))
        if d is not None: parts.append(d.assign(pred=os.path.basename(p),name=fc_name(p)))
    if not parts: return None
    df=pd.concat(parts,ignore_index=True)
    df['date']=pd.to_datetime(df['date'])
    return df

@st.cache_data(show_spinner=False)
def _img_ratio(p,mt):
    try:
        from PIL import Image
        with Image.open(p) as im: w,h=im.size
        return w/h
    except Exception:
        return None
def img_ratio(p,default):
    # width / height of a png, so images side by side in st.columns get the same height
    r=_img_ratio(p,mtime(p)) if os.path.exists(p) else None
    return r or default

# ---------------------------------------------------------------
# Page: overview
# ---------------------------------------------------------------
def page_overview():
    st.title('CryoSCOPE snow water equivalent (SWE) ML model')
    st.caption('add caption here')
    runs=training_runs(MOD_DIR)
    trained=list(dict.fromkeys(r['optuna_tag'] for r in runs))
    c=st.columns(4)
    c[0].metric('training runs',len(runs))
    c[1].metric('latest run',runs[0]['optuna_tag'] if runs else 'none')
    c[2].metric('latest valid KGE',fmt(runs[0].get('valid_kge_all'),'{:.3f}') if runs else 'n/a')
    c[3].metric('validated models',f"{sum(t in VAL_BY_TAG for t in trained)} of {len(trained)}")

    if runs:
        st.subheader('Training runs')
        cols=['stem','modified','valid_rmse_all','valid_kge_all','valid_r2_all','valid_bias','valid_rmse_snowon',
              'n_estimators_earlystop','n_features','pareto_size','chosen_trial','tweedie_variance_power','max_depth','learning_rate']
        t=pd.DataFrame([{c:h.get(c) for c in cols} for h in runs])
        num=cols[2:]; t[num]=t[num].apply(pd.to_numeric,errors='coerce')
        t.insert(2,'independent validation',[h['optuna_tag'] in VAL_BY_TAG for h in runs])
        table(t.round(4),hide_index=True)
        tt=t.dropna(subset=['valid_rmse_all','valid_kge_all'])
        if len(tt)>1:
            fig=px.scatter(tt,x='valid_rmse_all',y='valid_kge_all',text='stem',color='n_features',height=420,
                           title='runs on the tuning validation seasons (optimistic)')
            fig.update_traces(textposition='top center',marker=dict(size=12))
            chart(fig)
        with st.expander('features used in each training run',expanded=False):
            used={h['stem']:run_features(h) for h in runs}
            allf=list(dict.fromkeys(f for fs in used.values() for f in fs))
            if allf:
                m=pd.DataFrame({'feature':allf,'group':[feature_info(f)[0] for f in allf],'description':[feature_info(f)[1] for f in allf]})
                for k,fs in used.items(): m[k]=m['feature'].isin(fs)
                table(m,hide_index=True,height=min(36*len(m)+40,800))
            else:
                st.info('no feature lists stored in the hyperparams files')

    # independent validation of every validated model side by side; baselines are the same in every folder, shown once
    parts=[]
    for tag,d in VAL_BY_TAG.items():
        v=read_csv(os.path.join(base,d,'validation_metrics.csv'))
        if v is None: continue
        v=v[v['subset']=='all_rows'].copy()
        xg=~v['model'].astype(str).str.startswith('baseline_')
        v['model']=v['model'].astype(str)
        v.loc[xg,'model']='XGBoost '+edition_label(tag,runs)
        parts.append(v)
    if parts:
        vm=pd.concat(parts,ignore_index=True).drop_duplicates(['model','domain'])
        st.subheader('Independent validation (validation locations, all rows)')
        c1,c2=st.columns(2)
        for col,m in [(c1,'RMSE'),(c2,'KGE')]:
            with col:
                st.markdown(f'**{m}**')
                table(vm.pivot_table(index='model',columns='domain',values=m).round(3))
                chart(px.bar(vm,x='domain',y=m,color='model',barmode='group',height=360,
                             labels={'model':''}).update_layout(legend=dict(orientation='h',y=-0.25)),key=f'ov_{m}')
        todo=[t for t in trained if t not in VAL_BY_TAG]
        if todo: st.caption('not validated yet: '+', '.join(edition_label(t,runs) for t in todo))

# ---------------------------------------------------------------
# Page: training run
# ---------------------------------------------------------------
def page_training():
    runs=training_runs(MOD_DIR)
    if not runs:
        st.warning(f'no hyperparams_swe_*.json in {MOD_DIR}'); return

    # model edition (optuna_tag), newest first; kept in session_state so it survives page switches
    eds=list(dict.fromkeys(r['optuna_tag'] for r in runs))
    keep=st.session_state.get('edition_keep')
    ed=st.radio('model edition',eds,index=eds.index(keep) if keep in eds else 0,horizontal=True,
                format_func=lambda t:edition_label(t,runs))
    st.session_state['edition_keep']=ed
    er=[r for r in runs if r['optuna_tag']==ed]
    stem=st.selectbox('training run',[r['stem'] for r in er],key=f'run_{ed}') if len(er)>1 else er[0]['stem']
    h=next(r for r in er if r['stem']==stem)
    P=lambda name,ext='png': os.path.join(RES_DIR,f'{name}_{stem}.{ext}')

    st.title(f'Training run: {edition_name(h["optuna_tag"])}')
    st.caption(f"variant {h['variant']} | domain {h['domain']} | objective {h.get('objective')} | split by {h.get('split_by','?')} | "
               f"train seasons {h.get('train_seasons','?')} | valid seasons {h.get('valid_seasons','?')} | "
               f"{h.get('n_features','?')} features | written {h['modified']}")
    cards=[('valid RMSE (mm)','valid_rmse_all','{:.2f}'),('valid KGE','valid_kge_all','{:.3f}'),('valid R²','valid_r2_all','{:.3f}'),
           ('bias (mm)','valid_bias','{:+.2f}'),('RMSE snow-on (mm)','valid_rmse_snowon','{:.2f}'),('trees','n_estimators_earlystop','{}')]
    for col,(lab,k,f) in zip(st.columns(len(cards)),cards): col.metric(lab,fmt(h.get(k),f))
    st.caption('These scores use the validation seasons that also drove tuning and early stopping, so they are optimistic. '
               'See the Validation page for the independent locations.')
    vd=val_dir_for(stem)
    vmr=read_csv(os.path.join(base,vd,'validation_metrics.csv')) if vd else None
    if vmr is not None:
        st.markdown(f'**Independent validation of this run** (`{vd}`, validation locations not used in training, all rows)')
        table(train_vs_indep(h,vmr).round(3),hide_index=True)
    else:
        st.info(f"no independent validation for this run yet: python xgb-validate-cryo-swe.py {h['optuna_tag']}")

    tabs=st.tabs(['Features','Tuning','Fit','Feature importance','SHAP','EDA','Config','All figures','Compare editions','Training locations'])

    with tabs[0]:
        gi=read_csv(P('gain_importance','csv')); si=read_csv(P('shap_importance','csv'))
        feats=run_features(h,gi,si)
        if not feats:
            st.info('no feature list in the hyperparams json and no importance csv for this run')
        else:
            ft=feature_table(feats,gi,si)
            c=st.columns(4)
            c[0].metric('features',len(ft))
            c[1].metric('feature groups',ft['group'].nunique())
            c[2].metric('target','SWE (mm)')
            c[3].metric('input file',h.get('input_file','n/a'))
            c1,c2=st.columns([3,1])
            q=c1.text_input('filter by name, group or description','')
            sortby=c2.selectbox('sort by',[x for x in ['#','shap_rank','gain_rank','group','feature'] if x in ft])
            v=ft
            if q:
                ql=q.lower()
                v=ft[ft[['feature','group','description']].apply(lambda r:' '.join(r).lower(),axis=1).str.contains(ql,regex=False)]
            v=v.sort_values(sortby,kind='stable')
            cfg={'#':st.column_config.NumberColumn('#',help='column order in the model'),
                 'gain':None,'mean_abs_shap':st.column_config.NumberColumn('mean |SHAP|',format='%.4f')}
            if 'gain_%' in v: cfg['gain_%']=st.column_config.ProgressColumn('gain %',format='%.1f',min_value=0,max_value=float(ft['gain_%'].max() or 1))
            if 'shap_%' in v: cfg['shap_%']=st.column_config.ProgressColumn('|SHAP| %',format='%.1f',min_value=0,max_value=float(ft['shap_%'].max() or 1))
            table(v,hide_index=True,column_config=cfg,height=min(36*len(v)+40,900))
            st.caption('Importance is from this run (gain on all trees, mean |SHAP| on a validation sample, log-SWE margin). '
                       'Descriptions come from FEATURE_INFO at the top of the app.')
            st.download_button('download feature table (csv)',ft.to_csv(index=False),file_name=f'features_{stem}.csv',mime='text/csv')
            share=[x for x in ['shap_%','gain_%'] if x in ft]
            if share:
                grp=ft.groupby('group')[share].sum().reset_index().melt(id_vars='group',var_name='measure',value_name='share %')
                fig=px.bar(grp,x='share %',y='group',color='measure',barmode='group',orientation='h',height=60+45*ft['group'].nunique(),
                           title='importance share by feature group')
                fig.update_layout(yaxis={'categoryorder':'total ascending'},yaxis_title=None,margin=dict(l=0,r=10,t=40,b=0))
                chart(fig)

    with tabs[1]:
        par=read_csv(P('optuna_pareto','csv'))
        allt=load_study(optuna_db,h['study_name'],mtime(optuna_db)) if h.get('study_name') and os.path.exists(optuna_db) else None
        if par is None and allt is None:
            st.info('no Pareto csv or multi-objective optuna study found for this run')
        else:
            fig=go.Figure()
            if allt is not None:
                fig.add_scatter(x=allt['values_0'],y=allt['values_1'],mode='markers',name=f'all trials ({len(allt)})',
                                marker=dict(color='lightgray',size=7),customdata=allt['number'],
                                hovertemplate='trial %{customdata}<br>RMSE %{x:.3f}<br>KGE %{y:.3f}<extra></extra>')
            if par is not None:
                par=par.sort_values('valid_rmse').round(4)
                hov=[c for c in par.columns if c not in ('valid_rmse','valid_kge')]
                fig.add_scatter(x=par['valid_rmse'],y=par['valid_kge'],mode='lines+markers',name='Pareto front',
                                line=dict(color='crimson',width=1),
                                marker=dict(size=10,color=par['std_ratio'],colorscale='Viridis',showscale=True,colorbar=dict(title='std ratio')),
                                customdata=par[hov].values,
                                hovertemplate='<br>'.join(f'{c}: %{{customdata[{i}]}}' for i,c in enumerate(hov))+'<extra></extra>')
                if h.get('rmse_tol') is not None:
                    fig.add_vline(x=par['valid_rmse'].min()*(1+h['rmse_tol']),line_dash='dot',line_color='gray',
                                  annotation_text=f"RMSE_TOL {h['rmse_tol']}")
            if h.get('chosen_trial_rmse') is not None:
                fig.add_scatter(x=[h['chosen_trial_rmse']],y=[h['chosen_trial_kge']],mode='markers',name=f"chosen trial {h.get('chosen_trial')}",
                                marker=dict(symbol='circle-open',size=20,color='black',line=dict(width=3)))
            fig.update_layout(xaxis_title='valid RMSE',yaxis_title='valid KGE',height=560,legend=dict(orientation='h',y=1.08))
            chart(fig)
            if par is not None:
                st.markdown('**Pareto front trials** (std ratio below 1 means too flat, high SWE underestimated)')
                table(par,hide_index=True)
        img(P('optuna_history'),'Optuna Pareto front and optimization history')

    with tabs[2]:
        c1,c2=st.columns([1,1])
        with c1: img(P('valid_scatter'),'observed vs predicted SWE, validation seasons')
        with c2:
            st.markdown('**Scores on validation seasons**')
            keys=[k for k in h if k.startswith('valid_') and not isinstance(h[k],dict)]
            table(pd.DataFrame({'metric':keys,'value':[h[k] for k in keys]}),hide_index=True)
            if isinstance(h.get('valid_metrics_by_domain'),dict):
                st.markdown('**By domain**')
                table(pd.DataFrame(h['valid_metrics_by_domain']).T.round(3))

    with tabs[3]:
        g=read_csv(P('gain_importance','csv')); s=read_csv(P('shap_importance','csv'))
        nmax=max(len(g) if g is not None else 0,len(s) if s is not None else 0,1)
        n=st.slider('features shown',5,max(5,nmax),min(25,max(5,nmax)))
        c1,c2=st.columns(2)
        for col,d,x,title in [(c1,g,'gain','XGBoost gain'),(c2,s,'mean_abs_shap','mean |SHAP| (log-SWE margin)')]:
            with col:
                if d is None: st.info(f'no {title} file')
                else: chart(hbar(d,x,title,n))

    with tabs[4]:
        st.caption('With reg:tweedie the SHAP values are in log(SWE) margin units: contributions multiply on the SWE scale.')
        c1,c2=st.columns(2)
        with c1: img(P('shap_beeswarm'),'SHAP beeswarm')
        with c2: img(P('shap_bar'),'mean |SHAP|')
        deps=sorted(glob.glob(os.path.join(RES_DIR,f'shap_dep_*_{stem}.png')))
        if deps:
            st.markdown('**Dependence plots, top features**')
            cols=st.columns(2)
            for i,p in enumerate(deps):
                with cols[i%2]: img(p,os.path.basename(p)[len('shap_dep_'):-len(f'_{stem}.png')])

    with tabs[5]:
        c1,c2=st.columns(2)
        with c1: img(P('eda_swe_hist'),'snow-on SWE distribution, train vs valid')
        with c2: img(P('eda_swe_seasonal'),'mean SWE by month, train vs valid')

    with tabs[6]:
        meta={'objective','eval_metric','tree_method','num_parallel_tree','variant','domain','features','n_features',
              'split_by','snow_start_month','train_seasons','valid_seasons','tuning','rmse_tol','input_file','study_name'}
        hp={k:v for k,v in h.items() if k not in meta and not k.startswith(('valid_','chosen_','_')) and k not in ('stem','modified','optuna_tag','pareto_size','n_estimators_earlystop')}
        c1,c2=st.columns(2)
        with c1:
            st.markdown('**Chosen hyperparameters**')
            table(pd.DataFrame({'parameter':list(hp),'value':[str(v) for v in hp.values()]}),hide_index=True)
        with c2:
            st.markdown(f"**Features ({len(h.get('features',[]))})**")
            st.code(', '.join(h.get('features',[])),language=None)
            st.markdown('**Setup**')
            st.json({k:h[k] for k in meta if k in h and k!='features'},expanded=False)
        mp=os.path.join(MOD_DIR,f'mdl_swe_{stem}.json')
        st.caption(f'model file: {mp} ({mtime(mp) and os.path.getsize(mp)/1e6:.1f} MB)' if os.path.exists(mp) else f'model file missing: {mp}')
        st.download_button('download hyperparams json',json.dumps({k:v for k,v in h.items() if not k.startswith('_')},indent=2),
                           file_name=os.path.basename(h['_path']),mime='application/json')

    with tabs[7]:
        # every figure the training script wrote for this run: results/*_<variant>_<domain>_<optuna_tag>.png
        figs=sorted(glob.glob(os.path.join(RES_DIR,f'*_{stem}.png')))
        if not figs:
            st.info(f'no figures for {stem} in {RES_DIR}')
        else:
            names=[os.path.basename(p) for p in figs]
            pick=st.multiselect(f'{len(figs)} figures in {RES_DIR}',names,default=names,key='train_figs')
            ncol=st.radio('columns',[1,2,3],index=1,horizontal=True,key='train_figs_cols')
            cols=st.columns(ncol)
            for i,p in enumerate([p for p in figs if os.path.basename(p) in pick]):
                with cols[i%ncol]: img(p,os.path.basename(p)[:-len(f'_{stem}.png')])

    with tabs[8]:
        # newest run of each edition for the same domain as the selected run
        seen={}
        for r in runs:
            if r['domain']==h['domain']: seen.setdefault(r['optuna_tag'],r)
        pick=st.multiselect('editions to compare',list(seen),default=list(seen),
                            format_func=lambda t:edition_label(t,runs),key='cmp_eds')
        cmp=[seen[t] for t in pick]
        if len(cmp)<2:
            st.info('pick at least two editions')
        else:
            # scores: tuning validation seasons (same seasons for every edition) and independent locations if validated
            rows={}
            for r in cmp:
                d={'features':r.get('n_features'),'trees':r.get('n_estimators_earlystop'),
                   'tuning valid RMSE (mm)':r.get('valid_rmse_all'),'tuning valid KGE':r.get('valid_kge_all'),
                   'tuning valid R²':r.get('valid_r2_all'),'tuning valid bias (mm)':r.get('valid_bias'),
                   'tuning valid RMSE snow-on (mm)':r.get('valid_rmse_snowon')}
                for dom,m in (r.get('valid_metrics_by_domain') or {}).items():
                    d[f'tuning valid RMSE {dom}']=m.get('rmse'); d[f'tuning valid KGE {dom}']=m.get('kge')
                rvd=val_dir_for(r['stem'])
                rvm=read_csv(os.path.join(base,rvd,'validation_metrics.csv')) if rvd else None
                if rvm is not None:
                    for _,x in train_vs_indep(r,rvm).iterrows():
                        d[f"independent RMSE {x['domain']}"]=x['RMSE independent']
                        d[f"independent KGE {x['domain']}"]=x['KGE independent']
                rows[edition_name(r['optuna_tag'])]=d
            st.markdown('**Scores**')
            table(pd.DataFrame(rows).apply(pd.to_numeric,errors='coerce').round(3))
            missing=[edition_name(r['optuna_tag']) for r in cmp if val_dir_for(r['stem']) is None]
            if missing:
                st.caption('no independent validation yet for: '+', '.join(missing))

            # Pareto fronts on top of each other, chosen trial circled
            fig=go.Figure()
            for r in cmp:
                par=read_csv(os.path.join(RES_DIR,f"optuna_pareto_{r['stem']}.csv"))
                if par is None: continue
                par=par.sort_values('valid_rmse')
                fig.add_scatter(x=par['valid_rmse'],y=par['valid_kge'],mode='lines+markers',name=edition_name(r['optuna_tag']))
                if r.get('chosen_trial_rmse') is not None:
                    fig.add_scatter(x=[r['chosen_trial_rmse']],y=[r['chosen_trial_kge']],mode='markers',showlegend=False,
                                    marker=dict(symbol='circle-open',size=18,color='black',line=dict(width=2)))
            if fig.data:
                fig.update_layout(xaxis_title='valid RMSE',yaxis_title='valid KGE',height=480,
                                  title='Pareto fronts (circle = chosen trial)',legend=dict(orientation='h',y=1.08))
                chart(fig,key='cmp_pareto')

            # SHAP share per feature and per feature group
            parts=[]
            for r in cmp:
                s=read_csv(os.path.join(RES_DIR,f"shap_importance_{r['stem']}.csv"))
                if s is not None and s['mean_abs_shap'].sum()>0:
                    parts.append(s.assign(**{'share %':100*s['mean_abs_shap']/s['mean_abs_shap'].sum(),
                                             'edition':edition_name(r['optuna_tag'])}))
            if parts:
                imp=pd.concat(parts)
                nf=max(6,int(imp['feature'].nunique()))
                n=st.slider('features shown',5,nf,min(25,nf),key='cmp_n')
                top=imp.groupby('feature')['share %'].max().nlargest(n).index
                d=imp[imp['feature'].isin(top)]
                order=d.groupby('feature')['share %'].max().sort_values().index.tolist()
                f=px.bar(d,x='share %',y='feature',color='edition',barmode='group',orientation='h',height=28*len(order)+140,
                         category_orders={'feature':order[::-1]},title='mean |SHAP| share per feature (log-SWE margin)')
                f.update_layout(yaxis_title=None,margin=dict(l=0,r=10,t=40,b=0),legend=dict(orientation='h',y=1.04))
                chart(f,key='cmp_shap')
                imp['group']=imp['feature'].map(lambda x:feature_info(x)[0])
                grp=imp.groupby(['edition','group'])['share %'].sum().reset_index()
                f=px.bar(grp,x='share %',y='group',color='edition',barmode='group',orientation='h',
                         height=60+45*grp['group'].nunique(),title='mean |SHAP| share per feature group')
                f.update_layout(yaxis={'categoryorder':'total ascending'},yaxis_title=None,margin=dict(l=0,r=10,t=40,b=0))
                chart(f,key='cmp_grp')
                st.caption('Shares are within each edition, so MANYfeatures spreads the same total over more features.')

    with tabs[9]:
        # terrain and forest of the training locations (same for every run): maps of height, slope, aspect, canopy height and histograms
        areas=[a for a in TRAIN_LOC_FIGS if glob.glob(os.path.join(base,TRAIN_LOC_FIGS[a]+'_*.png'))]
        if not areas:
            st.info(f"no training location figures ({', '.join(v+'_*.png' for v in TRAIN_LOC_FIGS.values())}) in {base}")
        else:
            opts=areas+(['both side by side'] if len(areas)>1 else [])
            area=st.radio('area',opts,horizontal=True,key='trainloc_area')
            st.caption('Terrain and forest canopy height at the training locations. These describe the training data set, '
                       'so they are the same for every training run.')
            if area=='both side by side':
                for kind,cap in TRAIN_LOC_KINDS:
                    figs=[os.path.join(base,f'{TRAIN_LOC_FIGS[a]}_{kind}.png') for a in areas]
                    if not any(os.path.exists(f) for f in figs): continue
                    st.markdown(f'**{cap}**')
                    for col,a,f in zip(st.columns(len(areas)),areas,figs):
                        with col: img(f,a)
            else:
                pre=os.path.join(base,TRAIN_LOC_FIGS[area])
                maps=[(f'{pre}_{k}.png',c) for k,c in TRAIN_LOC_KINDS if k.startswith('map_') and os.path.exists(f'{pre}_{k}.png')]
                cols=st.columns(2)
                for i,(f,c) in enumerate(maps):
                    with cols[i%2]: img(f,f'{area}: {c}')
                for k,c in TRAIN_LOC_KINDS:
                    if not k.startswith('map_') and os.path.exists(f'{pre}_{k}.png'): img(f'{pre}_{k}.png',f'{area}: {c}')

# ---------------------------------------------------------------
# Page: validation
# ---------------------------------------------------------------
def page_validation():
    tag=pick_val_model()
    if tag is None: return
    vpath=os.path.join(VAL_DIR,'validation_metrics.csv')
    vm=read_csv(vpath)
    if vm is None:
        st.warning(f'no validation_metrics.csv in {VAL_DIR}'); return
    st.title(f'Independent validation: {edition_name(tag)}')
    info=read_json(os.path.join(VAL_DIR,'run_info.json'))
    st.caption(f'validation locations not used in training | results in {VAL_DIR} | written {stamp(vpath)}'
               + (f" | model from training run {info['training_stem']}" if info.get('training_stem') else '')
               + (' | validation still running or interrupted, results incomplete' if info.get('complete') is False else ''))

    models=[m for m in vm['model'].unique() if not str(m).startswith('baseline_')]
    c1,c2,c3=st.columns([1,1,1])
    mname=models[0] if len(models)==1 else c1.selectbox('model',models)
    dom=c2.selectbox('evaluated on',vm.loc[vm['model']==mname,'domain'].unique().tolist())
    subset=c3.radio('rows',['all_rows','snow_rows'],horizontal=True,
                    format_func=lambda s:'all rows' if s=='all_rows' else 'snow rows (obs or pred > 0)')
    label=f'{mname}model_{dom}'
    bases=[c[len('skill_MSE_vs_'):] for c in vm.columns if c.startswith('skill_MSE_vs_')]
    row=vm[(vm['model']==mname)&(vm['domain']==dom)&(vm['subset']==subset)]
    if row.empty:
        st.warning('no scores for this selection'); return
    r=row.iloc[0]
    cards=[('RMSE (mm)',r.get('RMSE'),'{:.1f}'),('ubRMSE (mm)',r.get('ubRMSE'),'{:.1f}'),('ME (mm)',r.get('ME'),'{:+.1f}'),
           ('R²',r.get('R2'),'{:.3f}'),('KGE',r.get('KGE'),'{:.3f}')]+[(f'MSE skill vs {b}',r.get(f'skill_MSE_vs_{b}'),'{:.2f}') for b in bases]
    for col,(lab,v,f) in zip(st.columns(len(cards)),cards): col.metric(lab,fmt(v,f))

    tabs=st.tabs(['Scores','Map','Monthly & snow year','Snow season','Figures','Training vs validation','Feature importance','Validation locations'])

    with tabs[0]:
        comp=vm[(vm['domain']==dom)&(vm['subset']==subset)&vm['model'].isin([mname]+[f'baseline_{b}' for b in bases])].copy()
        comp['predictor']=comp['model'].str.replace('baseline_','',regex=False).replace({mname:f'XGBoost ({edition_name(tag)})'})
        show=[c for c in ['n','obs_mean','pred_mean','ME','MAE','MedAE','P95_AE','RMSE','ubRMSE','nRMSE','PBIAS_%','R2','pearson_r',
                          'spearman_r','explained_var','std_ratio','KGE','POD','FAR','CSI','accuracy','HSS'] if c in comp]
        show+=[c for c in comp.columns if c.startswith(('skill_','RMSE_','n_')) and c not in show]   # skill against each baseline, on the same rows
        table(comp.set_index('predictor')[show].T.round(3))
        st.caption('Detection scores (POD, FAR, CSI, HSS) use a 1 mm snow threshold. For baselines, snow rows are rows where obs or that baseline is above 0.')
        long=comp.melt(id_vars='predictor',value_vars=[c for c in ['RMSE','ubRMSE','MAE'] if c in comp],var_name='metric')
        c1,c2=st.columns(2)
        with c1: chart(px.bar(long,x='metric',y='value',color='predictor',barmode='group',height=380,title='errors (mm)'))
        with c2: chart(px.bar(comp,x='predictor',y=[c for c in ['R2','KGE','pearson_r'] if c in comp],barmode='group',height=380,title='skill scores'))

    with tabs[1]:
        if dom=='all':
            st.info('per location maps exist for single domains only, pick Scandinavia or Alps')
        else:
            pl=read_csv(os.path.join(VAL_DIR,f'per_location_{label}.csv'))
            if pl is None: st.info(f'no per_location_{label}.csv')
            else:
                opts=[c for c in pl.columns if c not in ('latitude','longitude') and pd.api.types.is_numeric_dtype(pl[c])]
                rmse_base=[c for c in opts if c.startswith('RMSE_')]
                # defaults: XGBoost RMSE on the left, ERA5-Land RMSE (or the first baseline RMSE) on the right
                d1='RMSE' if 'RMSE' in opts else opts[0]
                d2=next((c for c in rmse_base if 'era5' in c.lower()),rmse_base[0] if rmse_base else opts[min(1,len(opts)-1)])
                emax=st.number_input('red above (mm), RMSE/MAE maps',value=ERR_MAX,min_value=10.0,step=25.0)
                hov={c:':.2f' for c in ['RMSE','ME','r','obs_mean','pred_mean','n','dist_to_train_m']+rmse_base if c in pl}
                for i,(pane,default) in enumerate(zip(st.columns(2),[d1,d2])):
                    with pane:
                        col=st.selectbox(f'map {i+1}: color by',opts,index=opts.index(default),key=f'valmap_col{i}_{label}')
                        chart(map_fig(pl,col,hover=hov,height=560,err_max=emax),key=f'valmap{i}')
                        if is_err(col):
                            st.caption(f'{int((pl[col]>emax).sum())} of {len(pl)} locations above {emax:g} mm (red)')
                        chart(px.histogram(pl,x=col,nbins=80,height=260,title=f'{col} across {len(pl)} locations'),key=f'valhist{i}')

    with tabs[2]:
        mt=read_csv(os.path.join(VAL_DIR,'validation_metrics_monthly.csv'))
        if mt is not None:
            mt=mt[(mt['model']==mname)&(mt['domain']==dom)].copy()
            mt['mon']=mt['month'].astype(int).map(MONTH_NAMES)
            order={'mon':[MONTH_NAMES[m] for m in SEASON_MONTHS]}
            c1,c2=st.columns(2)
            with c1:
                chart(px.bar(mt.melt(id_vars='mon',value_vars=['RMSE','ubRMSE','ME']),x='mon',y='value',color='variable',
                             barmode='group',category_orders=order,height=380,title='monthly errors (mm)',labels={'mon':'month','value':'mm'}))
            with c2:
                chart(px.line(mt.melt(id_vars='mon',value_vars=['obs_mean','pred_mean']),x='mon',y='value',color='variable',markers=True,
                              category_orders=order,height=380,title='monthly mean SWE (mm)',labels={'mon':'month','value':'mm'}))
            sk=[c for c in ['KGE','R2','explained_var'] if c in mt]
            if sk:
                chart(px.line(mt.melt(id_vars='mon',value_vars=sk),x='mon',y='value',color='variable',markers=True,
                              category_orders=order,height=340,title='monthly skill',labels={'mon':'month','value':''}))
            with st.expander('monthly scores table'):
                table(mt.drop(columns=['model','domain','mon']).round(3),hide_index=True)
        sy=read_csv(os.path.join(VAL_DIR,'validation_metrics_snowyear.csv'))
        if sy is not None:
            sy=sy[(sy['model']==mname)&(sy['domain']==dom)]
            obs=sy.drop_duplicates('season')[['season','obs_mean']].rename(columns={'obs_mean':'value'}).assign(predictor='observed')
            mean=pd.concat([obs,sy[['season','predictor','pred_mean']].rename(columns={'pred_mean':'value'})])
            c=st.columns(3)
            with c[0]: chart(px.line(sy,x='season',y='RMSE',color='predictor',markers=True,height=360,title='RMSE per snow year (mm)'))
            with c[1]:
                f=px.line(sy,x='season',y='ME',color='predictor',markers=True,height=360,title='ME per snow year (mm)'); f.add_hline(y=0,line_color='black',line_width=0.5); chart(f)
            with c[2]: chart(px.line(mean,x='season',y='value',color='predictor',markers=True,height=360,title='mean SWE per snow year (mm)'))
            with st.expander('snow year scores table'):
                table(sy.drop(columns=[c for c in ['model','domain','period'] if c in sy]).round(3),hide_index=True)

    with tabs[3]:
        ss=read_csv(os.path.join(VAL_DIR,'validation_metrics_snowseason.csv'))
        if ss is None or dom=='all':
            st.info('snow season metrics exist for single domains only' if dom=='all' else 'no validation_metrics_snowseason.csv')
        else:
            s=ss[(ss['model']==mname)&(ss['domain']==dom)].set_index('predictor').drop(columns=['model','domain'])
            c1,c2=st.columns([1,2])
            with c1:
                st.markdown('**Peak SWE, timing and duration**')
                table(s.T.round(2))
                st.caption(f'timing scores (peak, onset, melt-out) only for location seasons with observed peak ≥ {PEAK_MIN:g} mm; onset and melt-out use 1 mm')
            with c2:
                stab=read_csv(os.path.join(VAL_DIR,f'snowseason_{label}.csv'))
                if stab is not None:
                    preds=[c[len('peak_'):] for c in stab.columns if c.startswith('peak_SWE_') and c!='peak_SWE_observed' and not c.startswith('peak_dos')]
                    pc=st.radio('predictor',preds,horizontal=True,format_func=lambda c:PRED_STYLE.get(c,(c,))[0])
                    smp=stab.dropna(subset=[f'peak_{pc}'])
                    smp=smp.sample(min(20000,len(smp)),random_state=1)   # baselines can be NaN at some locations, so sample from the rows left
                    lim=float(np.nanpercentile(np.r_[smp['peak_SWE_observed'],smp[f'peak_{pc}']],99.5))
                    f=px.scatter(smp,x='peak_SWE_observed',y=f'peak_{pc}',opacity=0.3,render_mode='webgl',height=420,
                                 hover_data=['latitude','longitude','season'],title='peak SWE per location season (mm)',
                                 labels={'peak_SWE_observed':'observed peak',f'peak_{pc}':'predicted peak'})
                    f.add_scatter(x=[0,lim],y=[0,lim],mode='lines',line=dict(color='red',dash='dash'),showlegend=False)
                    f.update_layout(xaxis_range=[0,lim],yaxis_range=[0,lim]); chart(f)
                    snowy=stab[stab['peak_SWE_observed']>=PEAK_MIN]
                    errs=pd.concat([pd.DataFrame({'error (days)':(snowy[f'{m}_{c}']-snowy[f'{m}_SWE_observed']).clip(-90,90),
                                                  'metric':m,'predictor':PRED_STYLE.get(c,(c,))[0]}).dropna()
                                    for m in ['peak_dos','onset','meltout'] for c in ['SWE_xgboost']+[p for p in preds if p!='SWE_xgboost'] if f'{m}_{c}' in snowy])
                    chart(px.histogram(errs,x='error (days)',color='predictor',facet_col='metric',barmode='overlay',nbins=60,opacity=0.55,height=330,
                                       title='timing errors, predicted minus observed (clipped at ±90 d)'))

    with tabs[4]:
        figs=[p for p in sorted(glob.glob(os.path.join(FIG_DIR,f'*_{label}.png'))) if not os.path.basename(p).startswith('distance_')]
        figs+=[p for p in [os.path.join(FIG_DIR,f'shap_{mname}model.png')] if os.path.exists(p)]
        if not figs: st.info(f'no figures for {label} in {FIG_DIR}')
        pick=st.multiselect('figures',[os.path.basename(p) for p in figs],default=[os.path.basename(p) for p in figs])
        for p in figs:
            if os.path.basename(p) in pick: img(p,os.path.basename(p))

    with tabs[5]:
        h=next((x for x in training_runs(MOD_DIR) if x['stem']==info.get('training_stem')),None)
        if h is None:
            st.info('no training run linked to this folder (run_info.json missing or its hyperparams file not found)')
        else:
            st.markdown(f"**Training run `{h['stem']}`** | train seasons {h.get('train_seasons')} | tuning validation seasons {h.get('valid_seasons')} | "
                        f"chosen trial {h.get('chosen_trial')} of {h.get('pareto_size')} on the Pareto front")
            table(train_vs_indep(h,vm).round(3),hide_index=True)
            st.caption('tuning valid: held-out seasons at the training locations, also used for tuning and early stopping, so optimistic. '
                       'independent: validation locations never used in training.')

    with tabs[6]:
        sv=read_csv(os.path.join(VAL_DIR,f'shap_importance_{mname}model.csv'))
        stn=info.get('training_stem')
        stv=read_csv(os.path.join(RES_DIR,f'shap_importance_{stn}.csv')) if stn else None
        if sv is None:
            st.info(f'no shap_importance_{mname}model.csv in this folder')
        else:
            parts=[sv.assign(**{'share %':100*sv['mean_abs_shap']/sv['mean_abs_shap'].sum(),'data':'validation locations'})]
            if stv is not None:
                parts.append(stv.assign(**{'share %':100*stv['mean_abs_shap']/stv['mean_abs_shap'].sum(),'data':'training run (tuning validation seasons)'}))
            imp=pd.concat(parts)
            order=sv.sort_values('mean_abs_shap')['feature'].tolist()
            fig=px.bar(imp,x='share %',y='feature',color='data',barmode='group',orientation='h',height=28*len(order)+140,
                       category_orders={'feature':order[::-1]},title='mean |SHAP| share per feature (log-SWE margin)')
            fig.update_layout(yaxis_title=None,margin=dict(l=0,r=10,t=40,b=0),legend=dict(orientation='h',y=1.04))
            chart(fig)
            st.caption('A feature that matters much more on the validation locations than in training points to conditions the training data covered poorly.')
        img(os.path.join(FIG_DIR,f'shap_{mname}model.png'),'SHAP beeswarm on validation locations')

    with tabs[7]:
        # maps of the independent validation locations (same for every validation run)
        areas=[a for a in VAL_LOC_FIGS if os.path.exists(os.path.join(base,VAL_LOC_FIGS[a]))]
        if not areas:
            st.info(f"no validation location maps ({', '.join(VAL_LOC_FIGS.values())}) in {base}")
        else:
            opts=areas+(['both side by side'] if len(areas)>1 else [])
            # start with the domain picked in 'evaluated on' above, both for 'all'
            a=st.radio('area',opts,index=opts.index(dom) if dom in opts else len(opts)-1,horizontal=True,key=f'valloc_area_{dom}')
            st.caption('Locations used for the independent validation, never used in training. '
                       'The terrain of the training locations is on the Training run page, tab Training locations.')
            show=areas if a=='both side by side' else [a]
            for col,x in zip(st.columns(len(show)),show):
                with col: img(os.path.join(base,VAL_LOC_FIGS[x]),f'{x}: validation locations')

# ---------------------------------------------------------------
# Page: validation explorer (every validation row: observed, XGBoost and baselines)
# ---------------------------------------------------------------
def set_loc(lt,lat,lon): st.session_state['loc']=nearest(lt,lat,lon)

def page_predictions():
    tag=pick_val_model()
    if tag is None: return
    files=sorted(glob.glob(os.path.join(VAL_DIR,'SWE_xgboost_validation_*model.csv.gz')))
    if not files:
        st.warning(f'no SWE_xgboost_validation_*model.csv.gz in {VAL_DIR}. Set save_predictions=True in xgb-validate-cryo-swe.py to write it.'); return
    st.title(f'Validation explorer: {edition_name(tag)}')
    f=files[0] if len(files)==1 else st.selectbox('prediction file',files,format_func=os.path.basename)
    mt=mtime(f)
    df=load_pred(f,mt); lt=loc_table(f,mt)
    preds=[c for c in PRED_STYLE if c in df]
    st.caption(f"{os.path.basename(f)} | {len(df):,} rows | {len(lt):,} locations | {df['utctime'].min():%Y-%m-%d} to {df['utctime'].max():%Y-%m-%d} | written {stamp(f)}")

    tabs=st.tabs(['Location time series','Daily map','Domain mean'])

    with tabs[0]:
        c1,c2=st.columns([1,3])
        tags=c1.multiselect('domains',sorted(lt['tag'].unique()),default=sorted(lt['tag'].unique()))
        colby=c1.selectbox('color map by',[c for c in ['RMSE','ME','obs_mean','obs_max','pred_mean','height','dist_to_train_m'] if c in lt])
        worst=lt.nlargest(50,'RMSE')
        c1.selectbox('jump to a worst RMSE location',[None]+list(zip(worst['latitude'],worst['longitude'])),key='worst_pick',
                     format_func=lambda x:'' if x is None else f'{x[0]:.3f}, {x[1]:.3f}',
                     on_change=lambda: st.session_state['worst_pick'] and set_loc(lt,*st.session_state['worst_pick']))
        la=c1.number_input('latitude',value=float(lt['latitude'].median()),format='%.4f')
        lo=c1.number_input('longitude',value=float(lt['longitude'].median()),format='%.4f')
        c1.button('go to nearest location',on_click=set_loc,args=(lt,la,lo))
        sub=lt[lt['tag'].isin(tags)]
        if 'loc' not in st.session_state:
            r=lt.loc[lt['obs_mean'].idxmax()]; st.session_state['loc']=(float(r['latitude']),float(r['longitude']))
        with c2:
            if sub.empty: st.info('no locations for the chosen domains')
            else:
                fig=map_fig(sub,colby,hover={c:':.2f' for c in ['RMSE','ME','obs_mean','height'] if c in sub},height=520)
                lat0,lon0=st.session_state['loc']
                fig.add_trace(go.Scattermap(lat=[lat0],lon=[lon0],mode='markers',marker=dict(size=16,color='red'),name='selected',
                                            customdata=[[lat0,lon0]],showlegend=False)
                              if hasattr(go,'Scattermap') else go.Scattermapbox(lat=[lat0],lon=[lon0],mode='markers',marker=dict(size=16,color='red'),
                                                                                 customdata=[[lat0,lon0]],showlegend=False))
                try:
                    ev=chart(fig,on_select='rerun',selection_mode='points',key='locmap')
                    pts=ev.selection.points if ev is not None else []
                except Exception:
                    st.caption('click selection needs streamlit ≥ 1.35, use the inputs on the left'); pts=[]
                ll=next((x for x in (click_latlon(p) for p in pts) if x is not None),None) if pts else None
                if ll is not None and ll!=st.session_state.get('last_click'):
                    st.session_state['last_click']=ll; set_loc(lt,*ll); st.rerun()
                st.caption('click a point on the map to show its time series')

        lat,lon=st.session_state['loc']
        idx=loc_index(f,mt).get((lat,lon))
        if idx is None: st.info('selected location not in this file, pick another'); return
        d=df.iloc[idx].sort_values('utctime')
        info=lt[(lt['latitude']==lat)&(lt['longitude']==lon)].iloc[0]
        seasons=sorted(d['season'].unique())
        sel=st.select_slider('snow year',options=['all']+[f'{s}/{(s+1)%100:02d}' for s in seasons],value='all')
        if sel!='all': d=d[d['season']==int(sel[:4])]
        st.subheader(f"{info['tag']} {lat:.4f}N {lon:.4f}E, {info['height']:.0f} m"
                     + (f", {info['dist_to_train_m']/1000:.1f} km from nearest training point" if 'dist_to_train_m' in info and pd.notna(info['dist_to_train_m']) else ''))
        fig=go.Figure()
        for c in preds:
            name,color,dash=PRED_STYLE[c]
            fig.add_scatter(x=d['utctime'],y=d[c],mode='lines',name=name,line=dict(color=color,dash=dash,width=2.2 if c in ('SWE_observed','SWE_xgboost') else 1.4))
        fig.update_layout(height=420,yaxis_title='SWE (mm)',hovermode='x unified',legend=dict(orientation='h',y=1.1),margin=dict(t=30))
        chart(fig)
        sc=pd.DataFrame({PRED_STYLE[c][0]:quick_scores(d['SWE_observed'],d[c]) for c in preds if c!='SWE_observed'}).T
        table(sc.round(2))

    with tabs[1]:
        dmin,dmax=df['utctime'].min().date(),df['utctime'].max().date()
        last=pd.Timestamp(f"{int(df['season'].max())+1}-03-01").date()
        c1,c2=st.columns([1,3])
        day=c1.date_input('day',value=min(max(last,dmin),dmax),min_value=dmin,max_value=dmax)
        comp=c1.selectbox('compare with',[c for c in preds if c!='SWE_observed'],format_func=lambda c:PRED_STYLE[c][0])
        t0=pd.Timestamp(day); m=(df['utctime']>=t0)&(df['utctime']<t0+pd.Timedelta(days=1))
        dd=df.loc[m,['latitude','longitude','tag','height']+preds].copy()
        if dd.empty: st.info('no data for this day'); return
        dd[f'{comp} minus observed']=dd[comp]-dd['SWE_observed']
        s=quick_scores(dd['SWE_observed'],dd[comp])
        for k,v in [('locations',f"{len(dd):,}"),('RMSE (mm)',fmt(s.get('RMSE'),'{:.1f}')),('ME (mm)',fmt(s.get('ME'),'{:+.1f}')),('KGE',fmt(s.get('KGE'),'{:.3f}'))]:
            c1.metric(k,v)
        hi=float(np.nanpercentile(np.r_[dd['SWE_observed'],dd[comp]],99)) or 1.0
        with c2:
            a,b=st.columns(2)
            with a: chart(map_fig(dd,'SWE_observed',hover={'height':':.0f'},height=430,title='observed SWE (mm)',rng=(0,hi),cscale='Blues'))
            with b: chart(map_fig(dd,comp,hover={'height':':.0f'},height=430,title=f'{PRED_STYLE[comp][0]} SWE (mm)',rng=(0,hi),cscale='Blues'))
            chart(map_fig(dd,f'{comp} minus observed',hover={'SWE_observed':':.1f',comp:':.1f','height':':.0f'},height=480,
                          title=f'{PRED_STYLE[comp][0]} minus observed (mm)'))

    with tabs[2]:
        dm=domain_mean(f,mt)
        long=dm.melt(id_vars=['date','tag'],var_name='predictor',value_name='SWE')
        long['predictor']=long['predictor'].map(lambda c:PRED_STYLE[c][0])
        fig=px.line(long,x='date',y='SWE',color='predictor',facet_row='tag',height=330*dm['tag'].nunique(),
                    color_discrete_map={v[0]:v[1] for v in PRED_STYLE.values()},labels={'SWE':'domain mean SWE (mm)'})
        fig.for_each_annotation(lambda a:a.update(text=a.text.split('=')[-1]))
        fig.update_yaxes(matches=None); fig.update_layout(hovermode='x unified')
        chart(fig)

# ---------------------------------------------------------------
# Page: SWE predictions (ECXENS forecasts validated against HKL, figures and scores from the forecast validation script)
# ---------------------------------------------------------------
def page_forecasts():
    st.title('SWE predictions')
    known=tuple(sorted(set(EDITION_NAMES)|{h['optuna_tag'] for h in training_runs(MOD_DIR)}))
    runs=fc_runs(PRED_DIR,known)
    if not runs:
        st.warning(f'no forecast validation results in {PRED_DIR}')
        st.markdown('Copy the output of the forecast validation script here as `<model>/<start date>/<prediction>/`, '
                    'for example `FINmulti/20230210/wettest/` with `swe_metrics.csv` and the png figures in each prediction folder:')
        st.code(f'rsync -av smartmet@smart.nsdc.fmi.fi:/home/users/smartmet/data/ec-ens/validation/figs_swe_validation/ {PRED_DIR}/',language='bash')
        return

    # ---- model edition, then start date (forecast base time) of that model; both kept when switching pages
    models=sorted({r['model'] for r in runs},key=lambda m:(m!=FC_DEFAULT_MODEL,m))
    nstart={m:sum(r['model']==m for r in runs) for m in models}
    # widgets with fixed keys: their value lives in session_state, so a pick is never reset by a changing default
    if st.session_state.get('fc_model_sel') not in models:
        ed=st.session_state.get('edition_keep')   # default: edition picked on the Training run page
        st.session_state['fc_model_sel']=ed if ed in models else models[0]
    c1,c2=st.columns([2,3])
    with c1:
        model=st.radio('model',models,horizontal=True,key='fc_model_sel',
                       format_func=lambda m:f'{edition_name(m)} ({nstart[m]} start date{"s" if nstart[m]!=1 else ""})')
    runs=[r for r in runs if r['model']==model]
    paths=[r['run'] for r in runs]
    skey=f'fc_run_sel_{model}'   # one start date choice per model
    if st.session_state.get(skey) not in paths: st.session_state[skey]=paths[0]
    with c2:
        run_path=st.selectbox('start date',paths,key=skey,
                              format_func=lambda p:fc_run_label(next(r for r in runs if r['run']==p)))
    R=next(r for r in runs if r['run']==run_path)
    preds,base_time=R['preds'],R['base_time']
    df=fc_metrics(preds)
    if df is None:
        st.warning('no readable swe_metrics.csv in this run'); return
    names=[fc_name(p) for p in preds]
    # the folder's start date and the dates inside swe_metrics.csv must agree, otherwise the folder holds another run's results
    bt_scores=fc_base_from_scores(preds[0])
    if base_time is not None and bt_scores is not None and abs((bt_scores-base_time).days)>0:
        st.warning(f"`{fc_rel(run_path)}` says start date {base_time:%Y-%m-%d}, but its swe_metrics.csv has dates "
                   f"{df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d} (start {bt_scores:%Y-%m-%d}): "
                   "these files come from another forecast run, check what was copied into this folder")
    st.caption(f"model {edition_name(model)} | start {fc_run_label(R).split('  (')[0]} | {', '.join(names)} | {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d} | reference HKL SWE | "
               f"bias is ECXENS minus HKL | written {stamp(os.path.join(preds[0],'swe_metrics.csv'))}")

    # ---- daily maps first: per prediction raw ECENS, ECXENS, HKL (one figure) and ECXENS minus HKL
    st.subheader('Daily maps')
    dates=sorted({os.path.basename(f)[len('swe_compare_'):-4] for p in preds for f in glob.glob(os.path.join(p,'swe_compare_*.png'))})
    if not dates:
        st.info('no swe_compare_YYYYMMDD.png figures in this run')
    else:
        lab=lambda t: f"{pd.Timestamp(t):%Y-%m-%d}"+(f" (lead {(pd.Timestamp(t)-base_time).days} d)" if base_time is not None else '')
        c1,c2=st.columns([3,2])
        if len(dates)>1:
            day=c1.select_slider('day',options=dates,value=dates[0],format_func=lab,key=f'fc_day_{run_path}')
        else:
            day=dates[0]; c1.markdown(f'**day** {lab(day)}')
        show=c2.multiselect('predictions',preds,default=preds,format_func=fc_name,key=f'fc_show_{run_path}')
        st.caption(f'Each row: raw ECENS (as in the file, 0.1° grid), ECXENS and HKL SWE with the same legend; ECXENS minus HKL (500 m); '
                   f'ECENS minus HKL (0.1°, HKL averaged into the ECENS cells); |ECXENS bias| minus |ECENS bias| (0.1°, greens and blues = ECXENS closer '
                   f'to HKL, yellows and reds = further). All bias maps: soft greens and yellows within ±{MILD_BIAS:g} mm, strong blues and reds beyond.')
        ts=pd.Timestamp(day)
        for p in show:
            st.markdown(f'#### {fc_name(p)}')
            r=df[(df['pred']==os.path.basename(p))&(df['date']==ts)]
            if not r.empty:
                r=r.iloc[0]
                parts=[('bias',r.get('bias'),'{:+.1f} mm'),('RMSE',r.get('rmse'),'{:.1f} mm'),('KGE',r.get('kge'),'{:.3f}'),
                       ('CSI 1 mm',r.get('csi_1'),'{:.3f}'),('POD 1 mm',r.get('pod_1'),'{:.3f}'),('FAR 1 mm',r.get('far_1'),'{:.3f}')]
                if pd.notna(r.get('vol_ecx_km3')) and pd.notna(r.get('vol_hkl_km3')) and r.get('vol_hkl_km3'):
                    parts.append(('SWE volume ECXENS / HKL',r['vol_ecx_km3']/r['vol_hkl_km3'],'{:.2f}'))
                if pd.notna(r.get('skill_mse_vs_ecens')):
                    parts+=[('0.1° RMSE ECENS',r.get('ecens_rmse'),'{:.1f} mm'),('ECXENS',r.get('ecx01_rmse'),'{:.1f} mm'),
                            ('MSE skill vs ECENS',r.get('skill_mse_vs_ecens'),'{:+.2f}'),('cells improved',r.get('frac_cells_improved'),'{:.0%}')]
                st.markdown(' | '.join(f'{k} **{fmt(v,f)}**' for k,v,f in parts))
            # widths follow the png aspect ratios, so all images in the row get the same height
            row=[os.path.join(p,f'swe_compare_{day}.png'),os.path.join(p,f'swe_bias_{day}.png')]
            # ECENS bias and bias size change; runs from before that have the ECXENS minus ECENS map or nothing
            extra=[os.path.join(p,f'{n}_{day}.png') for n in ('swe_ecens_bias','swe_bias_gain')]
            extra=[x for x in extra if os.path.exists(x)] or [x for x in [os.path.join(p,f'swe_ecx_minus_ecens_{day}.png')] if os.path.exists(x)]
            row+=extra
            for col,fp in zip(st.columns([img_ratio(fp,d) for fp,d in zip(row,[17.7/8]+[7/8]*(len(row)-1))]),row):
                with col: img(fp)

    # ---- everything else below the maps
    st.divider()
    st.subheader('Scores and summaries')
    tabs=st.tabs(['Scores vs lead day','Period summary','Ensemble members','All figures','Scores table'])
    order={'name':names}

    with tabs[0]:
        c1,c2=st.columns([3,1])
        mets=c1.multiselect('scores',list(FC_METRICS),default=['bias','rmse','kge'],format_func=lambda m:FC_METRICS[m][0])
        snow=c2.radio('pixels',['all','snow'],horizontal=True,format_func=lambda s:'all' if s=='all' else 'snow (≥ 1 mm)')
        cols=st.columns(2)
        for i,m in enumerate(mets):
            k=m if snow=='all' else f'snow_{m}'
            if k not in df: continue
            f=px.line(df,x='lead_day',y=k,color='name',markers=True,height=340,title=FC_METRICS[m][0],category_orders=order,
                      labels={'lead_day':'lead day','name':'prediction',k:FC_METRICS[m][1] or m})
            if m=='bias': f.add_hline(y=0,line_color='black',line_width=0.6)
            with cols[i%2]: chart(f,key=f'fc_met_{m}')
        thrs=sorted({int(c.split('_')[1]) for c in df.columns if c.startswith('csi_')})
        if thrs:
            st.markdown('**Snow detection** (forecast and HKL above the threshold)')
            thr=st.radio('threshold (mm)',thrs,horizontal=True)
            cat=df.melt(id_vars=['lead_day','name'],value_vars=[f'{s}_{thr}' for s in ['pod','far','csi','hss'] if f'{s}_{thr}' in df],
                        var_name='score',value_name='value')
            cat['score']=cat['score'].str.split('_').str[0].str.upper()
            f=px.line(cat,x='lead_day',y='value',color='name',facet_col='score',markers=True,height=330,category_orders=order,
                      labels={'lead_day':'lead day','name':'prediction','value':''})
            f.for_each_annotation(lambda a:a.update(text=a.text.split('=')[-1]))
            f.update_yaxes(range=[min(0,float(cat['value'].min()) if cat['value'].notna().any() else 0),1])   # HSS can be negative
            chart(f,key='fc_cat')
        if {'vol_ecx_km3','vol_hkl_km3'}<=set(df.columns):
            vol=pd.concat([df[['lead_day','name','vol_ecx_km3']].rename(columns={'vol_ecx_km3':'km3'}),
                           df.drop_duplicates('lead_day')[['lead_day','vol_hkl_km3']].rename(columns={'vol_hkl_km3':'km3'}).assign(name='HKL')])
            f=px.line(vol,x='lead_day',y='km3',color='name',markers=True,height=340,title='total SWE volume (common valid pixels)',
                      category_orders={'name':names+['HKL']},labels={'lead_day':'lead day','name':'','km3':'km³ water'})
            chart(f,key='fc_vol')
        if 'skill_mse_vs_ecens' in df and df['skill_mse_vs_ecens'].notna().any():
            st.markdown('**ECXENS vs raw ECENS, both against HKL on the 0.1° ECENS grid** (same cells for both)')
            sc=st.radio('score',['rmse','bias','mae','kge'],horizontal=True,format_func=lambda m:FC_METRICS[m][0],key='fc_ecens_score')
            if f'ecens_{sc}' in df and f'ecx01_{sc}' in df:
                both=pd.concat([df[['lead_day','name',f'ecens_{sc}']].rename(columns={f'ecens_{sc}':'v'}).assign(source='ECENS raw'),
                                df[['lead_day','name',f'ecx01_{sc}']].rename(columns={f'ecx01_{sc}':'v'}).assign(source='ECXENS')])
                c1,c2=st.columns(2)
                f=px.line(both,x='lead_day',y='v',color='name',line_dash='source',markers=True,height=360,category_orders=order,
                          title=f'{FC_METRICS[sc][0]}, 0.1° (solid ECENS raw, dashed ECXENS)',
                          labels={'lead_day':'lead day','name':'prediction','v':FC_METRICS[sc][1] or sc,'source':''})
                if sc=='bias': f.add_hline(y=0,line_color='black',line_width=0.6)
                with c1: chart(f,key='fc_ecens_both')
                sk=df.melt(id_vars=['lead_day','name'],value_vars=[c for c in ['skill_mse_vs_ecens','frac_cells_improved'] if c in df],
                           var_name='score',value_name='value')
                sk['score']=sk['score'].map({'skill_mse_vs_ecens':'MSE skill vs ECENS','frac_cells_improved':'fraction of cells improved'})
                f=px.line(sk,x='lead_day',y='value',color='name',facet_col='score',markers=True,height=360,category_orders=order,
                          labels={'lead_day':'lead day','name':'prediction','value':''})
                f.for_each_annotation(lambda a:a.update(text=a.text.split('=')[-1]))
                f.add_hline(y=0,line_color='black',line_width=0.6,col=1); f.add_hline(y=0.5,line_dash='dot',line_color='gray',col=2)
                with c2: chart(f,key='fc_ecens_skill')
                st.caption('MSE skill above 0 and more than half of the cells improved mean the downscaling brings the forecast closer to HKL.')

    with tabs[1]:
        summ=[]
        for p in preds:
            d=df[df['pred']==os.path.basename(p)]
            row={'prediction':fc_name(p),'days':len(d)}
            for k,l in [('bias','bias (mm)'),('mae','MAE (mm)'),('rmse','RMSE (mm)'),('r','r'),('kge','KGE'),
                        ('snow_bias','bias snow px (mm)'),('snow_rmse','RMSE snow px (mm)'),('snow_kge','KGE snow px'),
                        ('csi_1','CSI 1 mm'),('pod_1','POD 1 mm'),('far_1','FAR 1 mm')]:
                if k in d: row[l]=d[k].mean()
            if {'vol_ecx_km3','vol_hkl_km3'}<=set(d.columns): row['SWE volume ECXENS / HKL']=(d['vol_ecx_km3']/d['vol_hkl_km3']).mean()
            for k,l in [('ecens_bias','0.1° bias ECENS (mm)'),('ecx01_bias','0.1° bias ECXENS (mm)'),('ecens_rmse','0.1° RMSE ECENS (mm)'),
                        ('ecx01_rmse','0.1° RMSE ECXENS (mm)'),('skill_mse_vs_ecens','MSE skill vs ECENS'),('frac_cells_improved','cells improved')]:
                if k in d: row[l]=d[k].mean()
            summ.append(row)
        st.markdown('**Means over all lead days**')
        table(pd.DataFrame(summ).round(3),hide_index=True)
        figs={'swe_bias_mean_rmse.png':'mean bias and RMSE maps','swe_ecens_bias_gain_mean.png':'ECENS bias and MAE change vs ECENS',
              'swe_density_scatter.png':'pixel density, all days pooled','swe_scores_vs_lead.png':'scores per lead day'}
        kind=st.radio('figure',list(figs),horizontal=True,format_func=figs.get,key='fc_period_fig')
        cols=st.columns(2)
        for i,p in enumerate(preds):
            with cols[i%2]:
                st.markdown(f'**{fc_name(p)}**')
                img(os.path.join(p,kind))

    with tabs[2]:
        rk=read_csv(os.path.join(run_path,'member_ranking.csv'))
        if rk is None:
            st.info('no member_ranking.csv in this run (written by the validation script when ADD_EXTREME_MEMBERS = True)')
        else:
            stats=[c for c in MEMBER_STATS if c in rk]
            rank_col=next((c for c in rk.columns if c.startswith('rank_')),None)
            rank_by=rank_col[len('rank_'):] if rank_col and rank_col[len('rank_'):] in rk else (stats[0] if stats else None)
            if rank_by is None:
                st.info('member_ranking.csv has no snow amount columns')
            else:
                srt=rk.sort_values(rank_by)
                dri,wet=int(srt['member'].iloc[0]),int(srt['member'].iloc[-1])
                c1,c2=st.columns([1,4])
                mcol=c1.radio('show',stats,index=stats.index(rank_by),format_func=MEMBER_STATS.get,key='fc_member_stat')
                c1.metric('driest member',dri); c1.metric('wettest member',wet)
                c1.caption(f'ranked by {MEMBER_STATS.get(rank_by,rank_by)}, {len(rk)} members')
                missing=sorted(set(range(1,int(rk['member'].max())+1))-set(rk['member'].astype(int)))
                if missing: c1.caption('missing members: '+', '.join(map(str,missing)))
                d=rk.sort_values(mcol).copy()
                d['member_s']=d['member'].astype(int).astype(str)
                d['role']=np.select([d['member']==dri,d['member']==wet,d['member']==1],['driest','wettest','member 1'],'other')
                f=px.bar(d,x='member_s',y=mcol,color='role',height=420,category_orders={'member_s':d['member_s'].tolist()},
                         color_discrete_map={'driest':'#f5b800','wettest':'#1747c9','member 1':'#3fae49','other':'#b8b8b8'},
                         labels={'member_s':'member (sorted)',mcol:f'{MEMBER_STATS.get(mcol,mcol)} (mm)','role':''},
                         title=f'{MEMBER_STATS.get(mcol,mcol)} per ECXENS member')
                f.add_hline(y=float(rk[mcol].mean()),line_dash='dot',line_color='black',annotation_text='mean of members')
                with c2: chart(f,key='fc_members')
                table(rk.round(3),hide_index=True)

    with tabs[3]:
        p=st.selectbox('prediction',preds,format_func=fc_name,key='fc_allfig_pred')
        figs=sorted(glob.glob(os.path.join(p,'*.png')))
        if not figs:
            st.info(f'no figures in {p}')
        else:
            fnames=[os.path.basename(x) for x in figs]
            pick=st.multiselect(f'{len(figs)} figures in {p}',fnames,default=[n for n in fnames if not n.startswith(('swe_compare_','swe_bias_2'))],key='fc_allfig_pick')
            ncol=st.radio('columns',[1,2,3],index=1,horizontal=True,key='fc_allfig_cols')
            gc=st.columns(ncol)
            for i,x in enumerate([x for x in figs if os.path.basename(x) in pick]):
                with gc[i%ncol]: img(x,os.path.basename(x))

    with tabs[4]:
        show=df.drop(columns=['pred']).rename(columns={'name':'prediction'})
        show['date']=show['date'].dt.date
        table(show.round(4),hide_index=True)
        st.download_button('download scores (csv)',show.to_csv(index=False),file_name=f'swe_metrics_{fc_rel(run_path).replace(os.sep,"_")}.csv',mime='text/csv')

# ---------------------------------------------------------------
{'Overview':page_overview,'Training run':page_training,'Validation':page_validation,'Validation explorer':page_predictions,
 'SWE predictions':page_forecasts}[page]()

# footer on every page: CryoSCOPE contact and funding statement (EU, SERI, MoES)
if logo('cryoscope.png'):
    st.divider()
    _ftr=st.columns([3,2])
    _ftr[1].image(logo('cryoscope.png'),width=640)