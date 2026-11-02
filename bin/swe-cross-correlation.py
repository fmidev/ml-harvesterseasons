import os,sys,time,warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless server
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
# Cross-correlation of the SWE training data for Scandinavia / Alps / both, for ONE snow season
# Snow season 1 Sep to 31 Aug, label = start year: 2018 -> 2018-09-01..2019-08-31 (winter 2018/19), same as the model script
# Columns: cols_used minus utctime and tag, plus doy_cos / doy_sin (appended at the end). Target SWE.
# VARIABLE ORDER: every output (matrices, heatmaps, SWE tables, bar charts, pair lists) uses the order of cols_used,
#                 identical for Scandinavia / Alps / both. Columns that are constant or empty in a domain are kept
#                 as NaN (grey in heatmaps) so all domains stay aligned.
# Pearson: exact pairwise-complete (NaN handled per pair), all rows of the season, computed in row blocks
#          over the in-memory df (no per-domain copies of the data).
# Pearson snow-on: same, SWE>0 rows only, since the many zeros dominate the all-rows r.
# Spearman: ranks on a random row sample (SPEARMAN_SAMPLE), then the same pairwise Pearson on ranks
#           (with NaNs a close approximation of pairwise-complete Spearman).
# usage: python swe_correlation.py <run_tag> [season]     e.g. python swe_correlation.py corr1 2018
#        season 'all' uses every season

startTime=time.time()
run_tag=sys.argv[1] if len(sys.argv)>1 else 'corr1'
SEASON=sys.argv[2] if len(sys.argv)>2 else '2018'     # snow season start year, or 'all'
SEASON=None if SEASON=='all' else int(SEASON)
SNOW_START_MONTH=9
season_lbl='allseasons' if SEASON is None else f'season{SEASON}-{(SEASON+1)%100:02d}'

data_dir='/home/ubuntu/data/cryoscope/snowmodel/' # training data
res_dir=f'{data_dir}results/correlation'
os.makedirs(res_dir,exist_ok=True)

TARGET='SWE'
HIGH_R=0.9                      # |r| threshold for the highly correlated feature pairs table
SPEARMAN_SAMPLE=5_000_000       # rows per domain for Spearman (None = all rows, needs a lot of RAM)
BLOCK=1_000_000                 # rows per block in the Pearson accumulation (in memory, not file reading)
SEED=99
domains=['Scandinavia','Alps','both']

# Read in training data; the ORDER of this list is the order used in all outputs
cols_used=['utctime','tag',
            'SWE',
            'swe_mean','swe_max','swe_min',
            'snowdepth_mean','snowdepth_max','snowdepth_min',
            'skt_00',#'skt_12',
            'stl1_00',#'stl1_12',
            't2_00',#'t2_12',
            'td2_00',#'td2_12',
            'tsn_00',#'tsn_12',
            'lict_00',#'lict_12',
            'tmax_max','tmin_min',
            'rsn_00',#'rsn_12',#'sd_00','sd_12',
            'sde_00',#'sde_12',
            'asn_00',#'asn_12',#'fal_00','fal_12',
            #'snowc_00','snowc_12',
            'latitude','longitude','height','slope','aspect','tcf',#'fch_2021',
            'e','tp','pev','ro','sf','slhf','smlt','sro','sshf','ssr','ssrd','ssro','str','strd',#'es',
            'laihv_00',#'laihv_12',
            'lailv_00',#'lailv_12',
            #'licd_00','licd_12',
            'sp_00',#'sp_12',
            'src_00',#'src_12',
            'swvl1_00',#'swvl1_12',
            'swvl2_00',#'swvl2_12',#'swvl3_00',#'swvl3_12',#'swvl4_00',#'swvl4_12',
            'u10_00',#'u10_12',
            'v10_00',#'v10_12',
            'q500_00','t500_00','u500_00','v500_00','z500_00',
            #'q500_12','t500_12','u500_12','v500_12','z500_12',
            'q700_00','u700_00','t700_00','v700_00','z700_00',
            #'q700_12','u700_12','t700_12','v700_12','z700_12',
            'q850_00','u850_00','t850_00','v850_00','z850_00',
            #'q850_12','u850_12','t850_12','v850_12','z850_12',
            #'sktmax_max','sktmin_min',
           ]
dups=[c for c in set(cols_used) if cols_used.count(c)>1]
if dups: sys.exit(f'duplicate names in cols_used: {dups}')
input=f'{data_dir}training_all_scandi_alps_ext.csv.gz'
# testing with:
#input=f'{data_dir}sample10_scandi_alps.csv.gz'
#input=f'{data_dir}sample1000_scandi_alps.csv.gz'

# fail fast (before the long read) if a name in cols_used is not in the file
header=pd.read_csv(input,nrows=0).columns
missing=[c for c in cols_used if c not in header]
if missing: sys.exit(f'columns in cols_used not found in {input}: {missing}')

# numeric columns read straight as float32 (lower peak RAM than reading float64 and casting afterwards)
dtypes={c:np.float32 for c in cols_used if c not in ['utctime','tag']}; dtypes['tag']=str
df=pd.read_csv(input,usecols=cols_used,dtype=dtypes)   # ~45 minutes for the full training data
print('read done, minutes: %.2f'%((time.time()-startTime)/60))

# snow season: Sep-Dec belong to that year's season, Jan-Aug to the previous year's season
ut=pd.to_datetime(df['utctime'])
yr=ut.dt.year.to_numpy(); mo=ut.dt.month.to_numpy()
season=np.where(mo>=SNOW_START_MONTH,yr,yr-1)
print('seasons present:',sorted(np.unique(season).tolist()))
if SEASON is not None:
    keep_rows=season==SEASON
    if not keep_rows.any(): sys.exit(f'no rows for season {SEASON}')
    df=df[keep_rows].reset_index(drop=True)   # original frame freed when rebound
    ut=ut[keep_rows].reset_index(drop=True)
    print(f'kept season {SEASON} ({season_lbl}): {len(df)} rows,',ut.min(),'..',ut.max())
del yr,mo,season

# change negative SWE values to 0.0 (same as the model script)
df[TARGET]=df[TARGET].clip(lower=0.0)

# add day of year cosine and sine as columns, then drop utctime (not needed further)
doy=ut.dt.dayofyear.to_numpy()
df['doy_cos']=np.cos(2*np.pi*doy/365.25).astype(np.float32)
df['doy_sin']=np.sin(2*np.pi*doy/365.25).astype(np.float32)
del doy,ut
df.drop(columns=['utctime'],inplace=True)
print(df)
print('tags present:',df['tag'].unique())

# correlation columns in cols_used order (read_csv returns file order, so never use df.columns for ordering)
cols=[TARGET]+[c for c in cols_used if c not in ['utctime','tag',TARGET]]+['doy_cos','doy_sin']
cpos=[df.columns.get_loc(c) for c in cols]
feat=[c for c in cols if c!=TARGET]
k=len(cols)
print(k,'columns in correlation (incl. SWE), fixed order:',cols)


def pairwise_pearson(frame,cpos,cols,rows=None,block=BLOCK):
    """Exact pairwise-complete Pearson matrix via blockwise matrix products.
    frame: DataFrame, cpos: column positions, rows: row positions (None = all rows).
    Returns (r DataFrame, n_pairs DataFrame, n_rows). Constant or empty columns come out as NaN."""
    kk=len(cpos); n=len(frame) if rows is None else len(rows)
    S_n=np.zeros((kk,kk)); S_x=np.zeros((kk,kk)); S_xx=np.zeros((kk,kk)); S_xy=np.zeros((kk,kk))
    shift=None
    for s in range(0,n,block):
        e=min(s+block,n)
        blk=frame.iloc[s:e,cpos] if rows is None else frame.iloc[rows[s:e],cpos]
        B=blk.to_numpy(dtype=np.float64); del blk
        if shift is None:   # first-block mean shift for numerical stability (r is shift invariant)
            shift=np.nan_to_num(np.nanmean(B,axis=0))
        B-=shift
        M=~np.isnan(B); B[~M]=0.0; Mf=M.astype(np.float64); del M
        S_n+=Mf.T@Mf          # rows where both i and j valid
        S_x+=B.T@Mf           # [i,j]: sum of x_i over rows where i and j valid
        S_xx+=(B*B).T@Mf      # [i,j]: sum of x_i^2 over rows where i and j valid
        S_xy+=B.T@B           # [i,j]: sum of x_i*x_j over rows where both valid
        del B,Mf
    with np.errstate(divide='ignore',invalid='ignore'):
        cov=S_xy-S_x*S_x.T/S_n
        var_i=S_xx-S_x**2/S_n
        r=cov/np.sqrt(var_i*var_i.T)
    r[S_n<3]=np.nan
    r=np.clip(r,-1.0,1.0)
    bad=~(np.diag(var_i)>0)
    r[bad,:]=np.nan; r[:,bad]=np.nan
    r[np.diag_indices(kk)]=np.where(bad,np.nan,1.0)
    return pd.DataFrame(r,index=cols,columns=cols),pd.DataFrame(S_n,index=cols,columns=cols),n


def plot_heatmap(rdf,title,fname):
    """Heatmap in the given (fixed) order; NaN cells shown grey."""
    kk=len(rdf); sz=max(8,0.18*kk+3); fs=max(4,min(9,500/kk))
    cmap=plt.get_cmap('RdBu_r').copy(); cmap.set_bad('lightgrey')
    fig,ax=plt.subplots(figsize=(sz,sz*0.9))
    im=ax.imshow(np.ma.masked_invalid(rdf.to_numpy()),cmap=cmap,vmin=-1,vmax=1,interpolation='nearest')
    ax.set_xticks(range(kk)); ax.set_xticklabels(rdf.columns,rotation=90,fontsize=fs)
    ax.set_yticks(range(kk)); ax.set_yticklabels(rdf.index,fontsize=fs)
    fig.colorbar(im,ax=ax,fraction=0.03,pad=0.01,label='r')
    ax.set_title(title); plt.tight_layout()
    plt.savefig(fname+'.png',dpi=150); plt.savefig(fname+'.svg'); plt.close()


rng=np.random.RandomState(SEED)
swe_all_domains={}
swe=df[TARGET].to_numpy()
tag=df['tag'].to_numpy()

# ======================================================================
# loop over the three domains: Scandinavia-only, Alps-only, both pooled
# ======================================================================
for domain in domains:
    print('\n'+'='*70)
    print('DOMAIN:',domain,'|',season_lbl)
    domStart=time.time()

    # row positions only, no copy of the data
    if domain=='both':
        rows=None; pool=np.arange(len(df))
        rows_on=np.flatnonzero(swe>0)
    else:
        rows=np.flatnonzero(tag==domain); pool=rows
        rows_on=rows[swe[rows]>0]
    n_dom=len(pool)
    if n_dom==0:
        print('  no rows for',domain,'(check the tag values printed above), skipping')
        continue
    print(f'  rows {n_dom} | snow-on rows {len(rows_on)} ({len(rows_on)/n_dom:.3f})')

    # ---- Pearson, all rows ----
    t0=time.time()
    r_p,n_p,_=pairwise_pearson(df,cpos,cols,rows)
    print('  pearson done, minutes: %.2f'%((time.time()-t0)/60))

    # ---- Pearson, snow-on rows ----
    t0=time.time()
    r_on,_,_=pairwise_pearson(df,cpos,cols,rows_on)
    print('  pearson snow-on done, minutes: %.2f'%((time.time()-t0)/60))
    del rows_on

    # ---- Spearman on a row sample ----
    t0=time.time()
    if SPEARMAN_SAMPLE and len(pool)>SPEARMAN_SAMPLE:
        samp=np.sort(rng.choice(pool,SPEARMAN_SAMPLE,replace=False))
    else:
        samp=pool
    n_samp=len(samp)
    ranks=df.iloc[samp,cpos].rank(method='average')
    del samp
    r_s,_,_=pairwise_pearson(ranks,list(range(k)),cols)
    del ranks
    print(f'  spearman done on {n_samp} rows, minutes: %.2f'%((time.time()-t0)/60))

    # constant / empty columns: kept as NaN so every domain has the same rows/columns in the same order
    dead=[c for c in cols if np.isnan(r_p.loc[c,c])]
    if dead: print('  constant or all-NaN in this domain (kept as NaN, grey in heatmaps):',dead)

    # ---- save full matrices (fixed order) ----
    tagname=f'{domain}_{season_lbl}_{run_tag}'
    r_p.to_csv(f'{res_dir}/corr_pearson_{tagname}.csv',float_format='%.4f')
    r_s.to_csv(f'{res_dir}/corr_spearman_{tagname}.csv',float_format='%.4f')
    r_on.to_csv(f'{res_dir}/corr_pearson_snowon_{tagname}.csv',float_format='%.4f')
    n_p.astype(np.int64).to_csv(f'{res_dir}/corr_npairs_{tagname}.csv')

    # ---- SWE vs every feature (fixed order; abs_pearson column for sorting afterwards) ----
    swe_tab=pd.DataFrame({'pearson':r_p.loc[TARGET,feat],
                          'spearman':r_s.loc[TARGET,feat],
                          'pearson_snowon':r_on.loc[TARGET,feat],
                          'n_pairs':n_p.loc[TARGET,feat].astype(np.int64),
                          'nan_frac':1.0-n_p.loc[TARGET,feat]/n_dom})
    swe_tab['abs_pearson']=swe_tab['pearson'].abs()
    swe_tab.index.name='feature'
    swe_tab.to_csv(f'{res_dir}/corr_swe_{tagname}.csv',float_format='%.4f')
    swe_all_domains[domain]=swe_tab[['pearson','spearman','pearson_snowon']]
    print('  top 20 |pearson| with SWE (log only; files keep the fixed order):')
    print(swe_tab.sort_values('abs_pearson',ascending=False).head(20)
          [['pearson','spearman','pearson_snowon','nan_frac']].round(3).to_string())

    # ---- highly correlated feature pairs (collinearity check, SWE excluded), fixed order ----
    rp=r_p.loc[feat,feat].to_numpy(); rs=r_s.loc[feat,feat].to_numpy()
    iu=np.triu_indices(len(feat),1)   # row-major: feature1 in cols order, then feature2 in cols order
    vals=rp[iu]; sel=np.abs(vals)>=HIGH_R
    fa=np.array(feat)
    pairs=pd.DataFrame({'feature1':fa[iu[0][sel]],'feature2':fa[iu[1][sel]],
                        'pearson':vals[sel],'spearman':rs[iu][sel]})
    pairs.to_csv(f'{res_dir}/highcorr_pairs_{tagname}.csv',index=False,float_format='%.4f')
    print(f'  feature pairs with |pearson|>={HIGH_R}: {len(pairs)}')
    print(pairs.head(20).round(3).to_string(index=False))

    # ---- heatmaps, fixed order ----
    plot_heatmap(r_p,f'Pearson r, {domain}, {season_lbl} ({run_tag})',
                 f'{res_dir}/heatmap_pearson_{tagname}')
    plot_heatmap(r_on,f'Pearson r snow-on (SWE>0), {domain}, {season_lbl} ({run_tag})',
                 f'{res_dir}/heatmap_pearson_snowon_{tagname}')
    plot_heatmap(r_s,f'Spearman rho, {domain}, {season_lbl}, {n_samp} row sample ({run_tag})',
                 f'{res_dir}/heatmap_spearman_{tagname}')

    # ---- bar chart: SWE correlation with every feature, fixed order top to bottom ----
    top=swe_tab.iloc[::-1]   # reversed so the first feature is at the top of barh
    y=np.arange(len(top)); h=0.27
    fig,ax=plt.subplots(figsize=(8,max(5,0.26*len(top)+1)))
    ax.barh(y+h,top['pearson'],h,label='Pearson (all rows)')
    ax.barh(y,top['spearman'],h,label='Spearman (all rows, sample)')
    ax.barh(y-h,top['pearson_snowon'],h,label='Pearson (snow-on)')
    ax.set_yticks(y); ax.set_yticklabels(top.index,fontsize=7)
    ax.set_ylim(-0.7,len(top)-0.3)
    ax.axvline(0,color='k',lw=0.8); ax.set_xlim(-1,1); ax.set_xlabel('correlation with SWE')
    ax.set_title(f'SWE correlation, {domain}, {season_lbl} ({run_tag})',pad=26)
    # legend above the axes (between title and plot) so it never covers any bars
    ax.legend(fontsize=8,loc='lower center',bbox_to_anchor=(0.5,1.0),ncol=3,frameon=False)
    ax.grid(True,axis='x',alpha=0.3); plt.tight_layout()
    plt.savefig(f'{res_dir}/bar_swe_corr_{tagname}.png',dpi=150,bbox_inches='tight')
    plt.savefig(f'{res_dir}/bar_swe_corr_{tagname}.svg',bbox_inches='tight'); plt.close()

    print('  outputs ->',f'{res_dir}/*_{tagname}.*')
    print('  domain time (min): %.2f'%((time.time()-domStart)/60))

# ---- SWE correlations side by side across domains, fixed order ----
if swe_all_domains:
    comb=pd.concat(swe_all_domains,axis=1).reindex(feat)
    comb.columns=[f'{m}_{d}' for d,m in comb.columns]
    comb.index.name='feature'
    comb.to_csv(f'{res_dir}/corr_swe_all_domains_{season_lbl}_{run_tag}.csv',float_format='%.4f')
    print('\ncombined SWE correlations ->',f'{res_dir}/corr_swe_all_domains_{season_lbl}_{run_tag}.csv')

executionTime=(time.time()-startTime)
print('\nAll domains done. Total execution time in minutes: %.2f'%(executionTime/60))