import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # headless server
import matplotlib.pyplot as plt

# ---- config ----
path   = '/home/ubuntu/data/cryoscope/snowmodel/'
infile = f'{path}training_all_scandi_alps.csv.gz'
outdir = f'{path}eda_swe/'
os.makedirs(outdir, exist_ok=True)

TARGET   = 'SWE'
SAMPLE_N = 500_000
SEED     = 0

df = pd.read_csv(infile, dtype={'latitude': str, 'longitude': str},
                 parse_dates=['utctime'])
df['month'] = df['utctime'].dt.month
df['year']  = df['utctime'].dt.year
# hydrological year: Sep-Aug, labelled by the Jan it contains
df['hydro_year'] = np.where(df['month'] >= 9, df['year'] + 1, df['year'])

num_cols  = list(df.select_dtypes(include=[np.number]).columns)
feat_cols = [c for c in num_cols if c not in (TARGET, 'month', 'year', 'hydro_year')]

print('shape:', df.shape)
print('rows per tag:\n', df['tag'].value_counts())

# ---- 0. target integrity: SWE must be physical ----
swe = df[TARGET]
report = {
    'n':            len(swe),
    'n_nan':        int(swe.isna().sum()),
    'n_negative':   int((swe < 0).sum()),          # SWE can never be < 0
    'n_zero':       int((swe == 0).sum()),
    'zero_frac':    float((swe == 0).mean()),
    'max':          float(swe.max()),
    'p99':          float(swe.quantile(0.99)),
    'p999':         float(swe.quantile(0.999)),
}
pd.Series(report).to_csv(f'{outdir}swe_integrity.csv')
print('\nSWE integrity:\n', pd.Series(report))
if report['n_negative']:
    print('WARNING: negative SWE present, these are physically impossible')
# after the *1000 scaling, Scandi SWE is in mm; flag suspicit magnitudes
print('SWE rows above 2000 mm (extreme, verify):', int((swe > 2000).sum()))

# ---- 1. SWE distribution, per tag, split zero vs positive ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for tag, g in df.groupby('tag'):
    ax[0].bar(tag, (g[TARGET] == 0).mean())
    pos = g[TARGET][g[TARGET] > 0]
    ax[1].hist(np.log1p(pos), bins=100, alpha=.5, label=f'{tag} (n={len(pos):,})')
ax[0].set_title('zero-SWE fraction'); ax[0].set_ylabel('fraction')
ax[1].set_title('log1p(SWE), positive only'); ax[1].set_xlabel('log1p(mm)'); ax[1].legend()
plt.tight_layout(); plt.savefig(f'{outdir}swe_distribution.png', dpi=120); plt.close()

# ---- 2. seasonal cycle: mean SWE and snow-cover frequency by month ----
season = df.groupby(['tag', 'month']).agg(
    mean_swe=(TARGET, 'mean'),
    cover_freq=(TARGET, lambda s: (s > 0).mean()),
).reset_index()
season.to_csv(f'{outdir}swe_seasonal_cycle.csv', index=False)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for tag, g in season.groupby('tag'):
    ax[0].plot(g['month'], g['mean_swe'], marker='o', label=tag)
    ax[1].plot(g['month'], g['cover_freq'], marker='o', label=tag)
ax[0].set_title('mean SWE by month'); ax[0].set_xlabel('month'); ax[0].set_ylabel('mm'); ax[0].legend()
ax[1].set_title('snow-cover frequency (SWE>0)'); ax[1].set_xlabel('month'); ax[1].legend()
plt.tight_layout(); plt.savefig(f'{outdir}swe_seasonal_cycle.png', dpi=120); plt.close()

# ---- 3. SWE vs elevation (the dominant snow control) ----
if 'height' in df.columns:
    samp = df.sample(min(SAMPLE_N, len(df)), random_state=SEED)
    samp = samp[samp[TARGET] > 0]                 # snow-on only
    bins = pd.cut(samp['height'], bins=20)
    prof = samp.groupby([bins, 'tag'], observed=True)[TARGET].median().unstack('tag')
    prof.index = [int(iv.mid) for iv in prof.index]
    prof.to_csv(f'{outdir}swe_vs_elevation.csv')
    ax = prof.plot(marker='o', figsize=(8, 4))
    ax.set_xlabel('elevation (m)'); ax.set_ylabel('median SWE (mm), snow-on')
    ax.set_title('SWE vs elevation')
    plt.tight_layout(); plt.savefig(f'{outdir}swe_vs_elevation.png', dpi=120); plt.close()

# ---- 4. peak-season SWE map (spot bad points / stripes) ----
peak = df[(df['month'].isin([2, 3])) & (df[TARGET] > 0)]
peak = peak.sample(min(SAMPLE_N, len(peak)), random_state=SEED)
lat = peak['latitude'].astype(float); lon = peak['longitude'].astype(float)
sc = plt.scatter(lon, lat, c=peak[TARGET], s=2, cmap='turbo',
                 vmax=peak[TARGET].quantile(0.99))
plt.colorbar(sc, label='SWE (mm)'); plt.title('Feb-Mar SWE (99th-pct clip)')
plt.xlabel('lon'); plt.ylabel('lat'); plt.tight_layout()
plt.savefig(f'{outdir}swe_map_peak.png', dpi=120); plt.close()

# ---- 5. climatology as a reference feature: does clim track observed SWE? ----
clim_cols = [c for c in ['swe_mean', 'swe_max', 'swe_min'] if c in df.columns]
if clim_cols:
    samp = df.sample(min(SAMPLE_N, len(df)), random_state=SEED)
    rows = []
    for c in clim_cols:
        r = samp[[TARGET, c]].dropna()
        rows.append({'feature': c,
                     'spearman_vs_SWE': r[TARGET].corr(r[c], method='spearman')})
    pd.DataFrame(rows).to_csv(f'{outdir}swe_climatology_link.csv', index=False)
    print('\nclimatology vs observed SWE (Spearman):')
    print(pd.DataFrame(rows))

# ---- 6. feature ranking against SWE, computed on snow-on rows ----
# correlating over all rows lets the zero mass dominate; restrict to SWE>0
samp = df[df[TARGET] > 0].sample(min(SAMPLE_N, (df[TARGET] > 0).sum()),
                                 random_state=SEED)
corr = samp[[TARGET] + feat_cols].corr(method='spearman')[TARGET].drop(TARGET)
corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
corr.to_csv(f'{outdir}swe_feature_corr.csv')
print('\ntop |corr| with SWE (snow-on rows):\n', corr.head(20))
ax = corr.head(25)[::-1].plot.barh(figsize=(8, 7))
ax.set_xlabel('Spearman corr with SWE (SWE>0)'); plt.tight_layout()
plt.savefig(f'{outdir}swe_feature_corr.png', dpi=120); plt.close()

# ---- 7. missingness, but weighted toward when it matters (snow-on rows) ----
miss_all   = df[num_cols].isna().mean()
miss_snow  = df[df[TARGET] > 0][num_cols].isna().mean()
miss = pd.DataFrame({'all_rows': miss_all, 'snow_on_rows': miss_snow})
miss = miss[miss['all_rows'] > 0].sort_values('all_rows', ascending=False)
miss.to_csv(f'{outdir}missingness.csv')
print('\ncolumns with NaN (fraction), all vs snow-on:\n', miss.head(20))

# ---- 8. per hydrological year: is every season populated? ----
hy = df.groupby(['tag', 'hydro_year']).agg(
    n=(TARGET, 'size'),
    mean_swe=(TARGET, 'mean'),
    zero_frac=(TARGET, lambda s: (s == 0).mean()),
)
hy.to_csv(f'{outdir}swe_by_hydro_year.csv')
print('\nper hydrological year (head):\n', hy.head(12))

print('\nSWE EDA written to', outdir)