import glob
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import h3
import matplotlib.pyplot as plt
import pandas as pd
# Pre-process ECA&D precipitation data
# balance geographically, compute stats, and plot maps of station locations

# Paths
basepath = '/home/ubuntu/data/ML/training-data/precipitation/ecad/'
raw_path = basepath + 'RR/'
processed_path = basepath + 'RR_processed/'
stations_meta_file = basepath + 'stations_rr.txt'

period_start = pd.Timestamp('1995-01-01')
period_end = pd.Timestamp('2025-12-31')

# Europe bounding box 
BBOX = dict(lat_min=33, lat_max=83, lon_min=-30, lon_max=50)

# Per-country station targets for the balancing step
DOWNSAMPLE_TARGETS = {'DE': 2000} # had over 4900 stations, downsample to 2000 to avoid over-representation in ML training

# H3 resolution for within-country round-robin sampling (~26 km edge at res 4).
H3_RES = 4

# Extreme-event thresholds (mm/day)
THRESHOLDS = [50, 100, 200]

# --- Per-SOUID processing (from the original per-SOUID script) ------------
def dms_to_decimal(dms):
    s = dms.strip()
    sign = -1 if s.startswith('-') else 1
    deg, mins, secs = s.lstrip('+-').split(':')
    return sign * (int(deg) + int(mins) / 60 + int(secs) / 3600)

def find_header(path):
    with open(path, encoding='latin-1') as f:
        for i, line in enumerate(f):
            if line.lstrip().startswith('STAID'):
                return i

def read_ecad(path):
    return pd.read_csv(
        path,
        skiprows=find_header(path),
        skipinitialspace=True,
        parse_dates=['DATE'],
        na_values=-9999,
        encoding='latin-1',
    )


def read_stations(path):
    stations = pd.read_csv(
        path,
        skiprows=find_header(path),
        skipinitialspace=True,
        na_values=[-9999, -999],
        encoding='latin-1',
    )
    stations.columns = stations.columns.str.strip()
    for col in ['STANAME', 'CN']:
        stations[col] = stations[col].astype(str).str.strip()
        # Strip C0/C1 control bytes from latin-1 misdecodes
        stations[col] = stations[col].str.replace(r'[\x00-\x1f\x7f-\x9f]', '', regex=True)
    stations['LAT'] = stations['LAT'].apply(dms_to_decimal)
    stations['LON'] = stations['LON'].apply(dms_to_decimal)
    return stations


def process_one_souid(souid_file, stations_meta):
    df = read_ecad(souid_file)

    # Q_RR spec: 0=valid, 1=suspect, 9=missing. Drop anything else and
    # anything not flagged valid.
    df.loc[~df['Q_RR'].isin([0, 1, 9]), 'RR'] = pd.NA
    df.loc[df['Q_RR'] != 0, 'RR'] = pd.NA
    df.loc[df['RR'] < 0, 'RR'] = pd.NA

    df = df.merge(stations_meta, on='STAID', how='left')

    # Bounding-box filter
    df = df[(df['LAT'] <= BBOX['lat_max']) & (df['LAT'] >= BBOX['lat_min']) &
            (df['LON'] >= BBOX['lon_min']) & (df['LON'] <= BBOX['lon_max'])]

    # Date range
    df = df[(df['DATE'] >= period_start) & (df['DATE'] <= period_end)]

    df = df.dropna(subset=['RR', 'HGHT', 'LAT', 'LON'])
    df['RR'] = df['RR'] / 10  # 0.1 mm -> mm

    df['STAID'] = df['STAID'].astype(str)
    df['SOUID'] = df['SOUID'].astype(str)
    df['HGHT'] = df['HGHT'].astype(int)
    return df.reset_index(drop=True)


def process_all_souids():
    Path(processed_path).mkdir(parents=True, exist_ok=True)
    stations_meta = read_stations(stations_meta_file)

    files = sorted(glob.glob(raw_path + 'RR_SOUID*.txt'))
    print(f'Found {len(files)} SOUID files in {raw_path}')

    frames = []
    skipped = 0
    for i, f in enumerate(files, 1):
        try:
            df = process_one_souid(f, stations_meta)
        except Exception as e:
            print(f'  ERROR on {f}: {e}')
            skipped += 1
            continue
        if df.empty:
            skipped += 1
            continue
        souid = df['SOUID'].iloc[0]
        df.to_csv(f'{processed_path}RR_processed_SOUID{souid}.csv', index=False)
        frames.append(df)
        if i % 500 == 0:
            print(f'  {i}/{len(files)} processed (skipped {skipped})')

    print(f'Combining {len(frames)} non-empty frames...')
    combined = pd.concat(frames, ignore_index=True)
    print(f'Combined: {len(combined):,} rows from '
          f'{combined["STAID"].nunique()} unique stations')
    return combined


# --- Geographic balancing -------------------------------------------------
def build_station_inventory(df):
    """One row per station with coverage_pct over [period_start, period_end]."""
    total_days = (period_end - period_start).days + 1
    obs = df.groupby('STAID').size().rename('obs_count')
    meta = (
        df.groupby('STAID')
        .agg(LAT=('LAT', 'first'), LON=('LON', 'first'),
             CN=('CN', 'first'), HGHT=('HGHT', 'first'))
        .join(obs)
        .reset_index()
    )
    meta['coverage_pct'] = meta['obs_count'] / total_days
    return meta


def downsample_country(meta, country, target_n, h3_res):
    """H3-cell round-robin: cycle cells taking the best-coverage station each
    round, until target_n is reached or stations are exhausted."""
    sub = meta.loc[meta['CN'] == country].copy()
    if len(sub) <= target_n:
        print(f'  {country}: {len(sub)} <= target {target_n}, keeping all')
        return sub['STAID'].tolist()

    sub['h3_cell'] = [
        h3.latlng_to_cell(lat, lon, h3_res)
        for lat, lon in zip(sub['LAT'], sub['LON'])
    ]
    sub = sub.sort_values(['h3_cell', 'coverage_pct'], ascending=[True, False])
    by_cell = {c: g['STAID'].tolist() for c, g in sub.groupby('h3_cell', sort=False)}
    cells = list(by_cell.keys())

    selected = []
    rounds = 0
    while len(selected) < target_n:
        progressed = False
        for c in cells:
            if by_cell[c]:
                selected.append(by_cell[c].pop(0))
                progressed = True
                if len(selected) >= target_n:
                    break
        if not progressed:
            break
        rounds += 1

    print(f'  {country}: {len(sub)} -> {len(selected)} stations '
          f'across {len(cells)} H3 cells ({rounds} rounds)')
    return selected


def apply_balancing(df):
    meta = build_station_inventory(df)
    print(f'Station inventory: {len(meta)} stations, '
          f'mean coverage {100 * meta["coverage_pct"].mean():.1f}%')

    keep = []
    for cn in sorted(meta['CN'].unique()):
        if cn in DOWNSAMPLE_TARGETS:
            keep.extend(downsample_country(meta, cn, DOWNSAMPLE_TARGETS[cn], H3_RES))
        else:
            keep.extend(meta.loc[meta['CN'] == cn, 'STAID'].tolist())

    df_bal = df[df['STAID'].isin(keep)].reset_index(drop=True)
    print(f'After balancing: {len(df_bal):,} rows, '
          f'{df_bal["STAID"].nunique()} stations')
    return df_bal


# --- Statistics + map (from the original stats script) --------------------
def compute_statistics(df, label, output_path):
    stations_unique = (
        df[['STAID', 'CN', 'HGHT']]
        .drop_duplicates(subset=['STAID']).reset_index(drop=True)
    )

    country_stations = stations_unique.groupby('CN').agg(
        n_stations=('STAID', 'count'),
        min_hght=('HGHT', 'min'),
        max_hght=('HGHT', 'max'),
        mean_hght=('HGHT', 'mean'),
    ).round(1)

    country_rr = df.groupby('CN')['RR'].agg(['min', 'max', 'mean']).round(2)
    stats = country_stations.join(country_rr)

    extreme_cols = []
    for t in THRESHOLDS:
        days = df[df['RR'] >= t].groupby('CN').size().rename(f'days_{t}mm')
        sta = df[df['RR'] >= t].groupby('CN')['STAID'].nunique().rename(f'sta_{t}mm')
        stats = stats.join(days).join(sta)
        stats[f'days_{t}mm'] = stats[f'days_{t}mm'].fillna(0).astype(int)
        stats[f'sta_{t}mm'] = stats[f'sta_{t}mm'].fillna(0).astype(int)
        extreme_cols += [f'days_{t}mm', f'sta_{t}mm']

    stats = stats[[
        'n_stations', 'min', 'max', 'mean',
        'min_hght', 'max_hght', 'mean_hght',
        *extreme_cols,
    ]]
    labels = ['Stations', 'Min RR (mm)', 'Max RR (mm)', 'Mean RR (mm)',
              'Min HGHT (m)', 'Max HGHT (m)', 'Mean HGHT (m)']
    for t in THRESHOLDS:
        labels += [f'Days ≥{t} mm', f'Stations ≥{t} mm']
    stats.columns = labels
    stats.index.name = 'Country'
    stats = stats.sort_values('Stations', ascending=False)

    with open(output_path, 'w') as f:
        f.write(f'# ECA&D precipitation statistics — {label}\n\n')
        f.write(f'Total unique stations (STAID): **{len(stations_unique)}**\n\n')
        f.write(f'Total daily measurements: **{len(df):,}**\n\n')
        thresh_str = ', '.join(f'**≥{t} mm/day**' for t in THRESHOLDS)
        f.write(f'Extreme-event thresholds: {thresh_str}\n\n')
        f.write(stats.to_markdown())
        f.write('\n')
    print(f'Saved statistics -> {output_path}')


def plot_map(df, label, output_png, output_svg):
    locs = df[['LON', 'LAT']].drop_duplicates().reset_index(drop=True)
    pad = 2
    lon_min, lon_max = locs['LON'].min() - pad, locs['LON'].max() + pad
    lat_min, lat_max = locs['LAT'].min() - pad, locs['LAT'].max() + pad

    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f0', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#cfe2f3', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=':')
    ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
    ax.scatter(locs['LON'], locs['LAT'], s=4, color='crimson', alpha=0.6,
               transform=ccrs.PlateCarree(), zorder=3)
    ax.set_title(f'ECA&D precipitation stations — {label} '
                 f'({len(locs)} unique locations)')
    plt.savefig(output_png, dpi=200, bbox_inches='tight')
    plt.savefig(output_svg, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved map -> {output_png} and {output_svg}')


def write_dataset_outputs(df, suffix):
    """Save lonlat file, stations file, statistics MD, and map for one dataset."""
    n_sta = df['STAID'].nunique()
    label = f'{suffix}_{n_sta}stations'

    locs = df[['LON', 'LAT']].drop_duplicates().reset_index(drop=True)
    locs.to_csv(f'{basepath}ecad_lonlat_{label}.txt', sep=' ', index=False)
    print(f'Saved {len(locs)} lon/lat pairs -> ecad_lonlat_{label}.txt')

    stations = (
        df[['STAID', 'SOUID', 'LAT', 'LON']]
        .drop_duplicates().reset_index(drop=True)
    )
    stations.to_csv(f'{basepath}ecad_stations_{label}.csv', index=False)
    print(f'Saved {len(stations)} unique stations -> ecad_stations_{label}.csv')

    compute_statistics(df, label, f'{basepath}ecad_statistics_{label}.md')
    plot_map(df, label,
             f'{basepath}ecad_map_{label}.png',
             f'{basepath}ecad_map_{label}.svg')


# --- Main -----------------------------------------------------------------
# 1. Process every SOUID and combine
df_all = process_all_souids()

# 2. Save the combined all-stations CSV
n_all = df_all['STAID'].nunique()
all_csv = f'{basepath}RR_processed_all_{n_all}stations.csv'
df_all.to_csv(all_csv, index=False)
print(f'Saved combined CSV -> {all_csv}')

# 3. Stats + map for all stations
print(f'\n=== Outputs for ALL STATIONS ({n_all} stations) ===')
write_dataset_outputs(df_all, 'all')

# 4. Apply geographic balancing
print(f'\n=== Geographic balancing with targets {DOWNSAMPLE_TARGETS} ===')
df_bal = apply_balancing(df_all)

# 5. Save the balanced CSV
n_bal = df_bal['STAID'].nunique()
bal_csv = f'{basepath}RR_processed_balanced_{n_bal}stations.csv'
df_bal.to_csv(bal_csv, index=False)
print(f'Saved balanced CSV -> {bal_csv}')

# 6. Stats + map for balanced stations
print(f'\n=== Outputs for BALANCED STATIONS ({n_bal} stations) ===')
write_dataset_outputs(df_bal, 'balanced')
