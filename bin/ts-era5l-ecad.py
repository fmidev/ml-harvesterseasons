#!/usr/bin/env python3

import os, requests, json, yaml,sys
from pathlib import Path
import pandas as pd
import numpy as np
from urllib.parse import quote_plus

# ERA5L instant, accumulated and daily statistics features for ECA&D precipitation stations

ecad_dir='/home/ubuntu/data/ML/training-data/ecad-nonblend/'
era5l_dir='/home/ubuntu/data/ML/training-data/ecad-nonblend/ERA5L/'

source1='http://ml.harvesterseasons.com:8080'
source2='http://smartmet.xyz:8080'
source3='http://sm.cryo-scope.eu:8080'

### ---------------------------------------###
# Functions

def parse_array_string(s):
    if not isinstance(s, str):
        return s
    s = s.strip()
    if s.startswith('['): s = s[1:]
    if s.endswith(']'):   s = s[:-1]
    out = []
    for tok in s.split():
        try:
            out.append(float(tok))
        except ValueError:
            out.append(np.nan)
    return out

def explode_response(df, feats):
    # accept a single feature name or a list of value columns
    if isinstance(feats, str):
        feats = [feats]
    cols = ['latitude', 'longitude'] + list(feats)
    for col in cols:
        df[col] = df[col].apply(parse_array_string)
    df = df.explode(cols, ignore_index=True)
    df['latitude']  = df['latitude'].astype(str)
    df['longitude'] = df['longitude'].astype(str)
    return df

def year_window(y, gstart, gend):
    # clamp the calendar year to the global [start, end] window
    ys = max(f'{y}0101T000000', gstart)
    ye = min(f'{y}1231T230000', gend)
    return ys, ye

def fetch_json(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url=url)
            if not response.content.strip():
                raise ValueError('Empty response')
            return json.loads(response.content)
        except (json.JSONDecodeError, ValueError) as e:
            print(f'  attempt {attempt+1} failed: {e}')
            if attempt + 1 == max_retries:
                raise
            print('  retrying...')

### ---------------------------------------###

# Read in location information
obs_file = os.path.join(ecad_dir, 'ecad_sources_stations.csv')
df_obs = pd.read_csv(obs_file, dtype={'latitude': str, 'longitude': str,'TX_STAID': str,'RR_STAID': str,'TG_STAID': str,'TN_STAID': str,'SD_STAID': str})
staid_cols = [c for c in df_obs.columns if c.endswith('_STAID')]
df_obs[staid_cols] = df_obs[staid_cols].apply(pd.to_numeric, errors='coerce')
df_obs['STAID'] = df_obs[staid_cols].bfill(axis=1).iloc[:, 0]
df_locs_full = df_obs[['lat', 'lon', 'STAID']].dropna(subset=['STAID']).copy()
df_locs_full['STAID'] = df_locs_full['STAID'].astype(int)
df_locs_full = df_locs_full.reset_index(drop=True)
print(df_locs_full)
df_locs_full.to_csv(os.path.join(ecad_dir, 'ecad_locations_staid_16072.csv'), index=False)
df_locs_full['lat'] = df_locs_full['lat'].astype(str)
df_locs_full['lon'] = df_locs_full['lon'].astype(str)

# lat/lon string -> STAID lookup
latlon_to_staid = dict(zip(
    zip(df_locs_full['lat'], df_locs_full['lon']),
    df_locs_full['STAID']
))

# unique lat/lon for fetching
df_locs = df_locs_full[['lat', 'lon']].drop_duplicates().reset_index(drop=True)

# build batches of 500 locations
batch_size = 500
loc_list = list(zip(df_locs['lat'], df_locs['lon']))
wkt_batches = [
    quote_plus("MULTIPOINT(" + ",".join(f"{lon} {lat}" for lat, lon in loc_list[i:i+batch_size]) + ")")
    for i in range(0, len(loc_list), batch_size)
]
wkt_debug = quote_plus("MULTIPOINT(" + ",".join(f"{lon} {lat}" for lat, lon in loc_list[:3]) + ")")
print(f"Total {len(loc_list)} locations -> {len(wkt_batches)} batches of up to {batch_size}")

# period
start = '19950101T000000' 
end   = '20251231T230000'

os.makedirs(era5l_dir, exist_ok=True)

# per-year backup files live here; used as checkpoints so a re-run can resume
# without re-downloading years that already completed
yearly_dir = os.path.join(era5l_dir, 'yearly')
os.makedirs(yearly_dir, exist_ok=True)



# from source1 available 
era5l_feats = {
    'fal_era5l': 'ALBEDOSLR-0TO1:ERA5L:5099:1:0:1:0', # forecast albedo
    'asn_era5l': 'ASN-0TO1:ERA5L:5099:1:0:1:0', # snow albedo
    #'es_era5l': 'ES-M:ERA5L:5099:1:0:1:0', # snow evaporation
    #'evabs_era5l': 'EVABS-M:ERA5L:5099:1:0:0', # evaporation from bare soil
    #'evaow_era5l': 'EVAOW-M:ERA5L:5099:1:0:0', # evaporation from open water surfaces excluding oceans
    'evap_era5l': 'EVAP-M:ERA5L:5099:1:0:1:0', # evaporation 
    'evapp_era5l': 'EVAPP-M:ERA5L:5099:1:0:1:0', # potential evaporation
    #'evatc_era5l': 'EVATC-M:ERA5L:5099:1:0:0', # evaporation from the top of canopy
    #'evavt_era5l': 'EVAVT-M:ERA5L:5099:1:0:0', # evaporation from vegetation transpiration
    'slhf_era5l': 'FLLAT-JM2:ERA5L:5099:1:0:1:0', # surface latent heat flux
    'sshf_era5l': 'FLSEN-JM2:ERA5L:5099:1:0:1:0', # surface sensible heat flux
    'sd_era5l': 'HSNOW-M:ERA5L:5099:1:0:1:0', # snow depth (m)
    'laihv_era5l': 'LAI_HV-M2M2:ERA5L:5099:1:0:1:0', # leaf area index for high vegetation
    'lailv_era5l': 'LAI_LV-M2M2:ERA5L:5099:1:0:1:0', # leaf area index for low vegetation
    #'lblt_era5l': 'LBLT-K:ERA5L:5099:1:0:1:0', # lake bottom temperature
    #'licd_era5l': 'LICD-M:ERA5L:5099:1:0:1:0', # lake ice total depth
    #'lict_era5l': 'LICT-K:ERA5L:5099:1:0:1:0', # lake ice surface temperature
    #'lmld_era5l': 'LMLD-M:ERA5L:5099:1:0:1:0', # lake mix-layer depth
    #'lmlt_era5l': 'LMLT-K:ERA5L:5099:1:0:1:0', # lake mix-layer temperature
    #'lshf_era5l': 'LSHF:ERA5L:5099:1:0:1:0', # lake shape factor
    #'ltlt_era5l': 'LTLT-K:ERA5L:5099:1:0:1:0', # lake total layer temperature
    'sp_era5l': 'PGR-PA:ERA5L:5099:1:0:1:0', # surface pressure
    'ssrd_era5l': 'RADGLOA-JM2:ERA5L:5099:1:0:1:0', # surface solar radiation downwards
    'strd_era5l': 'RADLWA-JM2:ERA5L:5099:1:0:1:0', # surface thermal radiation downwards
    'str_era5l': 'RNETLWA-JM2:ERA5L:5099:1:0:1:0', # surface net thermal radiation
    'ssr_era5l': 'RNETSWA-JM2:ERA5L:5099:1:0:1:0', # surface net solar radiation
    'ro_era5l': 'RO-M:ERA5L:5099:1:0:1:0', # runoff
    'tp_era5l': 'RR-M:ERA5L:5099:1:0:1:0', # total precipitation
    'skt_era5l': 'SKT-K:ERA5L:5099:1:0:1:0', # skin temperature
    'smlt_era5l': 'SMLT-M:ERA5L:5099:1:0:1:0', # snow melt
    'sf_era5l': 'SNACC-KGM2:ERA5L:5099:1:0:1:0', # snowfall
    'rsn_era5l': 'SND-KGM3:ERA5L:5099:1:0:1:0', # snow density
    'snowc_era5l': 'SNOWC-PRCNT:ERA5L:5099:1:0:0', # snow cover
    'swvl1_era5l': 'SOILWET-M3M3:ERA5L:5099:9:7:1:0', # soil water layer 1
    'swvl2_era5l': 'SWVL2-M3M3:ERA5L:5099:9:1820:1:0', # soil water layer 2
    'swvl3_era5l': 'SWVL3-M3M3:ERA5L:5099:9:7268:1:0', # soil water layer 3
    'swvl4_era5l': 'SWVL4-M3M3:ERA5L:5099:9:25855:1:0', # soil water layer 4
    'src_era5l': 'SRC-M:ERA5L:5099:1:0:1:0', # skin reservoir content
    'sro_era5l': 'SRO-M:ERA5L:5099:1:0:1:0', # surface runoff
    'ssro_era5l': 'SSRO-M:ERA5L:5099:1:0:1:0', # sub-surface runoff
    'stl1_era5l': 'STL1-K:ERA5L:5099:9:7:1:0', # soil temperature layer 1
    'stl2_era5l': 'TSOIL-K:ERA5L:5099:9:1820:1:0', # soil temperature layer 2    
    'stl3_era5l': 'STL3-K:ERA5L:5099:9:7268:1:0', # soil temperature layer 3
    'stl4_era5l': 'STL4-K:ERA5L:5099:9:25855:1:0', # soil temperature layer 4
    't2_era5l': 'T2-K:ERA5L:5099:1:0:1:0', # 2m temperature
    'td2_era5l': 'TD2-K:ERA5L:5099:1:0:1:0', # 2m dewpoint temperature
    'u10_era5l': 'U10-M:ERA5L:5099:1:0:1:0', # 10m u wind component
    'v10_era5l': 'V10-M:ERA5L:5099:1:0:1:0', # 10m v wind component
}

for feat, fmikey in era5l_feats.items():
    out = os.path.join(era5l_dir, f'era5l_{feat}_{start[:8]}-{end[:8]}.csv')
    staid_dir = os.path.join(era5l_dir, 'by_staid', feat)
    if os.path.exists(out):
        print(f'skipping (exists): {feat}')
        continue

    yearly = []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        ys, ye = year_window(y, start, end)
        if ys > ye:
            continue

        year_file = os.path.join(yearly_dir, f'era5l_{feat}_{y}.csv')
        if os.path.exists(year_file):
            print(f'loading cached year {feat} {y}')
            df_year = pd.read_csv(year_file, dtype={'latitude': str, 'longitude': str})
            if not df_year.dropna().empty:
                yearly.append(df_year)
            continue

        print(f'fetching {feat} {ys[:8]}-{ye[:8]}')
        batch_dfs = []
        for b_idx, wkt_b in enumerate(wkt_batches):
            q = (
                f"{source1}/timeseries"
                f"?wkt={wkt_b}"
                f"&param=utctime,latitude,longitude,{fmikey} as {feat}"
                f"&starttime={ys}Z&endtime={ye}Z&hour=00,12"
                f"&format=json&precision=full&timeformat=sql&tz=utc"
            )
            if y == int(start[:4]) and b_idx == 0:
                print(f"  debug query (batch 1, first 3 locs): {q.replace(wkt_b, wkt_debug)}")
            results = fetch_json(q)
            df_b = pd.DataFrame(results)
            if df_b.empty:
                continue
            df_b.columns = ['utctime', 'latitude', 'longitude', feat]
            df_b = explode_response(df_b, feat)
            if not df_b.dropna().empty:
                batch_dfs.append(df_b)

        if batch_dfs:
            df_year = pd.concat(batch_dfs, ignore_index=True)
        else:
            df_year = pd.DataFrame(columns=['utctime', 'latitude', 'longitude', feat])
        df_year['latitude'] = df_year['latitude'].astype(str)
        df_year['longitude'] = df_year['longitude'].astype(str)
        df_year.to_csv(year_file, index=False)
        print(f'  saved yearly backup: {year_file}')
        if not df_year.dropna().empty:
            yearly.append(df_year)

    if not yearly:
        print(f"  no valid data for {feat} across all years, skipping save.")
        continue

    df_all = pd.concat(yearly, ignore_index=True)
    df_all['latitude'] = df_all['latitude'].astype(str)
    df_all['longitude'] = df_all['longitude'].astype(str)
    df_all.to_csv(out, index=False)

    os.makedirs(staid_dir, exist_ok=True)
    for (lat, lon), grp in df_all.groupby(['latitude', 'longitude']):
        staid = latlon_to_staid.get((str(lat), str(lon)))
        if staid is None:
            continue
        grp.reset_index(drop=True).to_csv(os.path.join(staid_dir, f'{feat}_{staid}.csv'), index=False)
    print(f'  saved per-STAID files to {staid_dir}')


era5ldaily_aggs = {
    't': 'T2-K:ERA5LD:5093:1:0:1', # daily 2m temp stats
}

source='http://sm.cryo-scope.eu:8080'
# daily maximum
for feat, fmikey in era5ldaily_aggs.items():
    feat = feat + 'max_era5ld'
    out = os.path.join(era5l_dir, f'era5ld_{feat}_{start[:8]}-{end[:8]}.csv')
    staid_dir = os.path.join(era5l_dir, 'by_staid', feat)
    if os.path.exists(out):
        print(f'skipping (exists): {feat}')
        continue

    yearly = []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        ys, ye = year_window(y, start, end)
        if ys > ye:
            continue

        year_file = os.path.join(yearly_dir, f'era5ld_{feat}_{y}.csv')
        if os.path.exists(year_file):
            print(f'loading cached year {feat} {y}')
            df_year = pd.read_csv(year_file, dtype={'latitude': str, 'longitude': str})
            if not df_year.dropna().empty:
                yearly.append(df_year)
            continue

        print(f'fetching {feat} {ys[:8]}-{ye[:8]}')
        batch_dfs = []
        for b_idx, wkt_b in enumerate(wkt_batches):
            q = (
                f"{source}/timeseries"
                f"?wkt={wkt_b}"
                f"&param=utctime,latitude,longitude,{fmikey} as {feat}"
                f"&starttime={ys}Z&endtime={ye}Z&hour=00"
                f"&format=json&precision=full&timeformat=sql&tz=utc&origintime=20000201T000000"
            )
            if y == int(start[:4]) and b_idx == 0:
                print(f"  debug query (batch 1, first 3 locs): {q.replace(wkt_b, wkt_debug)}")
            results = fetch_json(q)
            df_b = pd.DataFrame(results)
            if df_b.empty:
                continue
            df_b.columns = ['utctime', 'latitude', 'longitude', feat]
            df_b = explode_response(df_b, feat)
            if not df_b.dropna().empty:
                batch_dfs.append(df_b)

        if batch_dfs:
            df_year = pd.concat(batch_dfs, ignore_index=True)
        else:
            df_year = pd.DataFrame(columns=['utctime', 'latitude', 'longitude', feat])
        df_year['latitude'] = df_year['latitude'].astype(str)
        df_year['longitude'] = df_year['longitude'].astype(str)
        df_year.to_csv(year_file, index=False)
        print(f'  saved yearly backup: {year_file}')
        if not df_year.dropna().empty:
            yearly.append(df_year)

    if not yearly:
        print(f"  no valid data for {feat} across all years, skipping save.")
        continue

    df_all = pd.concat(yearly, ignore_index=True)
    df_all['latitude'] = df_all['latitude'].astype(str)
    df_all['longitude'] = df_all['longitude'].astype(str)
    df_all.to_csv(out, index=False)

    os.makedirs(staid_dir, exist_ok=True)
    for (lat, lon), grp in df_all.groupby(['latitude', 'longitude']):
        staid = latlon_to_staid.get((str(lat), str(lon)))
        if staid is None:
            continue
        grp.reset_index(drop=True).to_csv(os.path.join(staid_dir, f'{feat}_{staid}.csv'), index=False)
    print(f'  saved per-STAID files to {staid_dir}')

# daily minimum
for feat, fmikey in era5ldaily_aggs.items():
    feat = feat + 'min_era5ld'
    out = os.path.join(era5l_dir, f'era5ld_{feat}_{start[:8]}-{end[:8]}.csv')
    staid_dir = os.path.join(era5l_dir, 'by_staid', feat)
    if os.path.exists(out):
        print(f'skipping (exists): {feat}')
        continue

    yearly = []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        ys, ye = year_window(y, start, end)
        if ys > ye:
            continue

        year_file = os.path.join(yearly_dir, f'era5ld_{feat}_{y}.csv')
        if os.path.exists(year_file):
            print(f'loading cached year {feat} {y}')
            df_year = pd.read_csv(year_file, dtype={'latitude': str, 'longitude': str})
            if not df_year.dropna().empty:
                yearly.append(df_year)
            continue

        print(f'fetching {feat} {ys[:8]}-{ye[:8]}')
        batch_dfs = []
        for b_idx, wkt_b in enumerate(wkt_batches):
            q = (
                f"{source}/timeseries"
                f"?wkt={wkt_b}"
                f"&param=utctime,latitude,longitude,{fmikey} as {feat}"
                f"&starttime={ys}Z&endtime={ye}Z&hour=00"
                f"&format=json&precision=full&timeformat=sql&tz=utc&origintime=20000301T000000"
            )
            if y == int(start[:4]) and b_idx == 0:
                print(f"  debug query (batch 1, first 3 locs): {q.replace(wkt_b, wkt_debug)}")
            results = fetch_json(q)
            df_b = pd.DataFrame(results)
            if df_b.empty:
                continue
            df_b.columns = ['utctime', 'latitude', 'longitude', feat]
            df_b = explode_response(df_b, feat)
            if not df_b.dropna().empty:
                batch_dfs.append(df_b)

        if batch_dfs:
            df_year = pd.concat(batch_dfs, ignore_index=True)
        else:
            df_year = pd.DataFrame(columns=['utctime', 'latitude', 'longitude', feat])
        df_year['latitude'] = df_year['latitude'].astype(str)
        df_year['longitude'] = df_year['longitude'].astype(str)
        df_year.to_csv(year_file, index=False)
        print(f'  saved yearly backup: {year_file}')
        if not df_year.dropna().empty:
            yearly.append(df_year)

    if not yearly:
        print(f"  no valid data for {feat} across all years, skipping save.")
        continue

    df_all = pd.concat(yearly, ignore_index=True)
    df_all['latitude'] = df_all['latitude'].astype(str)
    df_all['longitude'] = df_all['longitude'].astype(str)
    df_all.to_csv(out, index=False)

    os.makedirs(staid_dir, exist_ok=True)
    for (lat, lon), grp in df_all.groupby(['latitude', 'longitude']):
        staid = latlon_to_staid.get((str(lat), str(lon)))
        if staid is None:
            continue
        grp.reset_index(drop=True).to_csv(os.path.join(staid_dir, f'{feat}_{staid}.csv'), index=False)
    print(f'  saved per-STAID files to {staid_dir}')

# daily mean
for feat, fmikey in era5ldaily_aggs.items():
    feat = feat + 'mean_era5ld'
    out = os.path.join(era5l_dir, f'era5ld_{feat}_{start[:8]}-{end[:8]}.csv')
    staid_dir = os.path.join(era5l_dir, 'by_staid', feat)
    if os.path.exists(out):
        print(f'skipping (exists): {feat}')
        continue

    yearly = []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        ys, ye = year_window(y, start, end)
        if ys > ye:
            continue

        year_file = os.path.join(yearly_dir, f'era5ld_{feat}_{y}.csv')
        if os.path.exists(year_file):
            print(f'loading cached year {feat} {y}')
            df_year = pd.read_csv(year_file, dtype={'latitude': str, 'longitude': str})
            if not df_year.dropna().empty:
                yearly.append(df_year)
            continue

        print(f'fetching {feat} {ys[:8]}-{ye[:8]}')
        batch_dfs = []
        for b_idx, wkt_b in enumerate(wkt_batches):
            q = (
                f"{source}/timeseries"
                f"?wkt={wkt_b}"
                f"&param=utctime,latitude,longitude,{fmikey} as {feat}"
                f"&starttime={ys}Z&endtime={ye}Z&hour=00"
                f"&format=json&precision=full&timeformat=sql&tz=utc&origintime=20000101T000000"
            )
            if y == int(start[:4]) and b_idx == 0:
                print(f"  debug query (batch 1, first 3 locs): {q.replace(wkt_b, wkt_debug)}")
            results = fetch_json(q)
            df_b = pd.DataFrame(results)
            if df_b.empty:
                continue
            df_b.columns = ['utctime', 'latitude', 'longitude', feat]
            df_b = explode_response(df_b, feat)
            if not df_b.dropna().empty:
                batch_dfs.append(df_b)

        if batch_dfs:
            df_year = pd.concat(batch_dfs, ignore_index=True)
        else:
            df_year = pd.DataFrame(columns=['utctime', 'latitude', 'longitude', feat])
        df_year['latitude'] = df_year['latitude'].astype(str)
        df_year['longitude'] = df_year['longitude'].astype(str)
        df_year.to_csv(year_file, index=False)
        print(f'  saved yearly backup: {year_file}')
        if not df_year.dropna().empty:
            yearly.append(df_year)

    if not yearly:
        print(f"  no valid data for {feat} across all years, skipping save.")
        continue

    df_all = pd.concat(yearly, ignore_index=True)
    df_all['latitude'] = df_all['latitude'].astype(str)
    df_all['longitude'] = df_all['longitude'].astype(str)
    df_all.to_csv(out, index=False)

    os.makedirs(staid_dir, exist_ok=True)
    for (lat, lon), grp in df_all.groupby(['latitude', 'longitude']):
        staid = latlon_to_staid.get((str(lat), str(lon)))
        if staid is None:
            continue
        grp.reset_index(drop=True).to_csv(os.path.join(staid_dir, f'{feat}_{staid}.csv'), index=False)
    print(f'  saved per-STAID files to {staid_dir}')


'''
# pressure-level instant features
era5_pls = {
    't500_era5': 'T-K:ERA5:5021:2:500:1:0', # 500hPa temperature
    't700_era5': 'T-K:ERA5:5021:2:700:1:0', # 700hPa temperature
    't850_era5': 'T-K:ERA5:5021:2:850:1:0', # 850hPa temperature
    'z500_era5': 'Z-M2S2:ERA5:5021:2:500:1:0', # 500hPa geopotential height
    'z700_era5': 'Z-M2S2:ERA5:5021:2:700:1:0', # 700hPa geopotential height
    'z850_era5': 'Z-M2S2:ERA5:5021:2:850:1:0', # 850hPa geopotential height
    'u500_era5': 'U-MS:ERA5:5021:2:500:1:0', # 500hPa u wind component
    'u700_era5': 'U-MS:ERA5:5021:2:700:1:0', # 700hPa u wind component
    'u850_era5': 'U-MS:ERA5:5021:2:850:1:0', # 850hPa u wind component
    'v500_era5': 'V-MS:ERA5:5021:2:500:1:0', # 500hPa v wind component
    'v700_era5': 'V-MS:ERA5:5021:2:700:1:0', # 700hPa v wind component
    'v850_era5': 'V-MS:ERA5:5021:2:850:1:0',  # 850hPa v wind component
    'q500_era5': 'Q-KGKG:ERA5:5021:2:500:1:0', # 500hPa specific humidity
    'q700_era5': 'Q-KGKG:ERA5:5021:2:700:1:0', # 700hPa specific humidity
    'q850_era5': 'Q-KGKG:ERA5:5021:2:850:1:0'  # 850hPa specific humidity
}

source = 'http://smartmet.xyz:8080'
for feat, fmikey in era5_pls.items():
    out = os.path.join(era5l_dir, f'era5_{feat}_{start[:8]}-{end[:8]}.csv')
    if os.path.exists(out):
        print(f'skipping (exists): {feat}')
        continue

    yearly = []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        ys, ye = year_window(y, start, end)
        if ys > ye:
            continue

        # per-year backup / checkpoint for this feature+year
        year_file = os.path.join(yearly_dir, f'era5_{feat}_{y}.csv')

        # if this year was already fetched, load the backup instead of downloading
        if os.path.exists(year_file):
            print(f'loading cached year {feat} {y}: {year_file}')
            df_year = pd.read_csv(year_file, dtype={'latitude': str, 'longitude': str})
            if not df_year.dropna().empty:
                yearly.append(df_year)
            continue

        print(f'fetching {feat} {ys[:8]}-{ye[:8]}')
        year_parts = []
            q = (
                f"{source}/timeseries"
                f"?wkt={wkt_param}"
                f"&param=time,latitude,longitude,{fmikey} as {feat}" # query is for local time!!
                f"&starttime={ys}Z&endtime={ye}Z&hour="
                f"&format=json&precision=full&timeformat=sql"
            )
            if y == int(start[:4]) and i == 0:
                print(f"  debug query (first 3 locs): {q.replace(wkt_param, quote_plus(wkt_debug))}")

            response = requests.get(url=q)
            df_h = pd.DataFrame(json.loads(response.content))
            if df_h.empty:                       # no rows for this hour/year -> skip
                continue
            df_h.columns = ['time', 'latitude', 'longitude', feat]
            df_h = explode_response(df_h, feat)
            if not df_h.dropna().empty:
                year_parts.append(df_h)
            print(df_h.dropna())
        # build the combined year frame; write it out as a backup checkpoint
        if year_parts:
            df_year = pd.concat(year_parts, ignore_index=True)
        else:
            # no valid data this year — still write an (empty) marker so the
            # year is treated as done and not re-fetched on the next run
            df_year = pd.DataFrame(columns=['time', 'latitude', 'longitude', feat])
        df_year['latitude']  = df_year['latitude'].astype(str)
        df_year['longitude'] = df_year['longitude'].astype(str)
        df_year.to_csv(year_file, index=False)
        print(f'  saved yearly backup: {year_file}')

        if not df_year.dropna().empty:
            yearly.append(df_year)

    if not yearly:
        print(f"  no valid data for {feat} across all years, skipping save.")
        continue

    # concat all years (freshly fetched + loaded from backups), then write once
    df_all = pd.concat(yearly, ignore_index=True)
    df_all['latitude']  = df_all['latitude'].astype(str)
    df_all['longitude'] = df_all['longitude'].astype(str)
    print(df_all)
    df_all.to_csv(out, index=False)

# era5 features
era5_feats = {
    'cape_era5': 'CAPE-JKG:ERA5:5101:1:0:1:0', 
    'cbh_era5': 'CBH-M:ERA5:5101:1:0:1:0', # cloud base height
    'cin_era5': 'CIN-JKG:ERA5:5101:1:0:1:0', # convective inhibition
    'kx_era5': 'KX:ERA5:5101:1:0:0', 
    'lspf_era5': 'LSRR-KGM2:ERA5:5101:1:0:1:0', # large scale rain rate
    'lssfr_era5': 'LSSFR-KGM2S:ERA5:5101:1:0:1:0', # large scale snow fall rate
    'blh_era5': 'MIXHGT-M:ERA5:5101:1:0:1:0', # boundary layer height
    'tcc_era5': 'N-0TO1:ERA5:5101:1:0:1:0', # total cloud cover
    'tptype_era5': 'PRECFORM-N:ERA5:5101:1:0:0', # precipitation type
    'tciw_era5': 'TCIW-KGM2:ERA5:5101:1:0:1:0', 
    'tlcw_era5': 'TCLW-KGM2:ERA5:5101:1:0:1:0', 
    'tcrw_era5': 'TCRW-KGM2:ERA5:5101:1:0:1:0', 
    'tcslw_era5': 'TCSLW-KGM2:ERA5:5101:1:0:1:0',
    'tcw_era5': 'TCW-KGM2:ERA5:5101:1:0:1:0', 
    'totcwv_era5': 'TOTCWV-KGM2:ERA5:5101:1:0:1:0',   
}
'''