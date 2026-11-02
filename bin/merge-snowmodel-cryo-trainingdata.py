import os, json, requests
import pandas as pd
import numpy as np

path=f'/home/ubuntu/data/cryoscope/snowmodel/'

# First, scandinavia datasets

# SWE from Snowcap
# SWE from Scandinavia must be multiplied by 1000 
hkl_input_scandi=f'{path}HKL_SWE_all_20150901-20230901.csv.gz'
df_hkl_scandi = pd.read_csv(hkl_input_scandi, dtype={'latitude': str, 'longitude': str})
df_hkl_scandi = df_hkl_scandi.dropna(subset=['SWE'])
df_hkl_scandi['SWE'] = df_hkl_scandi['SWE'] * 1000
print(df_hkl_scandi)

# DEM for Snowcap locations
dem_input_scandi =f'{path}training_dem+fch_merged_fin25_10000.csv'
df_dem_scandi = pd.read_csv(dem_input_scandi, dtype={'latitude': str, 'longitude': str})

# merge the two dataframes on latitude and longitude
df_merged_scandi = pd.merge(df_hkl_scandi, df_dem_scandi, on=['latitude', 'longitude'], how='inner')
df_merged_scandi['tag'] = 'Scandinavia'
df_merged_scandi['utctime'] = pd.to_datetime(df_merged_scandi['utctime'])
df_merged_scandi['day'] = df_merged_scandi['utctime'].dt.day
df_merged_scandi['month'] = df_merged_scandi['utctime'].dt.month
df_merged_scandi['year'] = df_merged_scandi['utctime'].dt.year
print(df_merged_scandi)

# TCF tree cover fraction (years > 2021 reuse the 2021 values)
tcf_scandi = f'{path}snow_scandi_tcf_long.csv'
df_tcf_scandi = pd.read_csv(tcf_scandi, dtype={'latitude': str, 'longitude': str})

df_merged_scandi['year_tcf'] = df_merged_scandi['year'].clip(upper=2021)
df_tcf_scandi = df_tcf_scandi.rename(columns={'year': 'year_tcf'})

df_merged_scandi = pd.merge(
    df_merged_scandi, df_tcf_scandi,
    on=['year_tcf', 'latitude', 'longitude'], how='inner'
).drop(columns='year_tcf')
# drop year
df_merged_scandi = df_merged_scandi.drop(columns=['year'])
print(df_merged_scandi)

# add SWE climatology
# also multiply scandi SWE climatology by 1000
swe_scandi = f'{path}SNOWCAP_swe_scandi_0-10000_20200101-20201231_dailystats.csv.gz'
sd_scandi = f'{path}SNOWCAP_snowdepth_scandi_0-10000_20200101-20201231_dailystats.csv.gz'
df_swe_scandi = pd.read_csv(swe_scandi, dtype={'latitude': str, 'longitude': str})
df_swe_scandi['swe_mean'] = df_swe_scandi['swe_mean'] * 1000
df_swe_scandi['swe_max'] = df_swe_scandi['swe_max'] * 1000
df_swe_scandi['swe_min'] = df_swe_scandi['swe_min'] * 1000
df_sd_scandi = pd.read_csv(sd_scandi, dtype={'latitude': str, 'longitude': str})
df_swe_scandi['utctime'] = pd.to_datetime(df_swe_scandi['utctime'])
df_sd_scandi['utctime'] = pd.to_datetime(df_sd_scandi['utctime'])
df_swe_scandi['day'] = df_swe_scandi['utctime'].dt.day
df_sd_scandi['day'] = df_sd_scandi['utctime'].dt.day
df_swe_scandi['month'] = df_swe_scandi['utctime'].dt.month
df_sd_scandi['month'] = df_sd_scandi['utctime'].dt.month
# drop utctime
df_swe_scandi = df_swe_scandi.drop(columns=['utctime'])
df_sd_scandi = df_sd_scandi.drop(columns=['utctime'])
# merge on day and month 
df_merged_scandi = pd.merge(df_merged_scandi, df_swe_scandi, on=['day', 'month', 'latitude', 'longitude'], how='inner')
df_merged_scandi = pd.merge(df_merged_scandi, df_sd_scandi, on=['day', 'month', 'latitude', 'longitude'], how='inner')
# drop day and month
df_merged_scandi = df_merged_scandi.drop(columns=['day', 'month'])
print(df_merged_scandi)

# reanalysis files
era5l_scandi_24h  = 'ERA5L_scandi_24hagg_00_10000.csv.gz' # only 00 utc values
era5l_scandi_inst = 'ERA5L_scandi_instant_wide_10000.csv.gz' # separated 00 and 12 utc to columns
era5ld_scandi     = 'ERA5LD_scandi_daily_10000.csv.gz'
era5_scandi       = 'ERA5_scandi_wide_10000.csv.gz' # separated 00 and 12 utc to columns

df_era5l_scandi_24h = pd.read_csv(f'{path}{era5l_scandi_24h}', dtype={'latitude': str, 'longitude': str})
df_era5l_scandi_24h['utctime'] = pd.to_datetime(df_era5l_scandi_24h['utctime'])
df_merged_scandi = pd.merge(df_merged_scandi, df_era5l_scandi_24h, on=['utctime', 'latitude', 'longitude'], how='inner')
print(df_merged_scandi)

df_era5l_scandi_inst = pd.read_csv(f'{path}{era5l_scandi_inst}', dtype={'latitude': str, 'longitude': str})
df_era5l_scandi_inst['utctime'] = pd.to_datetime(df_era5l_scandi_inst['date'])
# drop date columns
df_era5l_scandi_inst = df_era5l_scandi_inst.drop(columns=['date'])
df_merged_scandi = pd.merge(df_merged_scandi, df_era5l_scandi_inst, on=['utctime', 'latitude', 'longitude'], how='left')
print(df_merged_scandi)

df_era5_scandi = pd.read_csv(f'{path}{era5_scandi}', dtype={'latitude': str, 'longitude': str})
df_era5_scandi['utctime'] = pd.to_datetime(df_era5_scandi['date'])
# drop date column
df_era5_scandi = df_era5_scandi.drop(columns=['date'])
df_merged_scandi = pd.merge(df_merged_scandi, df_era5_scandi, on=['utctime', 'latitude', 'longitude'], how='inner')
print(df_merged_scandi)

df_era5ld_scandi = pd.read_csv(f'{path}{era5ld_scandi}', dtype={'latitude': str, 'longitude': str})
df_era5ld_scandi['utctime'] = pd.to_datetime(df_era5ld_scandi['utctime'])
df_merged_scandi = pd.merge(df_merged_scandi, df_era5ld_scandi, on=['utctime', 'latitude', 'longitude'], how='inner')

df_merged_scandi.columns = (
    df_merged_scandi.columns
      .str.replace('era5ld', '', regex=False)   # longer tag first
      .str.replace('_era5', '', regex=False)
      .str.replace('__', '_', regex=False)      # tidy any doubled underscore
      .str.strip('_')
)
print(df_merged_scandi)


# Then, Alps datasets

# SWE from Snowcap
hkl_input_alps=f'{path}HKL_SWE_all_alps_20160901-20240901.csv.gz'
df_hkl_alps = pd.read_csv(hkl_input_alps, dtype={'latitude': str, 'longitude': str})
df_hkl_alps = df_hkl_alps.dropna(subset=['SWE'])
# df_hkl_alps['SWE'] = df_hkl_alps['SWE'] * 1000   # uncomment only if Alps SWE is in metres too
print(df_hkl_alps)

# DEM for Snowcap locations
dem_input_alps =f'{path}training_dem+fch_merged_fin2_10000_alps.csv'
df_dem_alps = pd.read_csv(dem_input_alps, dtype={'latitude': str, 'longitude': str})

# merge the two dataframes on latitude and longitude
df_merged_alps = pd.merge(df_hkl_alps, df_dem_alps, on=['latitude', 'longitude'], how='inner')
df_merged_alps['tag'] = 'Alps'
df_merged_alps['utctime'] = pd.to_datetime(df_merged_alps['utctime'])
df_merged_alps['day'] = df_merged_alps['utctime'].dt.day
df_merged_alps['month'] = df_merged_alps['utctime'].dt.month
df_merged_alps['year'] = df_merged_alps['utctime'].dt.year
print(df_merged_alps)

# TCF tree cover fraction (years > 2021 reuse the 2021 values)
tcf_alps = f'{path}snow_alps_tcf_long.csv'
df_tcf_alps = pd.read_csv(tcf_alps, dtype={'latitude': str, 'longitude': str})

df_merged_alps['year_tcf'] = df_merged_alps['year'].clip(upper=2021)
df_tcf_alps = df_tcf_alps.rename(columns={'year': 'year_tcf'})

df_merged_alps = pd.merge(
    df_merged_alps, df_tcf_alps,
    on=['year_tcf', 'latitude', 'longitude'], how='inner'
).drop(columns='year_tcf')
# drop year
df_merged_alps = df_merged_alps.drop(columns=['year'])
print(df_merged_alps)

# add SWE climatology
swe_alps = f'{path}SNOWCAP_swe_alps_0-10000_20200101-20201231_dailystats.csv.gz'
sd_alps = f'{path}SNOWCAP_snowdepth_alps_0-10000_20200101-20201231_dailystats.csv.gz'
df_swe_alps = pd.read_csv(swe_alps, dtype={'latitude': str, 'longitude': str})
# df_swe_alps['swe_mean'] = df_swe_alps['swe_mean'] * 1000   # uncomment only if Alps SWE is in metres too
# df_swe_alps['swe_max']  = df_swe_alps['swe_max']  * 1000
# df_swe_alps['swe_min']  = df_swe_alps['swe_min']  * 1000
df_sd_alps = pd.read_csv(sd_alps, dtype={'latitude': str, 'longitude': str})
df_swe_alps['utctime'] = pd.to_datetime(df_swe_alps['utctime'])
df_sd_alps['utctime'] = pd.to_datetime(df_sd_alps['utctime'])
df_swe_alps['day'] = df_swe_alps['utctime'].dt.day
df_sd_alps['day'] = df_sd_alps['utctime'].dt.day
df_swe_alps['month'] = df_swe_alps['utctime'].dt.month
df_sd_alps['month'] = df_sd_alps['utctime'].dt.month
# drop utctime
df_swe_alps = df_swe_alps.drop(columns=['utctime'])
df_sd_alps = df_sd_alps.drop(columns=['utctime'])
# merge on day and month 
df_merged_alps = pd.merge(df_merged_alps, df_swe_alps, on=['day', 'month', 'latitude', 'longitude'], how='inner')
df_merged_alps = pd.merge(df_merged_alps, df_sd_alps, on=['day', 'month', 'latitude', 'longitude'], how='inner')
# drop day and month
df_merged_alps = df_merged_alps.drop(columns=['day', 'month'])
print(df_merged_alps)

# reanalysis files
era5l_alps_24h  = 'ERA5L_alps_24hagg_00_10000.csv.gz' # only 00 utc values
era5l_alps_inst = 'ERA5L_alps_instant_wide_10000.csv.gz' # separated 00 and 12 utc to columns
era5ld_alps     = 'ERA5LD_alps_daily_10000.csv.gz'
era5_alps       = 'ERA5_alps_wide_10000.csv.gz' # separated 00 and 12 utc to columns


df_era5l_alps_24h = pd.read_csv(f'{path}{era5l_alps_24h}', dtype={'latitude': str, 'longitude': str})
df_era5l_alps_24h['utctime'] = pd.to_datetime(df_era5l_alps_24h['utctime'])
df_merged_alps = pd.merge(df_merged_alps, df_era5l_alps_24h, on=['utctime', 'latitude', 'longitude'], how='inner')
print(df_merged_alps)

df_era5l_alps_inst = pd.read_csv(f'{path}{era5l_alps_inst}', dtype={'latitude': str, 'longitude': str})
df_era5l_alps_inst['utctime'] = pd.to_datetime(df_era5l_alps_inst['date'])
# drop date columns
df_era5l_alps_inst = df_era5l_alps_inst.drop(columns=['date'])
df_merged_alps = pd.merge(df_merged_alps, df_era5l_alps_inst, on=['utctime', 'latitude', 'longitude'], how='left')
print(df_merged_alps)

df_era5_alps = pd.read_csv(f'{path}{era5_alps}', dtype={'latitude': str, 'longitude': str})
df_era5_alps['utctime'] = pd.to_datetime(df_era5_alps['date'])
# drop date column
df_era5_alps = df_era5_alps.drop(columns=['date'])
df_merged_alps = pd.merge(df_merged_alps, df_era5_alps, on=['utctime', 'latitude', 'longitude'], how='inner')
print(df_merged_alps)

df_era5ld_alps = pd.read_csv(f'{path}{era5ld_alps}', dtype={'latitude': str, 'longitude': str})
df_era5ld_alps['utctime'] = pd.to_datetime(df_era5ld_alps['utctime'])
df_merged_alps = pd.merge(df_merged_alps, df_era5ld_alps, on=['utctime', 'latitude', 'longitude'], how='inner')

df_merged_alps.columns = (
    df_merged_alps.columns
      .str.replace('era5ld', '', regex=False)   # longer tag first
      .str.replace('_era5', '', regex=False)
      .str.replace('__', '_', regex=False)      # tidy any doubled underscore
      .str.strip('_')
)
print(df_merged_alps)


# Concatenate Scandinavia and Alps, then save

# sanity check that the columns line up before stacking
only_scandi = set(df_merged_scandi.columns) - set(df_merged_alps.columns)
only_alps   = set(df_merged_alps.columns) - set(df_merged_scandi.columns)
print('columns only in scandi:', only_scandi)
print('columns only in alps  :', only_alps)

df_all = pd.concat([df_merged_scandi, df_merged_alps], ignore_index=True)
print(df_all)
print('total rows:', len(df_all), 'scandi:', len(df_merged_scandi), 'alps:', len(df_merged_alps))

out_all = f'{path}training_all_scandi_alps_ext.csv'
df_all.to_csv(out_all, index=False)
print('saved to', out_all)


# separating 00 and 12 utc to columns: 

'''if os.path.exists(f'{path}trainingdata/ERA5_alps_wide_10000.csv.gz'):
    wide = pd.read_csv(f'{path}trainingdata/ERA5_alps_wide_10000.csv.gz', dtype={'latitude': str, 'longitude': str})
    wide['date'] = pd.to_datetime(wide['date']).dt.date
    print(wide)
else:
    df = pd.read_csv(f'{path}{era5_alps}', dtype={'latitude': str, 'longitude': str})
    print(df) # 5-6 mins

    # separate 00 and 12 steps into feat_00 / feat_12 columns, for ALL variables
    df['utctime'] = pd.to_datetime(df['utctime'])
    df['date']    = df['utctime'].dt.date
    df['hour']    = df['utctime'].dt.hour            # 0 or 12

    # every column that is not a key or a helper is a value column to pivot
    id_cols    = ['date', 'latitude', 'longitude']
    value_cols = [c for c in df.columns
                  if c not in id_cols + ['utctime', 'hour']]

    wide = df.pivot_table(index=id_cols,
                          columns='hour',
                          values=value_cols,
                          aggfunc='first')

    # flatten the (variable, hour) column index -> 'asn_00', 'asn_12', ...
    wide.columns = [f'{var}_{hour:02d}' for var, hour in wide.columns]
    wide = wide.reset_index()

    print(wide) # finished around 8-9 min
    # save wide to csv
    wide.to_csv(wide_alps_out, index=False)
'''