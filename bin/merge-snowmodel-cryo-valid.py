import os, json, requests
import pandas as pd
import numpy as np

# merge validation datasets 

path=f'/home/ubuntu/data/cryoscope/snowmodel/validationdata/'

# First, scandinavia datasets

# SWE from Snowcap
# SWE from Scandinavia must be multiplied by 1000 
hkl_input_scandi=f'{path}HKL_SWE_scandi_valid_0-10000_20150901-20230901.csv.gz'
df_hkl_scandi = pd.read_csv(hkl_input_scandi, dtype={'latitude': str, 'longitude': str})
df_hkl_scandi = df_hkl_scandi.dropna(subset=['SWE'])
df_hkl_scandi['SWE'] = df_hkl_scandi['SWE'] * 1000
print(df_hkl_scandi)

# DEM for Snowcap locations
dem_input_scandi =f'{path}validation_dem+fch_merged_fin25_10000_fscs1_scandi.csv.gz'
df_dem_scandi = pd.read_csv(dem_input_scandi, dtype={'latitude': str, 'longitude': str})
# drop "dist_to_train_m" column
df_dem_scandi = df_dem_scandi.drop(columns=['dist_to_train_m'], errors='ignore')

# merge the two dataframes on latitude and longitude
df_merged_scandi = pd.merge(df_hkl_scandi, df_dem_scandi, on=['latitude', 'longitude'], how='inner')
df_merged_scandi['tag'] = 'Scandinavia'
df_merged_scandi['utctime'] = pd.to_datetime(df_merged_scandi['utctime'])
df_merged_scandi['day'] = df_merged_scandi['utctime'].dt.day
df_merged_scandi['month'] = df_merged_scandi['utctime'].dt.month
df_merged_scandi['year'] = df_merged_scandi['utctime'].dt.year
print(df_merged_scandi)

# TCF tree cover fraction (years > 2021 reuse the 2021 values)
tcf_scandi = f'{path}snow_scandi_tcf_valid.txt'
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
swe_scandi = f'{path}SNOWCAP_swe_scandi_valid_0-10000_20200101-20201231_dailystats.csv.gz'
sd_scandi = f'{path}SNOWCAP_snowdepth_scandi_valid_0-10000_20200101-20201231_dailystats.csv.gz'
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
era5l_scandi = 'ERA5L_scandi_valid_10000.csv.gz'
era5l_scandi_ext = 'ERA5L_new_scandi_valid_10000.csv.gz'
era5ld_scandi = 'ERA5LD_tmax_era5ld_scandi_valid_0-10000_20150831-20230901_dailymax.csv.gz'
df_era5l_scandi = pd.read_csv(f'{path}{era5l_scandi}', dtype={'latitude': str, 'longitude': str})
df_era5l_scandi['utctime'] = pd.to_datetime(df_era5l_scandi['utctime'])
df_merged_scandi = pd.merge(df_merged_scandi, df_era5l_scandi, on=['utctime', 'latitude', 'longitude'], how='inner')
df_era5ld_scandi = pd.read_csv(f'{path}{era5ld_scandi}', dtype={'latitude': str, 'longitude': str})
df_era5ld_scandi['utctime'] = pd.to_datetime(df_era5ld_scandi['utctime'])
df_merged_scandi = pd.merge(df_merged_scandi, df_era5ld_scandi, on=['utctime', 'latitude', 'longitude'], how='inner')
df_era5l_scandi_ext = pd.read_csv(f'{path}{era5l_scandi_ext}', dtype={'latitude': str, 'longitude': str})
df_era5l_scandi_ext['utctime'] = pd.to_datetime(df_era5l_scandi_ext['utctime'])
df_merged_scandi = pd.merge(df_merged_scandi, df_era5l_scandi_ext, on=['utctime', 'latitude', 'longitude'], how='inner')
print(df_merged_scandi)

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
hkl_input_alps=f'{path}HKL_SWE_alps_valid_0-10000_20160901-20240901.csv.gz'
df_hkl_alps = pd.read_csv(hkl_input_alps, dtype={'latitude': str, 'longitude': str})
df_hkl_alps = df_hkl_alps.dropna(subset=['SWE'])
print(df_hkl_alps)

# DEM for Snowcap locations
dem_input_alps =f'{path}validation_dem+fch_merged_fin2_10000_fscs1_alps.csv.gz'
df_dem_alps = pd.read_csv(dem_input_alps, dtype={'latitude': str, 'longitude': str})
df_dem_alps = df_dem_alps.drop(columns=['dist_to_train_m'], errors='ignore')

# merge the two dataframes on latitude and longitude
df_merged_alps = pd.merge(df_hkl_alps, df_dem_alps, on=['latitude', 'longitude'], how='inner')
df_merged_alps['tag'] = 'Alps'
df_merged_alps['utctime'] = pd.to_datetime(df_merged_alps['utctime'])
df_merged_alps['day'] = df_merged_alps['utctime'].dt.day
df_merged_alps['month'] = df_merged_alps['utctime'].dt.month
df_merged_alps['year'] = df_merged_alps['utctime'].dt.year
print(df_merged_alps)

# TCF tree cover fraction (years > 2021 reuse the 2021 values)
tcf_alps = f'{path}snow_alps_tcf_valid.txt'
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
swe_alps = f'{path}SNOWCAP_swe_alps_valid_0-10000_20200101-20201231_dailystats.csv.gz'
sd_alps = f'{path}SNOWCAP_snowdepth_alps_valid_0-10000_20200101-20201231_dailystats.csv.gz'
df_swe_alps = pd.read_csv(swe_alps, dtype={'latitude': str, 'longitude': str})
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
era5l_alps = 'ERA5L_alps_valid_10000.csv.gz'
era5l_alps_ext = 'ERA5L_new_alps_valid_10000.csv.gz'
era5ld_alps = 'ERA5LD_tmax_era5ld_alps_valid_0-10000_20160831-20240901_dailymax.csv.gz'

df_era5l_alps = pd.read_csv(f'{path}{era5l_alps}', dtype={'latitude': str, 'longitude': str})
df_era5l_alps['utctime'] = pd.to_datetime(df_era5l_alps['utctime'])
df_merged_alps = pd.merge(df_merged_alps, df_era5l_alps, on=['utctime', 'latitude', 'longitude'], how='inner')
df_era5l_alps_ext = pd.read_csv(f'{path}{era5l_alps_ext}', dtype={'latitude': str, 'longitude': str})
df_era5l_alps_ext['utctime'] = pd.to_datetime(df_era5l_alps_ext['utctime'])
df_merged_alps = pd.merge(df_merged_alps, df_era5l_alps_ext, on=['utctime', 'latitude', 'longitude'], how='inner')
df_era5ld_alps = pd.read_csv(f'{path}{era5ld_alps}', dtype={'latitude': str, 'longitude': str})
df_era5ld_alps['utctime'] = pd.to_datetime(df_era5ld_alps['utctime'])
df_merged_alps = pd.merge(df_merged_alps, df_era5ld_alps, on=['utctime', 'latitude', 'longitude'], how='inner')
print(df_merged_alps)

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

out_all = f'{path}validation_all_scandi_alps_ext.csv'
df_all.to_csv(out_all, index=False)
print('saved to', out_all)