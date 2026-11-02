#!/usr/bin/env python3

import os
import pandas as pd

# ============================================================
# ERA5L request printer for IBA observation locations
# Prints SmartMet timeseries API requests only
# (does NOT download data)
# ============================================================

iba_dir = '/home/smartmet/copernicus/IBAML'
era5l_dir = '/home/smartmet/copernicus/IBAML/era5l/mlsmart/'

source = 'http://ml.harvesterseasons.com:8080'

# ------------------------------------------------------------
# Read observation locations
# ------------------------------------------------------------

obs_file = os.path.join(
    iba_dir,
    'iba_observations_2025_processed.csv'
)

df_obs = pd.read_csv(obs_file,dtype={'latlon_id': str}
)

# Build dict:
# latlon_id -> (lat, lon)

obsloc_dict = {}

for _, row in df_obs.iterrows():

    obsloc_dict[row['latlon_id']] = (
        row['latitude'],
        row['longitude']
    )

# ------------------------------------------------------------
# Batch setup
# ------------------------------------------------------------

locations = list(obsloc_dict.values())

batch_size = 2000

n_batches = (
    len(locations) + batch_size - 1
) // batch_size

print(
    f"Total {len(locations)} locations "
    f"-> {n_batches} batches "
    f"of up to {batch_size}"
)

# ------------------------------------------------------------
# ERA5L instant features
# ------------------------------------------------------------

era5l_features = {
    #'t2': 'T2-K:ERA5L:5078:1:0:1:0',
    #'td2': 'TD2-K:ERA5L:5078:1:0:1:0',
    #'laihv_era5l': 'LAI_HV-M2M2:ERA5L:5078:1:0:1:0',
    'lailv_era5l': 'LAI_LV-M2M2:ERA5L:5078:1:0:1:0',
    #'sp': 'PGR-PA:ERA5L:5078:1:0:1:0',
    #'sd': 'SD-M:ERA5L:5078:1:0:1:0',
    #'rsn': 'SND-KGM3:ERA5L:5078:1:0:1:0',
    #'swvl1': 'SOILWET-M3M3:ERA5L:5078:9:7:1:0',
    #'swvl2': 'SWVL2-M3M3:ERA5L:5078:9:1820:1:0',
    #'swvl3': 'SWVL3-M3M3:ERA5L:5078:9:7268:1:0',
    #'swvl4': 'SWVL4-M3M3:ERA5L:5078:9:25855:1:0',
    #'stl1': 'STL1-K:ERA5L:5078:9:7:1:0',
    #'stl2': 'TSOIL-K:ERA5L:5078:9:1820:1:0',
    #'stl3': 'STL3-K:ERA5L:5078:9:7268:1:0',
    #'stl4': 'STL4-K:ERA5L:5078:9:25855:1:0',
    #'u10': 'U10-MS:ERA5L:5078:1:0:1:0',
    #'v10': 'V10-MS:ERA5L:5078:1:0:1:0',
    #'skt': 'SKT-K:ERA5L:5078:1:0:1:0',
    #'src': 'SRC-M:ERA5L:5078:1:0:1:0',
}

# ------------------------------------------------------------
# Time settings
# ------------------------------------------------------------

hours_list = [
    f"{i:02d}" for i in range(24)
]

hours = ",".join(hours_list)

start_inst = '20250101T000000'
end_inst   = '20251101T000000'

# ============================================================
# PRINT REQUESTS
# ============================================================

for batch_idx in range(n_batches):

    batch_num = batch_idx + 1

    batch_locs = locations[
        batch_idx * batch_size:
        (batch_idx + 1) * batch_size
    ]

    coords_flat = [
        str(v)
        for pair in batch_locs
        for v in pair
    ]

    latlons_param = ",".join(coords_flat)

    print()
    print("=" * 80)
    print(
        f"BATCH {batch_num}/{n_batches} "
        f"({len(batch_locs)} locations)"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Instant feature requests
    # --------------------------------------------------------

    for feat, fmikey in era5l_features.items():

        out = os.path.join(
            era5l_dir,
            f'era5l_{feat}_2025_all-{batch_num}.csv'
        )

        print()
        print("-" * 80)
        print(f"FEATURE: {feat}")
        print(f"OUTPUT : {out}")
        print("-" * 80)

        q1 = (
            f"{source}/timeseries"
            f"?latlons={latlons_param}"
            f"&param=time,latitude,longitude,{fmikey}"
            f"&starttime={start_inst}Z"
            f"&endtime={end_inst}Z"
            f"&hour={hours}"
            "&format=json"
            "&precision=full"
            "&timeformat=sql"
        )

        print(q1)