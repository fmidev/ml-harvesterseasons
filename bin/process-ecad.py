import pandas as pd
import sys

souid_user = sys.argv[1]
basepath = '/home/ubuntu/data/ML/training-data/precipitation/ecad/'
stations_path = basepath + 'RR/'
processed_path = basepath + 'RR_processed/'

# NEW: folder for full-period datasets only
full_period_path = basepath + 'RR_processed_full_period/'

# ADDED: folder for stations with at least 18 years of data
eighteen_yr_path = basepath + 'RR_processed_18yr/'

station = f'RR_SOUID{souid_user}.txt'

def dms_to_decimal(dms):
    s = dms.strip()
    sign = -1 if s.startswith("-") else 1
    deg, mins, secs = s.lstrip("+-").split(":")
    return sign * (int(deg) + int(mins) / 60 + int(secs) / 3600)


def find_header(path):
    with open(path, encoding="latin-1") as f:
        for i, line in enumerate(f):
            if line.lstrip().startswith("STAID"):
                return i


def read_ecad(path):
    return pd.read_csv(
        path,
        skiprows=find_header(path),
        skipinitialspace=True,
        parse_dates=["DATE"],
        na_values=-9999,                # documented missing-value sentinel
        encoding="latin-1",
    )


def read_stations(path):
    stations = pd.read_csv(
        path,
        skiprows=find_header(path),
        skipinitialspace=True,
        na_values=[-9999, -999],        # sentinel values seen in real ECA&D files
        encoding="latin-1",
    )
    stations.columns = stations.columns.str.strip()

    for col in ["STANAME", "CN"]:
        stations[col] = stations[col].astype(str).str.strip()

        # Strip non-printable & C1 control bytes
        stations[col] = stations[col].str.replace(
            r"[\x00-\x1f\x7f-\x9f]",
            "",
            regex=True
        )

    stations["LAT"] = stations["LAT"].apply(dms_to_decimal)
    stations["LON"] = stations["LON"].apply(dms_to_decimal)

    return stations


# Load the precipitation data and the station metadata
df = read_ecad(stations_path + station)
stations = read_stations(basepath + 'stations_rr.txt')

# Mark RR as NaN for unexpected Q_RR values
# (spec says only 0, 1, 9 are valid codes)
df.loc[~df["Q_RR"].isin([0, 1, 9]), "RR"] = pd.NA

# Mark RR as NaN whenever the quality flag isn't "valid"
# (Q_RR: 0=valid, 1=suspect, 9=missing)
df.loc[df["Q_RR"] != 0, "RR"] = pd.NA

# Mark negative RR as NaN (precipitation can't be negative)
df.loc[df["RR"] < 0, "RR"] = pd.NA

# Merge metadata into df on STAID
df = df.merge(stations, on="STAID", how="left")

# ML area is 83N to 33N, 30W to 50E (ERA5L data availability)
df = df[
    (df["LAT"] <= 83) &
    (df["LAT"] >= 33) &
    (df["LON"] >= -30) &
    (df["LON"] <= 50)
]

# Filter to dates between 1995-01-01 and 2025-12-31
df = df[
    (df["DATE"] >= "1995-01-01") &
    (df["DATE"] <= "2025-12-31")
]

# Drop rows with missing precipitation values
df = df.dropna(subset=["RR"])

# Drop rows with missing station metadata
df = df.dropna(subset=["HGHT", "LAT", "LON"])

# Convert precipitation from 0.1 mm to mm
df["RR"] = df["RR"] / 10

# Reset index after filtering
df = df.reset_index(drop=True)

# Make sure datatype is uniform for later merging datasets
# change lat lon to str
df["STAID"] = df["STAID"].astype(str)
df["SOUID"] = df["SOUID"].astype(str)
df["LAT"] = df["LAT"].astype(str)
df["LON"] = df["LON"].astype(str)
df["HGHT"] = df["HGHT"].astype(int)

# Skip saving if dataframe is empty after filtering
if df.empty:
    print(f"No data for SOUID {souid_user} after filtering — nothing saved.")
    sys.exit(0)

countrycode=df["CN"].iloc[0]
print(countrycode)

station_out = f'{processed_path}RR_processed_SOUID{souid_user}_{countrycode}.csv'
station_out_full = f'{full_period_path}RR_processed_full_period_SOUID{souid_user}_{countrycode}.csv'

# Save processed data
df.to_csv(station_out, index=False)
print(f"Saved {len(df)} rows to {station_out}")

# ------------------------------------------------------------------
# Check only YEAR availability (1995-2025)
# ------------------------------------------------------------------

available_years = set(df["DATE"].dt.year.unique())

required_years = set(range(1995, 2026))

has_full_period = required_years.issubset(available_years)

if has_full_period:
    df.to_csv(station_out_full, index=False)
    print(f"Full-period dataset saved to {station_out_full}")
else:
    print(
        f"SOUID {souid_user} does not contain all years "
        f"from 1995-2025 — full-period file not saved."
    )

# ------------------------------------------------------------------
# ADDED: Save if station has at least 18 distinct years of data
# (doesn't need to be consecutive years or full years)
# ------------------------------------------------------------------

if len(available_years) >= 18:
    station_out_18yr = f'{eighteen_yr_path}RR_processed_18yr_SOUID{souid_user}_{countrycode}.csv'
    df.to_csv(station_out_18yr, index=False)
    print(f"18-year dataset ({len(available_years)} years) saved to {station_out_18yr}")
else:
    print(
        f"SOUID {souid_user} has only {len(available_years)} years "
        f"of data — 18-year file not saved."
    )

print(df)