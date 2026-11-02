import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

basepath = '/home/ubuntu/data/ML/training-data/precipitation/ecad/'

stations = f'{basepath}RR_processed_all.csv'
stations_filter = f'{basepath}RR_processed_filtered_FINAL.csv' 

# Extreme event thresholds (mm/day)
THRESHOLDS = [50, 100, 200]


def process_dataset(csv_path, label):
    output_lonlat = f'{basepath}ecad_lonlat_{label}.txt'
    output_stations = f'{basepath}ecad_stations_{label}.csv'
    output_stats = f'{basepath}ecad_statistics_{label}.md'
    output_map_png = f'{basepath}ecad_map_{label}.png'
    output_map_svg = f'{basepath}ecad_map_{label}.svg'

    print(f"\n========== {label} ==========")
    df = pd.read_csv(csv_path)
    print(df)

    # Extract unique LON/LAT pairs and save with space-separated values
    unique_locs = df[["LON", "LAT"]].drop_duplicates().reset_index(drop=True)
    unique_locs.to_csv(output_lonlat, sep=" ", index=False)
    print(f"\nSaved {len(unique_locs)} unique LON/LAT pairs to {output_lonlat}")

    # Extract unique STAID/SOUID/LAT/LON combinations and save as CSV
    unique_stations = (
        df[["STAID", "SOUID", "LAT", "LON"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    unique_stations.to_csv(output_stations, index=False)
    print(f"Saved {len(unique_stations)} unique stations to {output_stations}")

    # --- Statistics by country ---

    # Build a one-row-per-station dataframe (no duplicate STAIDs)
    stations_unique = df[["STAID", "CN", "HGHT"]].drop_duplicates(subset=["STAID"]).reset_index(drop=True)
    print(f"\nTotal unique physical stations (STAID): {len(stations_unique)}")

    # Station count and height stats from unique stations only
    country_stations = stations_unique.groupby("CN").agg(
        n_stations=("STAID", "count"),
        min_hght=("HGHT", "min"),
        max_hght=("HGHT", "max"),
        mean_hght=("HGHT", "mean"),
    ).round(1)

    # RR stats from the full daily timeseries (one row per measurement)
    country_rr = df.groupby("CN")["RR"].agg(["min", "max", "mean"]).round(2)

    # Combine the basic stats first
    stats = country_stations.join(country_rr)

    # Build extreme-event columns for each threshold
    extreme_cols = []
    for t in THRESHOLDS:
        days = df[df["RR"] >= t].groupby("CN").size().rename(f"days_{t}mm")
        sta = df[df["RR"] >= t].groupby("CN")["STAID"].nunique().rename(f"sta_{t}mm")
        stats = stats.join(days).join(sta)
        stats[f"days_{t}mm"] = stats[f"days_{t}mm"].fillna(0).astype(int)
        stats[f"sta_{t}mm"] = stats[f"sta_{t}mm"].fillna(0).astype(int)
        extreme_cols += [f"days_{t}mm", f"sta_{t}mm"]

    # Reorder columns
    stats = stats[[
        "n_stations",
        "min", "max", "mean",
        "min_hght", "max_hght", "mean_hght",
        *extreme_cols,
    ]]

    # Build column labels
    base_labels = [
        "Stations",
        "Min RR (mm)",
        "Max RR (mm)",
        "Mean RR (mm)",
        "Min HGHT (m)",
        "Max HGHT (m)",
        "Mean HGHT (m)",
    ]
    extreme_labels = []
    for t in THRESHOLDS:
        extreme_labels += [f"Days ≥{t} mm", f"Stations ≥{t} mm"]
    stats.columns = base_labels + extreme_labels
    stats.index.name = "Country"

    # Sort by station count descending
    stats = stats.sort_values("Stations", ascending=False)

    # Write markdown file
    with open(output_stats, "w") as f:
        f.write(f"# ECA&D precipitation statistics by country — {label}\n\n")
        f.write(f"Total unique stations (STAID): **{len(stations_unique)}**\n\n")
        f.write(f"Total daily measurements: **{len(df):,}**\n\n")
        thresh_str = ", ".join(f"**≥{t} mm/day**" for t in THRESHOLDS)
        f.write(f"Extreme-event thresholds: {thresh_str}\n\n")
        f.write(stats.to_markdown())
        f.write("\n")

    print(f"\nSaved statistics to {output_stats}")
    print(stats)

    # --- Map of all unique stations ---

    pad = 2
    lon_min, lon_max = unique_locs["LON"].min() - pad, unique_locs["LON"].max() + pad
    lat_min, lat_max = unique_locs["LAT"].min() - pad, unique_locs["LAT"].max() + pad

    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="#f5f5f0", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#cfe2f3", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)

    ax.scatter(
        unique_locs["LON"], unique_locs["LAT"],
        s=4, color="crimson", alpha=0.6,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )

    ax.set_title(f"ECA&D precipitation stations — {label} ({len(unique_locs)} unique locations)")

    plt.savefig(output_map_png, dpi=200, bbox_inches="tight")
    plt.savefig(output_map_svg, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved map to {output_map_png} and {output_map_svg}")


#process_dataset(stations, "all")
process_dataset(stations_filter, "filtered_FINAL")