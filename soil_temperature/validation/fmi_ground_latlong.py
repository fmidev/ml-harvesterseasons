import numpy as np
import pandas as pd

fmi_ground_obs_path = "/home/ubuntu/data/ML/validation-data/soiltemp/fmi_ground_temp_3hr/st-tg-3h-fmisids-202310-202411.csv"
save_path = "/home/ubuntu/data/ML/validation-data/soiltemp/fmi_ground_temp_3hr/fmi_tg_latlong.csv"

fmi_ground_obs = pd.read_csv(fmi_ground_obs_path, on_bad_lines="skip")
fmi_ground_obs.rename(columns={"lat": "Latitude", "lon": "Longitude"}, inplace=True)
fmi_ground_obs = fmi_ground_obs.dropna(subset=["Latitude", "Longitude"])
# Drop duplicate rows based on Latitude and Longitude
lat_long_unique = fmi_ground_obs.drop_duplicates(
    subset=["Latitude", "Longitude", "fmisid"]
)
lat_long_unique = lat_long_unique[["Latitude", "Longitude"]]
lat_long_unique.to_csv(
    save_path,
    index=False,
)
