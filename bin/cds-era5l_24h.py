import cdsapi,sys

year=sys.argv[1]
month=sys.argv[2]

dataset = "reanalysis-era5-land"
request = {
    "variable": [
        "surface_latent_heat_flux",
        "surface_net_solar_radiation",
        "surface_net_thermal_radiation",
        "surface_sensible_heat_flux",
        "surface_solar_radiation_downwards",
        "surface_thermal_radiation_downwards",
        "potential_evaporation",
        "runoff",
        "total_evaporation",
        "total_precipitation"
    ],
    "year": year,
    "month": month,
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": ["00:00"],
    "data_format": "grib",
    "download_format": "unarchived",
    "area": [83, -33, 28, 55]
}

client = cdsapi.Client()
outfile=f'/home/ubuntu/data/grib/ERA5L_19950101T000000_{year}{month}_24h.grib'
client.retrieve(dataset, request).download(outfile)
