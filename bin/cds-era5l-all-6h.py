import cdsapi,sys

year=sys.argv[1]
month=sys.argv[2]

dataset = "reanalysis-era5-land"
request = {
    "variable": [
        "2m_dewpoint_temperature",
        "2m_temperature",
        "skin_temperature",
        "soil_temperature_level_1",
        "soil_temperature_level_2",
        "soil_temperature_level_3",
        "soil_temperature_level_4",
        "lake_bottom_temperature",
        "lake_ice_depth",
        "lake_ice_temperature",
        "lake_mix_layer_depth",
        "lake_mix_layer_temperature",
        "lake_shape_factor",
        "lake_total_layer_temperature",
        "snow_albedo",
        "snow_cover",
        "snow_density",
        "snow_depth",
        "snow_depth_water_equivalent",
        "snowfall",
        "snowmelt",
        "temperature_of_snow_layer",
        "skin_reservoir_content",
        "volumetric_soil_water_layer_1",
        "volumetric_soil_water_layer_2",
        "volumetric_soil_water_layer_3",
        "volumetric_soil_water_layer_4",
        "forecast_albedo",
        "surface_latent_heat_flux",
        "surface_net_solar_radiation",
        "surface_net_thermal_radiation",
        "surface_sensible_heat_flux",
        "surface_solar_radiation_downwards",
        "surface_thermal_radiation_downwards",
        "evaporation_from_bare_soil",
        "evaporation_from_open_water_surfaces_excluding_oceans",
        "evaporation_from_the_top_of_canopy",
        "evaporation_from_vegetation_transpiration",
        "potential_evaporation",
        "runoff",
        "snow_evaporation",
        "sub_surface_runoff",
        "surface_runoff",
        "total_evaporation",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "surface_pressure",
        "total_precipitation",
        "leaf_area_index_high_vegetation",
        "leaf_area_index_low_vegetation"
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
    "time": [
        "00:00", "06:00", "12:00",
        "18:00"
    ],
    "data_format": "grib",
    "download_format": "unarchived",
    "area": [83, -33, 28, 55]
}

client = cdsapi.Client()
out_file=f'/home/ubuntu/data/grib/ERA5L_19950101T000000_{year}{month}_all_6h.grib'
client.retrieve(dataset, request).download(out_file)
