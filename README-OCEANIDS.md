## Training locations 

Helsinki Vuosaari harbor: FMISID 151028 latitude 60.20867 longitude 25.2959

### Training latitudes
Closest to harbor ERA5 grid points in Table 1.
|lat name|lon name|lat deg|lon deg|
|:-:|:-:|:--:|:--:|
|lat-1 |lon-1 |60.00 |25.00|
lat-2 |lon-2 |60.25 |25.25|
lat-3 |lon-3 |60.25 |25.00|
lat-4 |lon-4 |60.00| 25.25|

Table 1 Training locations and their ML names. 

![Training locations (blue) and Helsinki Vuosaari harbor measuring site (red)](HelsinkiVuosaariHarbor151028-3.jpg)
Figure 1 Training locations and Helsinki Vuosaari harbor measuring site.
## Predictand data

| Predictand | Units | Producer |  ML name |
| ------------- |---|:-------------:|-:|
| Average wind speed, 1 hour| m/s |  |WS_PT1H_AVG |
|Maximum wind speed, 1 hour |m/s||WG_PT1H_MAX|

## Predictors for fitting the ML model

| Predictor | Units | Producer | Spatial resolution | Temporal resolution | ML name |
| ------------- |---|:-------------:| --:|-:|-:|
| 10m u-component of wind | m/s |ERA5|0.25° x 0.25°|hourly data, 3h steps starting 00 UTC|u10|
| 10m v-component of wind | m/s |ERA5|0.25° x 0.25°||v10|
| 10m wind gust since previous post-processing  | m/s |ERA5|0.25° x 0.25°||fg10|
|2m dewpoint temperature|K|ERA5|0.25° x 0.25°||td2|
|2m temperature|K|ERA5|0.25° x 0.25°||t2|
|Eastward turbulent surface stress|N m-2 s|ERA5|0.25° x 0.25°||ewss|
|Evaporation|m of water equivalent|ERA5|0.25° x 0.25°||e|
|Land-sea mask|-|ERA5|0.25° x 0.25°||lsm|
|Mean sea level pressure|Pa|ERA5|0.25° x 0.25°||msl|
|Northward turbulent surface stress|N m-2 s|ERA5|0.25° x 0.25°||nsss|
|Sea surface temperature|K|ERA5|0.25° x 0.25°||tsea|
|Surface latent heat flux|W m-2|ERA5|0.25° x 0.25°||slhf|
|Surface net solar radiation|W m-2|ERA5|0.25° x 0.25°||ssr|
|Surface net thermal radiation|W m-2|ERA5|0.25° x 0.25°||str|
|Surface sensible heat flux|W m-2|ERA5|0.25° x 0.25°||sshf|
|Surface solar radiation downwards|W m-2|ERA5|0.25° x 0.25°||ssrd|
|Surface thermal radiation downwards|W m-2|ERA5|0.25° x 0.25°||strd|
|Total cloud cover|0 to 1|ERA5|0.25° x 0.25°||tcc|
|Total column cloud liquid water|kg m-2|ERA5|0.25° x 0.25°||tclw|
|Total precipitation|m|ERA5|0.25° x 0.25°||tp|

## Predictors for predicting with seasonal forecast

| Predictor | Units | Producer | Spatial resolution | Temporal resolution | ML name |
| ------------- |---|:-------------:| --:|-:|-:|
| 10m u-component of wind | m/s |||6h instantaneous|u10|
| 10m v-component of wind | m/s |||6h instantaneous|v10|
| 10m wind gust since previous post-processing  | m/s |||24h aggregation|fg10|
|2m dewpoint temperature|K|||6h instantaneous|td2|
|2m temperature|K|||6h instantaneous|t2|
|Eastward turbulent surface stress|N m-2 s|||24h aggregation since beginning of forecast|ewss|
|Evaporation|m of water equivalent|||24h aggregation since beginning of forecast|e|
|Land-sea mask|-|||	Static|lsm|
|Mean sea level pressure|Pa|||6h instantaneous|msl|
|Northward turbulent surface stress|N m-2 s|||24h aggregation since beginning of forecast|nsss|
|Sea surface temperature|K|||6h instantaneous|tsea|
|Surface latent heat flux|W m-2|||24h aggregation since beginning of forecast|slhf|
|Surface net solar radiation|W m-2|||24h aggregation since beginning of forecast|ssr|
|Surface net thermal radiation|W m-2|||24h aggregation since beginning of forecast|str|
|Surface sensible heat flux|W m-2|||24h aggregation since beginning of forecast|sshf|
|Surface solar radiation downwards|W m-2|||24h aggregation since beginning of forecast|ssrd|
|Surface thermal radiation downwards|W m-2|||24h aggregation since beginning of forecast|strd|
|Total cloud cover|0 to 1|||6h instantaneous|tcc|
|Total column cloud liquid water|kg m-2|||24h instantaneous|tlwc|
|Total precipitation|m|||24h aggregation since beginning of forecast|tp|

## KFold
Fold: 1 RMSE: 1.02
Train:  [2013 2015 2016 2017 2018 2019 2022 2023] Test:  [2014 2020 2021]

Fold: 2 RMSE: 0.99
Train:  [2014 2015 2016 2017 2019 2020 2021 2022 2023] Test:  [2013 2018]

Fold: 3 RMSE: 1.03
Train:  [2013 2014 2016 2017 2018 2020 2021 2022 2023] Test:  [2015 2019]

Fold: 4 RMSE: 1.01
Train:  [2013 2014 2015 2016 2018 2019 2020 2021 2022] Test:  [2017 2023]

Fold: 5 RMSE: 0.98
Train:  [2013 2014 2015 2017 2018 2019 2020 2021 2023] Test:  [2016 2022]

Best fold: 5