### Training

#### data
- Get time series features
```sh
python get_ts_latest.py
```
- Get Tif features
```sh
python get_features_tif.py
```

#### Hyper parameter tuning
- Optuna

```sh
# xgb
python xgb_fit_optuna_soil_rmse_latest.py
# lgbm
python lgbm_fit_optuna_soiltemp_latest.py
```

#### Training

```sh
# xgb
python xgb-fit-soil-latest.py
```
