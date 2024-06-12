# Training a model to forecast soil water index (SWI) with gradient boosting

This code reproduces the data and model training and prediction workflows used in `Strahlendorff et al.: Gradient boosting-based soil wetness for forestry climate adaptation in HarvesterSeasons service -training a model to forecast soil water index from a comprehensive set of IFS model predictors in Destination Earth`  

## System requirements

## Dependencies

xgb conda environment
screen adviced

## Downloading the predictor and predictand (target) data
For training the model you will need a table of all predictors and predictand in all chosen locations for the whole time period as input. We have several time series scripts in Python that use the request module to make http-requests to our SmartMet server (https://desm.harvesterseasons.com/grid-gui) Time Series API (https://github.com/fmidev/smartmet-plugin-timeseries). Use these scripts to get daily time series for all LUCAS locations from ERA5-Land, SWI, climatology for SWI, and the Leaf Area Index climatology for each day from 2015 to 2022. In addition, static variables, such as different land covers or inland water fractions, must be prepared as time series data. To run the time series (ts) scripts, you will need a csv file with LUCAS point-ids, and corresponding latitudes and longitudes. All the ts scripts need `functions.py` with functions for the time series queries etc. Output is a csv file for each location. 

To download the predictand SWI2 data, run the `get-swi-ts-all.py`. You can fetch data for up to 5000 points per query, info of locations stored in `llpdict` (change in the script to change locations). The time series query for SWI target parameters also replaces some of the missing values with linearly interpolated values using the two nearest values within a 4-day time interval with `interpolate_t`. The resulting time series per location are saved as csv files.  

To download the ERA5-Land predictor data, run the `get-era5l-ts-all.py`. It fetches the 24h accumulated, 00 and 12 UTC hourly, and 5-/15-/60-/100-day rolling cumulative daily sums time series data and saves them per location and predictor as csv files.

To download SWI2 climatology predictor data, run `get-swi-clim-ts.py`. 

To download static predictors such as soil type, run `get-ECC-static.py`. 

To download Copernicus DEM predictors, run `get-copernicus-ts.py`. 



## Training the model
The training scripts include one for performing the Optuna hyperparameter tuning runs and another229
for rendering with the best tuning settings: xgb-fit-optuna-swi2.py and xgb-fit-swi2.py. The same230
data are available for the lightGBM prefixed as the lgbm fit. These scripts are very similar to each231
other because both XGBoost and LightGBM are methods that are integrated with scikit-learn;232
therefore, training data are used from the same file, and the main differences in the scripts are the233
function calls and hyperparameter attributes that differ for the two tested methods. In addition to the234
fitting scripts, there are also Python scripts for K-fold analysis and cross-correlation, and scripts to235
plot figures for the location maps and feature importance.236

## Predicting soil water index
Predicting SWI2 requires two steps: The first experiment was performed as part of the get season. sh237
and get-edte.sh bash scripts. All input data must be re-gridded into the same grid in which the result238
is in. For seasonal forecasts it is the ERA5-Land grid for the European area from -30 degrees west to239
50 degrees east and 25 degrees north to 75 degrees north at 0.1 degrees increments. The EDTE was a240
0.04 °grid for the same bounds. These steps use the GNU parallel (Tange, 2021) and CDO by241
(Schulzweida, 2023). The second step was performed using our self-developed xgb-predict-swi2-242
era5l.py and -edte.py scripts. These Python scripts use Xarray (Hoyer, 2017) to join different input243
grids into one data frame that includes all time steps for each input in the target grid. This data frame244
was used by XGBoost to calculate the grid using the predicted SWI2
