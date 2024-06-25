# Training a model to forecast soil water index (SWI) with gradient boosting

This code reproduces the data and model training and prediction workflows used in `Strahlendorff et al.: Gradient boosting-based soil wetness for forestry climate adaptation in HarvesterSeasons service -training a model to forecast soil water index from a comprehensive set of IFS model predictors in Destination Earth`

Note: as of June 2024 this README is still a work in process, please be patient and we will get it ready asap :) 

## System requirements
Python version 3.10.13 in the UNIX/Linux environment was used in this project.

Running the model training takes XX hours with XX CPU cores and XX memory, for 10 000 locations. Prediction takes... 
desm ja ml pred ja train req
## Dependencies
To create xgb environment used in this project, check out the `requirements.txt` file.

To download the seasonal forecast data etc from the Climate Data Store, the CDS API client needs to be installed https://cds.climate.copernicus.eu/api-how-to.

optuna
xgb conda environment
screen adviced
cdo
parallel Tange, O., 2018. GNU Parallel 2018. Available at: https://doi.org/10.5281/zenodo.1146014
## Downloading the predictor and predictand (target) data
For training the model you will need a table of all predictors and predictand in all chosen locations for the whole time period as input. We have several time series scripts in Python that use the request module to make http-requests to our SmartMet server (https://desm.harvesterseasons.com/grid-gui) Time Series API (https://github.com/fmidev/smartmet-plugin-timeseries). Use these scripts to get daily time series for all LUCAS locations from ERA5-Land, SWI, climatology for SWI, and the Leaf Area Index climatology for each day from 2015 to 2022. In addition, static variables, such as different land covers or inland water fractions, must be prepared as time series data. To run the time series (ts) scripts, you will need a csv file with LUCAS point-ids, and corresponding latitudes and longitudes. All the ts scripts need `functions.py` with functions for the time series queries etc. You can fetch data for up to 5000 points per query. Output is a csv file for each location. Check the directory structures defined in the scripts. 

To download the predictand SWI2 data, run the `get-swi-ts-all-FIN.py`. The time series query for SWI target parameters also replaces some of the missing values with linearly interpolated values using the two nearest values within a 4-day time interval with `interpolate_t`. The resulting time series per location are saved as csv files.  

To download the ERA5-Land predictor data, run the `get-era5l-ts-all.py`. It fetches the 24h accumulated, 00 and 12 UTC hourly, and 5-/15-/60-/100-day rolling cumulative daily sums time series data and saves them per location and predictor as csv files.

To download SWI2 climatology predictor data, run `get-swi-clim-ts.py`. 

To download static predictors such as soil type, run `get-ECC-static.py`. 

To download Copernicus DEM predictors, run `get-copernicus-ts.py`. 

To plot the LUCAS locations on map (whole set or subset), run `plot-latlons-on-map.py`. You will need a NUTS_RG_20M_2021_4326.json file for the background map. Tää skripti ja moni muu pitää siivota (ja testata) 

lisää lucas tiedosto jakeluun tai silt etc mistä sai ohje
lisää lista meidän mallissa käytetyistä point-idstä jakeluun
## Training the model
You´ll need to combine all the predictors and predictand data as one input csv for the training scripts (all chosen locations, full time series). Note that static predictors need to be repeated daily. The first row of the input table should be column names, all parameters (not all were used in the final trained model): 

`utctime,swi2,evap,evap5d,evap15d,evap60d,evap100d,evapp,evapp5d,evapp15d,evapp60d,evapp100d,laihv-00,laihv-12,lailv-00,lailv-12,ro,ro5d,ro15d,ro60d,ro100d,rsn-00,rsn-12,sd-00,sd-12,sf,skt-00,skt-12,slhf,sro,sro5d,sro15d,sro60d,sro100d,sshf,ssr,ssrd,ssro,ssro5d,ssro15d,ssro60d,ssro100d,stl1-00,stl1-12,str,strd,swvl1-00,swvl1-12,swvl2-00,swvl2-12,swvl3-00,swvl3-12,swvl4-00,swvl4-12,t2-00,t2-12,td2-00,td2-12,tp,tp5d,tp15d,tp60d,tp100d,u10-00,u10-12,v10-00,v10-12,swi2clim,lake_cover,cvh,cvl,lake_depth,land_cover,soiltype,urban_cover,tvh,tvl,POINT_ID,TH_LAT,TH_LONG,DTM_height,DTM_slope,DTM_aspect,TCD,WAW,CorineLC,clay_0-5cm,clay_100-200cm,clay_15-30cm,clay_30-60cm,clay_5-15cm,clay_60-100cm,sand_0-5cm,sand_100-200cm,sand_15-30cm,sand_30-60cm,sand_5-15cm,sand_60-100cm,silt_0-5cm,silt_100-200cm,silt_15-30cm,silt_30-60cm,silt_5-15cm,silt_60-100cm,soc_0-5cm,soc_100-200cm,soc_15-30cm,soc_30-60cm,soc_5-15cm,soc_60-100cm`

To perform the Optuna hyperparameter tuning (https://optuna.org/), run `xgb-fit-optuna-swi2.py fname` where `fname` is the name of your input csv dataset file. Check the results on your Optuna Dashboard view. 

To train the model with tuned hyperparameters, run `xgb-fit-swi2.py fname` where `fname` is the name of your input csv dataset file. The fitted model is saved as txt file and RMSE/MAE is printed to terminal. 
Laita käytetty malli jakeluun? Kun joka treenauksella ei tismalleen sama malli? 

To perform the K-Fold cross-validation (split input dataset to optimal training and testing sets by years), run `xgb-fit-KFold-swi2.py`.  

To create the cross-correlation matrix and correlation bar chart figures, run `cross-correlation-swi2.py`.

To create the F-score (feature importance) figure, run `xgb-analysis-soilwater.py`. This needs as input the trained model.

## Predicting soil water index
nää skriptit on desm koneella eli ei ml puolella jaossa mutta piti muutenkin tehdä artikkelia varten "muokkaamattomat" final versiot jakoon niin voi iskeä desmille  

Predicting SWI2 requires first downlnoading the data and pre-processing as all input data must be re-gridded. `get-seasonal.sh` (ei ole ml-koneella jaossa) downloads (latest or user-specified start month+year) and remaps the seasonal forecast (https://cds.climate.copernicus.eu/cdsapp#!/dataset/seasonal-original-single-levels?tab=overview) and other necessary data to ERA5-Land grid for the European area. The script performs also statistical bias-adjusting and downscaling for several parameters. Preprocessing uses the GNU parallel and CDO. osa tarvittavista gribeistä meillä koneella valmiiksi (statistiset arvot) niin niistä jtn mistä saa. myös ne remap tiedostot. Similar processing with get-edte.sh. 

Predicting with the trained model happens with xgb-predict-swi2.py (seasonal forecast) and xgb-predict-edte.py (EDTE), with input gribs from previous bash scripts. These Python scripts use Xarray to join different input grids into one data frame that includes all time steps for each input in the target grid. Then prediction for SWI2 is made with XGBoost predict with the previously trained model. Ready file is then returned to the bash script so that the resulting grib file is fixed and the end product is set up to our SmartMet server ( https://desm.harvesterseasons.com/grid-gui?session=bg=light;bl=1;cl=Grey;cm=None;f=;fn=;ft=;g=;gm=;hu=128;is=DarkGrey;iv=Generated;k=;l=;lb=Grey;lm=LightGrey;lo=None;lt=;m=0;max=16;mi=Default;min=6;p=;pg=main;pi=;pn=;pre=Image;pro=;sa=60;sm=LightCyan;st=10;sy=None;t=;tg=;tgt=Month;u=;xx=;yy=;&pi=16). 
