
# Harvestability

## Training

new training data under /home/ubuntu/data/ml-harvestability/training/soildata/

#### Prepare dataset
1. convert Finland harvestability to 4326
 gdalwarp -t_srs EPSG:4326 /vsicurl/https://pta.data.lit.fmi.fi/geo/harvestability/KKL_SMK_Suomi_2021_06_01-UTM35.tif harvestability.tif

2. extracting all locations from ground truth data
 gdal_translate -of XYZ harvestability.tif harvestability_gt_locations.txt
 
3. Get random points from every place in finland, Get points near to LUCAS reference points. This will produce train_locations.txt file with random lat, long values.

```sh
python training/sampling.py
```

4. Get harvestability values from random points ( this is for checking)

```sh
cat train_locations.txt | gdallocationinfo -wgs84 -valonly harvestability.tif > train_harvestability.txt
```
 
5. run add_feature.sh to get every feature added to train_data.csv

```sh
bash add_feature.sh
```

6. remove space to , in train_data.csv

 :%s/ /,/g
 
7. open train_data.csv and add the below header in train_data.csv

sand_0-5cm_mean,sand_5-15cm_mean,silt_0-5cm_mean,silt_5-15cm_mean,clay_0-5cm_mean,clay_5-15cm_mean,soc_0-5cm_mean,soc_5-15cm_mean,TH_LONG,TH_LAT,DTM_height,DTM_slope,DTM_aspect,TCD,WAW,harvestability,TWI

#### Classifiers

1. XGB binary
```sh
python training/train_soil_binary.py
```
2. XGB mineral
```sh
python training/train_soil_mineral.py
```
3. XGB peatland
```sh
python training/train_soil_peatlands.py
```

#### Hyper parameter tuning

1. XGB binary
```sh
python training/xgb_harvestability_soil_binary_optuna.py
```
2. XGB mineral
```sh
python training/xgb_harvestability_soil_minerals_optuna.py
```
3. XGB peatland
```sh
python training/xgb_harvestability_soil_peatlands_optuna.py
```

#### Prediction

1. To run all countries

```sh
bash prediction/predict_parallel_direct.sh
```
2. To run for single country with upload functionality

```sh
bash prediction/upload_predict_direct.sh
```

3. To run just prediction 


```sh
python prediction/predict_direct.py
```

*Note* : If any country is having problems with tif generation run split_n_merge.sh script for those countries

```sh
bash prediction/split_n_merge.sh
```

#### Post processing

1. ##### country cut

```sh
parallel bash prediction/cut_country_shp.sh {} :::: crop_country.lst
```
2. ##### separate water body
```sh
parallel bash prediction/mask_water.sh {} :::: countries.lst 
```
## TODO



