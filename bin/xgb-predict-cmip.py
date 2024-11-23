#!/bin/env python3
import pandas as pd
import xgboost as xgb
import sys
import matplotlib.pyplot as plt
#from Sitia_023330_ece3 import *
from Bremerhaven_004885_ece3 import *


mod_dir='/home/ubuntu/data/ML/models/OCEANIDS/' # saved mdl
pred_dir='/home/ubuntu/data/ML/training-data/OCEANIDS/' # training data



# Load the prediction data
df_fin = pd.read_csv(pred_dir + "prediction_data_ece3-bremerhaven-2015-2050-updated.csv")
df_result = pd.DataFrame(df_fin['utctime'])

if pred == 'WG_PT24H_MAX' or pred == 'WS_PT24H_AVG':
    for i in [1,2,3,4]:
        df_result[f"sfcWind-{i}"] = df_fin[f"sfcWind-{i}"]
elif pred == 'TN_PT24H_MIN':
    for i in [1,2,3,4]:
        df_result[f"tasmin-{i}"] = df_fin[f"tasmin-{i}"] -273.15
elif pred == 'TX_PT24H_MAX':
    for i in [1,2,3,4]:
        df_result[f"tasmax-{i}"] = df_fin[f"tasmax-{i}"] -273.15
else:
    for i in [1,2,3,4]:
        df_result[f"pr-{i}"] = df_fin[f"pr-{i}"]
    

# Load the model
fitted_mdl = xgb.XGBRegressor()
fitted_mdl.load_model(mod_dir + mdl_name)

# Ensure the DataFrame has the correct columns
required_columns = fitted_mdl.get_booster().feature_names
df_fin = df_fin[required_columns]

# XGBoost predict without DMatrix
result = fitted_mdl.predict(df_fin)
result = result.tolist()
df_result['Predicted'] = result


df_result.to_csv(f"/home/ubuntu/data/ML/results/OCEANIDS/ece3-{harbor}-{pred}-updated-prediction.csv", index=False)
print(df_result)
