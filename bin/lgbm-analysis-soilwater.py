import os, time, random, warnings,sys
import pandas as pd
import numpy as np
#from lightgbm import LGBMRegressor as lgb
import lightgbm as lgb
from lightgbm import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

path='/home/ubuntu/data/ML/models/soilwater/'
model_name=path+'lgbm_mdl_swi2_2015-2022_10000points-1.txt'
lgmodel = lgb.Booster(model_file=model_name)

# plotting feature importance
plot_importance(lgmodel,figsize=(10,10),height=0.5,grid=None)
plt.savefig('/home/ubuntu/data/ML/results/soilwater/figures/plot_importance_swi2_lgbm-2.jpg')
