import xarray as xr
import cfgrib,time,sys
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta

y1=2020
day1='2023-08-02 00:00:00'
y2=int(day1[0:4])

tdelta=y2-y1
testi=datetime.timedelta(years=(y2-y1))
