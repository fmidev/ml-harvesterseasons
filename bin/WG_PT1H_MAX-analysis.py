import pandas as pd
import sys
#from Vuosaari_151028 import *
#from Raahe_101785 import *
from Rauma_101061 import *

data_dir='/home/ubuntu/data/ML/training-data/OCEANIDS/' # training data
print(fname)
df=pd.read_csv(data_dir+fname,usecols=cols_own)
print(df)
days=str(len(df))

# days when wind equal or over threshold k
k=int(sys.argv[1])
df_th=df[df.WG_PT1H_MAX>=k]
print(df_th)
days_th=str(len(df_th))
perc=len(df_th)/len(df)*100


print('From '+days+' days (2013-2023) over '+str(k)+' m/s winds were observed on '+days_th+' days')
print('This is '+str(round(perc,1))+' % of the time')
