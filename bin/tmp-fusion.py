import pandas as pd
import numpy as np

data_dir='/home/ubuntu/data/ML/training-data/precipitation-harmonia/' # input training data
output_dir='/home/ubuntu/data/ML/training-data/RR/' # output training data

fera5='RR_era5_training_6766stations_2000-2020_all.csv.gz' # era5 training data
feobs='RR_eobs_training_6766stations_2000-2020_all.csv.gz' # eobs training data
df_era5=pd.read_csv(data_dir+fera5)
print(df_era5)
eobs_cols=['STAID','DATE','RR','HGHT']
df_eobs=pd.read_csv(data_dir+feobs,usecols=eobs_cols)
print(df_eobs)
df_eobs.rename(columns={'DATE':'utctime','STAID':'pointID'}, inplace=True)#,'LAT': 'latitude', 'LON': 'longitude'
df=pd.merge(df_era5,df_eobs, on=['utctime','pointID'])#,'latitude','longitude'])
print(df)

'''s1=df.shape[0]
df=df.replace(-99999.0,np.nan)
df=df.replace(-9999.0,np.nan)
df=df.dropna(axis=0,how='any')
df = df[df['RR'] >= 0]
s2=df.shape[0]
print('From '+str(s1)+' dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')
'''
print(df)

df.to_csv(output_dir+'RR_training_5441stations_2000-2020_all.csv.gz',index=False,compression='gzip')