import pandas as pd
import numpy as np
from dataclasses import dataclass
import geopandas as gpd
import matplotlib.pyplot as plt
# script to check how lucas land cover types and countries are distributed in the ML subset of points

data_dir='/home/ubuntu/data/ML/training-data/'
lucas=data_dir+'lucas/LUCAS_2018_Copernicus_attr+additions_AT-UK.csv'
cols_own=['NUTS0','CPRN_LC','POINT_ID','CorineLC']
df_all=pd.read_csv(lucas,usecols=cols_own)
df_all['POINT_ID']=pd.to_numeric(df_all['POINT_ID'])
#print(df_all)

# subset
with open(r'/home/ubuntu/data/ML/training-data/10000pointIDs-2.txt', 'r') as file:
    lines = [line.rstrip() for line in file]
pointslst=[]
for sta in lines:
    pointslst.append(int(sta))
df_subset=df_all.loc[df_all['POINT_ID'].isin(pointslst)]

 
# display dataframe
print('Data:')
print(df_subset)
print('Occurrence counts of combined columns:')
 # count occurrences of combined columns
occur = df_subset.groupby(['NUTS0', 'CPRN_LC']).size()
 # display occurrences of combined columns
occur.to_csv('testi.csv')
print(occur)


cprnLC=['A1','A2','A3','B1','B2','B3','B4','B5','B7','B8','C1','C2','C3','D1','D2',
    'E1','E2','E3','F1','F2','F3','F4','G1','H1']
for lc in cprnLC: 
    lc_subset=len(df_subset[df_subset['CPRN_LC'] == lc])
    lc_all=len(df_all[df_all['CPRN_LC'] == lc])
    print(lc,lc_subset, lc_all)
print('')

countries=['AT','BE','BG','CY','CZ','DE','DK','EE','EL','ES','FI',
    'FR','HR','HU','IE','IT','LT','LU','LV','MT','NL','PL','PT','RO','SE','SI','SK','UK']
for c in countries: 
    c_subset=len(df_subset[df_subset['NUTS0'] == c])
    c_all=len(df_all[df_all['NUTS0'] == c])
    print(c,c_subset, c_all)


# koko setissä maapisteiden lkm per maa ja cprn-lc lkm per luokka
# prosenttiosuus esim 34% AT kaikista 10 000 points ja numero esim. 2345 points AT 
# prosenttiosuus esim 40% CPRN_LC C1 kaikista 10 000 points ja numero esim. 100 points C1 
# kuva jakaumista, C1 Euroopan kartalla jne
'''AF_df=df.loc[(df['CPRN_LC']=='A1') | (df['CPRN_LC']=='A2') | (df['CPRN_LC']=='A3') | (df['CPRN_LC']=='F1') | (df['CPRN_LC']=='F2') | (df['CPRN_LC']=='F3') | (df['CPRN_LC']=='F4')].sample(n=AF_rep_nr)
B_df=df.loc[(df['CPRN_LC']=='B1') | (df['CPRN_LC']=='B2') | (df['CPRN_LC']=='B3') | (df['CPRN_LC']=='B4') | (df['CPRN_LC']=='B5') | (df['CPRN_LC']=='B7') | (df['CPRN_LC']=='B8')].sample(n=B_rep_nr)
C1_df=df.loc[(df['CPRN_LC']=='C1')].sample(n=C1_rep_nr)
C2_df=df.loc[(df['CPRN_LC']=='C2')].sample(n=C2_rep_nr)
C3_df=df.loc[(df['CPRN_LC']=='C3')].sample(n=C3_rep_nr)
D_df=df.loc[(df['CPRN_LC']=='D1') | (df['CPRN_LC']=='D2')].sample(n=D_rep_nr)
E_df=df.loc[(df['CPRN_LC']=='E1') | (df['CPRN_LC']=='E2') | (df['CPRN_LC']=='E3')].sample(n=E_rep_nr)
G1_df=df.loc[(df['CPRN_LC']=='G1')].sample(n=G1_rep_nr) # no data for this point
H1_df=df.loc[(df['CPRN_LC']=='H1')].sample(n=H1_rep_nr)
'''