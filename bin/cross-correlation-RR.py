import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import r_regression
# PRECIPITATION (RR)
# creates cross-correlation matrix for all vars, and bar chart for RR comparison with others

def plot_corr_image(df,p,name1,name2):
    # cross-correlation matrix
    X=df[p].to_numpy()
    matrix=[]
    for val in p: 
        y=df[val].to_numpy()
        matrix.append(list(r_regression(X,y).round(decimals=2)))
        if val=='RR': # print RR correlation values
            ccvalues=list(r_regression(X,y).round(decimals=2))
    M=np.array(matrix)
    print(matrix)
    # plot the matrix as an image with an appropriate colormap
    plt.imshow(M.T, aspect='auto', cmap="bwr",vmin=-1,vmax=1)
    # add the column names and values
    for i,var in enumerate(p):
        plt.text(-1,i,var,va='center',ha='right',fontsize=3)
        plt.text(i,-1,var,rotation=90.0,ha='center',fontsize=3)
    for (i, j), value in np.ndenumerate(M):
        plt.text(i, j, "%.2f"%value, va='center', ha='center',fontsize=1.5)
    plt.axis('off')
    plt.colorbar()
    plt.savefig(name1,dpi=1000)

    # bar plot
    x=np.array(ccvalues)
    bigvals=np.where(x>=0.25)[0]
    smallvals=np.where(x<=-0.25)[0]
    fig = plt.figure(figsize = (10, 10))
    plt.grid(zorder=0)
    plt.bar(vars, ccvalues, color ='maroon',
            width = 0.4,zorder=3)
    plt.xticks(vars, rotation=90,fontsize=7)
    for i in bigvals:
        plt.gca().get_xticklabels()[i].set_color('red') 
    for i in smallvals:
        plt.gca().get_xticklabels()[i].set_color('blue')     
    plt.ylim(-1.0,1.0)
    plt.xlabel("parameters")
    plt.ylabel("corr.")
    plt.title("correlation with RR")
    plt.savefig(name2,dpi=1000)

data_dir='/home/ubuntu/data/ML/training-data/precipitation-harmonia/' # training data
res_dir='/home/ubuntu/data/ML/results/precipitation-harmonia/figures/'

cols_own=[
    'utctime','latitude','longitude','pointID',
    'anor','evap','kx-00','lsm','mn2t-00','msl-00','mx2t-00',
    'q500-00','q700-00','q850-00','ro','rsn-00','sd-00','sdor',
    'sf','skt-00','slhf','slor','sshf','ssr','ssrd','strd',#'sst-00',
    't2m-00','t500-00','t700-00','t850-00','tcc-00','td2m-00','tp',
    'tsr','ttr','u10-00','u500-00','u700-00','u850-00','v10-00',
    'v500-00',#'v700-00',
    'v850-00','z',#'z500-00',
    'z700-00','z850-00'
]
fera5='RR_era5_training_6766stations_2000-2020_all.csv.gz' # era5 training data
feobs='RR_eobs_training_6766stations_2000-2020_all.csv.gz' # eobs training data
print(fera5)
print(feobs)
df_era5=pd.read_csv(data_dir+fera5,usecols=cols_own)
df_eobs=pd.read_csv(data_dir+feobs)
df_eobs.rename(columns={'DATE':'utctime','STAID':'pointID','LAT': 'latitude', 'LON': 'longitude'}, inplace=True)
df=pd.merge(df_era5,df_eobs, on=['utctime','pointID','latitude','longitude'])

df['utctime']=pd.to_datetime(df['utctime'])
df['dayOfYear'] = df['utctime'].dt.dayofyear

# drop NaN values
s1=df.shape[0]
df=df.dropna(axis=0,how='any')
df=df[(df.HGHT !=-99999) & (df.SLOPE !=-99999) & (df.ASPECT !=-99999)
      & (df.TPI !=-99999) & (df.TRI !=-99999) & (df.DTW !=-99999) & (df.DTR !=-99999)
      & (df.DTL !=-99999) & (df.DTO !=-99999) & (df.FCH !=-99999)] 
df=df[(df.HGHT !=-9999) & (df.SLOPE !=-9999) & (df.ASPECT !=-9999)
      & (df.TPI !=-9999) & (df.TRI !=-9999) & (df.DTW !=-9999) & (df.DTR !=-9999)
      & (df.DTL !=-9999) & (df.DTO !=-9999) & (df.FCH !=-9999)] 
s2=df.shape[0]
print('From '+str(s1)+' dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')
print(df)

vars=['RR','latitude','longitude','HGHT','SLOPE','ASPECT',
    'TPI','TRI','DTW','DTR','DTL','DTO','FCH',
    'anor','evap','kx-00','lsm','mn2t-00','msl-00','mx2t-00',
    'q500-00','q700-00','q850-00','ro','rsn-00','sd-00','sdor',
    'sf','skt-00','slhf','slor','sshf','ssr','ssrd','strd',#'sst-00',
    't2m-00','t500-00','t700-00','t850-00','tcc-00','td2m-00','tp',
    'tsr','ttr','u10-00','u500-00','u700-00','u850-00','v10-00',
    'v500-00',#'v700-00',
    'v850-00','z',#'z500-00',
    'z700-00','z850-00',
    'dayOfYear'
]
print(len(vars))
df=df[vars]
print(df)

name1=res_dir+'RR_cross-correlation_matrix_ALLpreds.png'
name2=res_dir+'RR_cross-correlation_bar_ALLpreds.png'

plot_corr_image(df,vars,name1,name2)
