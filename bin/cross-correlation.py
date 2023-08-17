import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import r_regression
# creates cross-correlation matrix for all vars, and bar chart for swi2 comparison with others

def plot_corr_image(df,p,name1,name2):
    # cross-correlation matrix
    X=df[p].to_numpy()
    matrix=[]
    for val in p: 
        y=df[val].to_numpy()
        matrix.append(list(r_regression(X,y).round(decimals=2)))
        if val=='swi2': # print RR correlation values
            ccvalues=list(r_regression(X,y).round(decimals=2))
    M=np.array(matrix)
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
    plt.title("correlation with swi2")
    plt.savefig(name2,dpi=1000)

data_dir='/home/ubuntu/data/ML/training-data/soilwater/' # training data
res_dir='/home/ubuntu/data/ML/results/soilwater/figures/' # result figures

cols_own=['utctime','swi2','evap','evap5d','evap15d','evap60d','evap100d','evapp','evapp5d','evapp15d','evapp60d','evapp100d',
'laihv-00','laihv-12','lailv-00','lailv-12','ro','ro5d','ro15d','ro60d','ro100d','rsn-00','rsn-12','sd-00','sd-12','sf',
'skt-00','skt-12','slhf','sro','sro5d','sro15d','sro60d','sro100d','sshf','ssr','ssrd','ssro','ssro5d','ssro15d','ssro60d',
'ssro100d','stl1-00','stl1-12','str','strd','swvl1-00','swvl1-12','swvl2-00','swvl2-12','swvl3-00','swvl3-12','swvl4-00',
'swvl4-12','t2-00','t2-12','td2-00','td2-12','tp','tp5d','tp15d','tp60d','tp100d','u10-00','u10-12','v10-00','v10-12',
'TH_LAT','TH_LONG','DTM_height','DTM_slope', 'DTM_aspect','TCD','WAW'
]
#fname='swi2_training_236lucasPoints_2015-2022_all_FIXED.csv'
#fname='swi2_training_4108lucasPoints_2015-2022_all.csv'
fname='swi2_training_10000lucasPoints_2015-2022_all_fixed.csv'
df=pd.read_csv(data_dir+fname,usecols=cols_own)

df['utctime']=pd.to_datetime(df['utctime'])
df['dayOfYear'] = df['utctime'].dt.dayofyear
#print(df)
#df=df.drop(columns=['utctime'])

# drop negative swi2, NaN and -99999 values
s1=df.shape[0]
df=df.dropna(axis=0,how='any')
df=df[(df.DTM_slope !=-99999.000000) & (df.DTM_height !=-99999.000000) & (df.DTM_aspect !=-99999.000000)] 
df = df[df['swi2'] >= 0]
s2=df.shape[0]
print('From '+str(s1)+' dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')

vars=['swi2','evap','evap5d','evap15d','evap60d','evap100d','evapp','evapp5d','evapp15d','evapp60d','evapp100d',
'laihv-00','laihv-12','lailv-00','lailv-12','ro','ro5d','ro15d','ro60d','ro100d','rsn-00','rsn-12','sd-00','sd-12','sf',
'skt-00','skt-12','slhf','sro','sro5d','sro15d','sro60d','sro100d','sshf','ssr','ssrd','ssro','ssro5d','ssro15d','ssro60d',
'ssro100d','stl1-00','stl1-12','str','strd','swvl1-00','swvl1-12','swvl2-00','swvl2-12','swvl3-00','swvl3-12','swvl4-00',
'swvl4-12','t2-00','t2-12','td2-00','td2-12','tp','tp5d','tp15d','tp60d','tp100d','u10-00','u10-12','v10-00','v10-12',
'TH_LAT','TH_LONG','DTM_height','DTM_slope','DTM_aspect','dayOfYear'
]
df=df[vars]
print(df)

#name1=res_dir+'4108points_cross-correlation_matrix.png'
#name2=res_dir+'4108points_cross-correlation_bar.png'
name1=res_dir+'10000points_cross-correlation_matrix.png'
name2=res_dir+'10000points_cross-correlation_bar.png'

plot_corr_image(df,vars,name1,name2)
