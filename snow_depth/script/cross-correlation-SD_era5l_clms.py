import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import r_regression
# creates cross-correlation matrix for all vars, and bar chart for SD comparison with others

def plot_corr_image(df,p,name1,name2):
    # cross-correlation matrix
    X=df[p].to_numpy()
    matrix=[]
    for val in p: 
        y=df[val].to_numpy()
        matrix.append(list(r_regression(X,y).round(decimals=2)))
        if val=='SD': # print SD correlation values
            ccvalues=list(r_regression(X,y).round(decimals=2))
    M=np.array(matrix)
    #print(matrix)
    # plot the matrix as an image with an appropriate colormap
    plt.imshow(M.T, aspect='auto', cmap="bwr",vmin=-1,vmax=1)
    # add the column names and values
    for i,var in enumerate(p):
        plt.text(-1,i,var,va='center',ha='right',fontsize=6)
        plt.text(i,-1,var,rotation=90.0,ha='center',fontsize=6)
    for (i, j), value in np.ndenumerate(M):
        plt.text(i, j, "%.2f"%value, va='center', ha='center',fontsize=3.5)
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
    plt.xticks(vars, rotation=90,fontsize=12)
    for i in bigvals:
        plt.gca().get_xticklabels()[i].set_color('red') 
    for i in smallvals:
        plt.gca().get_xticklabels()[i].set_color('blue')     
    plt.ylim(-1.0,1.0)
    plt.xlabel("parameters")
    plt.ylabel("corr.")
    plt.title("correlation with SD")
    plt.savefig(name2,dpi=1000)


data_dir='/home/ubuntu/data/ML/training-data/snowdepth/' # training data
res_dir='/home/ubuntu/data/ML/results/snowdepth/figures/'  # FIRST create this folder
#res_dir='/home/ubuntu/data/ML/training-data/snowdepth/' # result figures

# era5-Land columns:
# utctime,pointID,latitude,longitude,slhf,sshf,ssrd,strd,str,ssr,sf,laihv-00,lailv-00,sd-00,
# rsn-00,stl1-00,swvl2-00,t2-00,td2-00,u10-00,v10-00,ro,evap,tp
# We exclude:'latitude','longitude'
# also parameter 'tp' excluded since lots of data missing
cols_own=[
    'utctime','pointID','slhf','sshf','ssrd',
    'strd','str','ssr','sf','laihv-00','lailv-00','sd-00','rsn-00',
    'stl1-00','swvl2-00','t2-00','td2-00','u10-00','v10-00',
    'ro','evap'
]
#ro,evap,tp

# CLMS columns: utctime, swe_clms, latitude, longitude, pointID
cols_clms=[
    'utctime','pointID','swe_clms'
]

# eobs SD columns: STAID,DATE,SD,LAT,LON,HGHT
# training data files
fera5l='SD_era5l_training_9790stations_2006-2020_all.csv.gz' # (predictors) era5-L training data
feobs='SD_eobs_fmi_training_9790stations_2006-2020.csv.gz' # eobs (predictand) training data
fclms='SD_clms_swe_training_9790stations_2006-2020.csv' # (predictors) clms training data

print(fera5l)
print(fclms)
print(feobs)

df_era5l=pd.read_csv(data_dir+fera5l,usecols=cols_own)
df_clms=pd.read_csv(data_dir+fclms,usecols=cols_clms)
df_eobs=pd.read_csv(data_dir+feobs)

#print(df_era5l)
#print(df_clms)
#print(df_eobs)

df_eobs.rename(columns={'DATE':'utctime','STAID':'pointID','LAT': 'latitude', 'LON': 'longitude'}, inplace=True)
# SD column: replace nan's with 0
df_eobs['SD']=df_eobs['SD'].replace(np.nan,0)
#print(df_eobs)

# merge SD eobs, clms and era5l data
df_eobs_clms=pd.merge(df_clms,df_eobs, on=['utctime','pointID'])
df=pd.merge(df_era5l,df_eobs_clms, on=['utctime','pointID'])

#print('merging done')
df['utctime']=pd.to_datetime(df['utctime'])
df['dayOfYear'] = df['utctime'].dt.dayofyear
#print(df)

# If want to save the merged file:
#df.to_csv(data_dir+'SD_trainingdata_merged_all_9657points_2006_2020.csv',index=False)

# drop NaN values
s1=df.shape[0]
df=df.dropna(axis=0,how='any')
#df=df[(df.HGHT !=-999)]
#df = df[df['rsn-00'] !=99.999985] # needed or not??? 
s2=df.shape[0]
print('From '+str(s1)+' dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')
#print(df)

# free up storage space
df_eobs = []
df_era5l =[]
df_clms = []
df_eobs_clms = []

# parameter 'tp' excluded since missing data
vars=['SD','slhf','sshf','ssrd',
    'strd','str','ssr','sf','laihv-00','lailv-00','sd-00','rsn-00',
    'stl1-00','swvl2-00','t2-00','td2-00','u10-00','v10-00',
    'ro','evap','swe_clms','dayOfYear'
]
#ro, evap, tp, swe_clms, dayOfYear

print(len(vars))
#df=df[vars]
#print(df)

# If want to save the merged file: (not tested how long this takes)
#df.to_csv(data_dir+'SD_trainingdata_merged_all_9790points_2006_2020.csv',index=False)

#name1=res_dir+'SD_cross-correlation_matrix_ALLpreds.png'
#name2=res_dir+'SD_cross-correlation_bar_ALLpreds.png'
name1=res_dir+'SD_cross-corr_matrix_era5L_swe_preds_9790pointIDs_1.png'
name2=res_dir+'SD_cross-corr_bar_era5L_swe_preds_9790pointIDs_1.png'

plot_corr_image(df,vars,name1,name2)
