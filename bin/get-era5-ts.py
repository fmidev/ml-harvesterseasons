#!/usr/bin/env python3
import requests, os, time, glob, json,sys
import pandas as pd
import functions as fcts
import numpy as np
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

### Timeseries query to smarmet-server to fetch ERA5 training data for XGBoost
# era5 dataa vaan 2.3.2022 asti, pitäis ladata loppuvuos myös ja 20230101000000
startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/'
#lucas=data_dir+'soilwater/lucas_allA-Hclass_rep.csv'
lucas=data_dir+'LUCAS_2018_Copernicus_attr+additions.csv'
cols_own=['POINT_ID','TH_LAT','TH_LONG']#,'NUTS0','CPRN_LC','LC1_LABEL','DTM_height','DTM_slope','DTM_aspect','TCD','WAW','CorineLC']
lucas_df=pd.read_csv(lucas,usecols=cols_own)
print(lucas_df)
lst=lucas_df[['TH_LAT','TH_LONG']].values.tolist() 
points=lucas_df[['POINT_ID']].values.tolist()
latlonlst=list(itertools.chain.from_iterable(lst))
latlonlst=latlonlst[:10000] # [:10000] [10000:20000] [20000:30000] ... [120000:]
pointslst=list(itertools.chain.from_iterable(points))
pointslst=pointslst[:5000] # [:5000 [5000:10000] [10000:15000] ... [60000:]

era50012sl = [
    {'t2m': 'TAS-K:ERA5:5021:1:0:1:0'},             # 2m temperature in K 
    {'td': 'TD2-K:ERA5:5021:1:0:1:0'},               # 2m dew point temperature in K
    {'msl': 'PSEA-PA:ERA5:5021:1:0:1:0'},          # mean sea level pressure in Pa
    #{'cape': 'CAPE-JKG:ERA5:5021:1:0:1:0'},       # CAPE in J/kg
    #{'cin': 'CIN-JKG:ERA5:5021:1:0:1:0'},         # CIN in J/kg
    {'u10': 'U10-MS:ERA5:5021:1:0:1:0'},           # U comp of wind in m/s
    {'v10': 'V10-MS:ERA5:5021:1:0:1:0'},           # V comp of wind in m/s
    {'z': 'Z-M:ERA5:5021:1:0:1:0'},                 # geopotential in m
    {'sdor': 'SDOR-M:ERA5:5021:1:0:1:0'},           # Standard deviation of orography
    {'slor': 'SLOR:ERA5:5021:1:0:1:0'},               # Slope of sub-gridscale orography
    {'anor': 'ANOR-RAD:ERA5:5021:1:0:1:0'},       # Angle of sub-gridscale orography
    #{'ssrc': 'SSRC-JM2:ERA5:5021:1:0:1:0'},       # Surface net solar radiation, clear sky
    #{'cbh': 'CBH-M:ERA5:5021:1:0:1:0'},             # Cloud base height
    #{'i10fg': 'I10FG-MS:ERA5:5021:1:0:1:0'},       # Instantaneous 10 m wind gust
    {'tcc': 'N-0TO1:ERA5:5021:1:0:1:0'},         # Cloudiness 0...1
    #{'hcc': 'NH-0TO1:ERA5:5021:1:0:1:0'},        # High cloud amount
    #{'lcc': 'NL-0TO1:ERA5:5021:1:0:1:0'},        # Low cloud amount
    #{'mcc': 'NM-0TO1:ERA5:5021:1:0:1:0'},        # Middle cloud amount
    #{'cp': 'RRC-KGM2:ERA5:5021:1:0:1:0'},           # Convective precipitation 
    {'sd': 'SD-KGM2:ERA5:5021:1:0:1:0'},         # Water equivalent of snow cover in mm
    {'skt': 'SKT-K:ERA5:5021:1:0:1:0'},             # Skin temperature
    {'rsn': 'SND-KGM3:ERA5:5021:1:0:1:0'},       # Snow density
    {'swvl1': 'SOILWET-M3M3:ERA5:5021:9:7:1:0'}, # Surface soil wetness
    #{'sro': 'SRO-M:ERA5:5021:1:0:1:0'},             # Surface runoff
    #{'ssro': 'SSRO-M:ERA5:5021:1:0:1:0'},           # Sub-surface runoff
    {'stl1': 'STL1-K:ERA5:5021:9:7:1:0'},           # Soil temperature level 1
    #{'stl3': 'STL3-K:ERA5:5021:9:7268:1:0'},        # level 3
    #{'stl4': 'STL4-K:ERA5:5021:9:25855:1:0'},       # level 4
    #{'strc': 'STRC-JM2:ERA5:5021:1:0:1:0'},       # Surface net thermal radiation, clear sky
    {'swvl2': 'SWVL2-M3M3:ERA5:5021:9:1820:1:0'}, # Volumetric soil water layer 2
    {'swvl3': 'SWVL3-M3M3:ERA5:5021:9:7268:1:0'}, # layer 3
    {'swvl4': 'SWVL4-M3M3:ERA5:5021:9:25855:1:0'}, # layer 4
    {'mx2t': 'TMAX-K:ERA5:5021:1:0:1:0'},           # Maximum temperature
    {'mn2t': 'TMIN-K:ERA5:5021:1:0:1:0'},           # Minimum temperature
    #{'stl2': 'TSOIL-K:ERA5:5021:9:1820:1:0'},          # Soil temperature
    #{'u100': 'U100-MS:ERA5:5021:1:0:1:0'},         # 100 metre wind components U and V
    #{'v100': 'V100-MS:ERA5:5021:1:0:1:0'},
    #{'mcpr': 'MCPR-KGM2S:ERA5:5021:1:0:1:0'},   # Mean convective precipitation rate
    #{'mer': 'MER-KGM2S:ERA5:5021:1:0:1:0'},     # Mean evaporation rate
    #{'mlspr': 'MLSPR-KGM2S:ERA5:5021:1:0:1:0'}, # Mean large-scale precipitation rate
    #{'mror': 'MROR-KGM2S:ERA5:5021:1:0:1:0'},   # Mean runoff rate
    #{'mslhf': 'MSLHF-WM2:ERA5:5021:1:0:1:0'},     # Mean surface latent heat flux
    #{'msr': 'MSR-KGM2S:ERA5:5021:1:0:1:0'},     # Mean snowfall rate
    #{'msror': 'MSROR-KGM2S:ERA5:5021:1:0:1:0'}, # Mean surface runoff rate
    #{'msshf': 'MSSHF-WM2:ERA5:5021:1:0:1:0'},     # Mean surface sensible heat flux
    #{'mssror': 'MSSROR-KGM2S:ERA5:5021:1:0:1:0'}, # Mean sub-surface runoff rate
    #{'mtpr': 'MTPR-KGM2S:ERA5:5021:1:0:1:0'},   # Mean total precipitation rate 
    {'pev': 'EVAPP-M:ERA5:5021:1:0:1:0'},           # Potential evaporation
    {'cl': 'CL-0TO1:ERA5:5021:1:0:1:0'},         # Lake cover
    {'cvh': 'CVH-N:ERA5:5021:1:0:1:0'},             # High vegetation cover
    {'cvl': 'CVL-N:ERA5:5021:1:0:1:0'},             # Low vegetation cover
    #{'dl': 'DL-M:ERA5:5021:1:0:1:0'},               # Lake total depth
    {'laihv': 'LAI_HV-M2M2:ERA5:5021:1:0:1:0'},  # Leaf area index high vegetation
    {'lailv': 'LAI_LV-M2M2:ERA5:5021:1:0:1:0'},  # Leaf area index low vegetation
    {'lsm': 'LC-0TO1:ERA5:5021:1:0:1:0'},        # Land cover: 1=land, 0=sea
    #{'lshf': 'LSHF:ERA5:5021:1:0:1:0'},               # Lake shape factor
    {'slt': 'SOILTY-N:ERA5:5021:1:0:1:0'},            # Soil type
    {'tvh': 'TVH-N:ERA5:5021:1:0:1:0'},             # Type of high vegetation
    {'tvl': 'TVL-N:ERA5:5021:1:0:1:0'}             # Type of low vegetation
    ]

# ERA5 pressure level hour-00 and hour-12 data 
era50012pl = [
    {'t850': 'T-K:ERA5:5021:2:850:1:0'}, # temperature in K        
    {'t700': 'T-K:ERA5:5021:2:700:1:0'},  
    {'t500': 'T-K:ERA5:5021:2:500:1:0'},
    {'q850': 'Q-KGKG:ERA5:5021:2:850:1:0'}, # specific humidity in kg/kg
    {'q700': 'Q-KGKG:ERA5:5021:2:700:1:0'},
    {'q500': 'Q-KGKG:ERA5:5021:2:500:1:0'},
    {'u850': 'U-MS:ERA5:5021:2:850:1:0'}, # U comp of wind in m/s
    {'u700': 'U-MS:ERA5:5021:2:700:1:0'},
    {'u500': 'U-MS:ERA5:5021:2:500:1:0'},
    {'v850': 'V-MS:ERA5:5021:2:850:1:0'}, # V comp of wind in m/s
    {'v700': 'V-MS:ERA5:5021:2:700:1:0'},
    {'v500': 'V-MS:ERA5:5021:2:500:1:0'},
    {'z850': 'Z-M:ERA5:5021:2:850:1:0'}, # geopotential in m
    {'z700': 'Z-M:ERA5:5021:2:700:1:0'},
    {'z500': 'Z-M:ERA5:5021:2:500:1:0'},            
    {'kx': 'KX:ERA5:5021:1:0:0'}, # K index
]

# ERA5 single level 24h aggregation data
era524h = [
    {'tp': 'RR-M:ERA5:5021:1:0:1:0'}, # total precipitation in m
    {'e': 'EVAP-M:ERA5:5021:1:0:1:0'}, # evaporation in m
    {'tsr': 'TSR-JM2:ERA5:5021:1:0:1:0'}, # Top net solar radiation
    {'fg10': 'FFG-MS:ERA5:5021:1:0:1:0'}, # 10 metre wind gust since previous post-processing
    {'slhf': 'FLLAT-JM2:ERA5:5021:1:0:1:0'}, # Surface latent heat flux
    {'sshf': 'FLSEN-JM2:ERA5:5021:1:0:1:0'}, # Surface sensible heat flux
    {'ro': 'RO-M:ERA5:5021:1:0:1:0'}, # Runoff in depth of water in meters
    {'sf': 'SNACC-KGM2:ERA5:5021:1:0:1:0'}, # Snowfall (m of water equivalent)
]

y_start='2015'
y_end='2022'
rr_yrs=list(range(int(y_start),int(y_end)+1))
nan=float('nan')

source='smdev.harvesterseasons.com:8080'

### sl for 00 and 12 UTC
tstep='1440' # 24h
for pardict in era50012sl:
    key,value=list(pardict.items())[0]
    
    # 1: generation 20100101T000000_20191231T120000
    start='20150101T000000Z'
    end='20191231T000000Z'
    era5sl1=fcts.smartmet_ts_query_multiplePoints(source,start,end,tstep,latlonlst,pardict,pointslst)
    # 2: generation 20200101T000000_20220320T120000
    start='20200101T000000Z'
    end='20221231T000000Z'
    era5sl2=fcts.smartmet_ts_query_multiplePoints(source,start,end,tstep,latlonlst,pardict,pointslst)
    # join generations
    df=pd.concat([era5sl1,era5sl2],ignore_index=True)
    # group together same staids and keep dates in order
    df=df.sort_values(by=['pointID','utctime'],ignore_index=True) 
    df.rename({key:key+'-00'}, axis=1, inplace=True)
    print(df)
    for point in pointslst:
        dfpoint = df[df['pointID'] == point]
        dfpoint.to_csv(data_dir+'soilwater/era5/era5-'+key+'-00_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)

    # 1: generation 20100101T000000_20191231T120000
    start='20141231T120000Z'
    end='20191231T120000Z'
    era5sl1=fcts.smartmet_ts_query_multiplePoints(source,start,end,tstep,latlonlst,pardict,pointslst)
    # 2: generation 20200101T000000_20220320T120000
    start='20200101T120000Z'
    end='20221231T120000Z'
    era5sl2=fcts.smartmet_ts_query_multiplePoints(source,start,end,tstep,latlonlst,pardict,pointslst)
    # join generations
    df=pd.concat([era5sl1,era5sl2],ignore_index=True)
    # group together same staids and keep dates in order
    df=df.sort_values(by=['pointID','utctime'],ignore_index=True) 
    df.rename({key:key+'-12'}, axis=1, inplace=True)
    print(df)
    for point in pointslst:
        dfpoint = df[df['pointID'] == point]
        dfpoint.to_csv(data_dir+'soilwater/era5/era5-'+key+'-12_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)

### pl for 00 and 12 UTC
tstep='1440' # 24h
for pardict in era50012pl:
    key,value=list(pardict.items())[0]
    
    # 1: generation 20100101T000000_20191231T120000
    start='20150101T000000Z'
    end='20191231T000000Z'
    era5pl1=fcts.smartmet_ts_query_multiplePoints(source,start,end,tstep,latlonlst,pardict,pointslst)
    # 2: generation 20200101T000000_20220320T120000
    start='20200101T000000Z'
    end='20221231T000000Z'
    era5pl2=fcts.smartmet_ts_query_multiplePoints(source,start,end,tstep,latlonlst,pardict,pointslst)
    # join generations
    df=pd.concat([era5pl1,era5pl2],ignore_index=True)
    # group together same staids and keep dates in order
    df=df.sort_values(by=['pointID','utctime'],ignore_index=True) 
    df.rename({key:key+'-00'}, axis=1, inplace=True)
    print(df)
    for point in pointslst:
        dfpoint = df[df['pointID'] == point]
        dfpoint.to_csv(data_dir+'soilwater/era5/era5-'+key+'-00_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)

    # 1: generation 20100101T000000_20191231T120000
    start='20141231T120000Z'
    end='20191231T120000Z'
    era5pl1=fcts.smartmet_ts_query_multiplePoints(source,start,end,tstep,latlonlst,pardict,pointslst)
    # 2: generation 20200101T000000_20220320T120000
    start='20200101T120000Z'
    end='20221231T120000Z'
    era5pl2=fcts.smartmet_ts_query_multiplePoints(source,start,end,tstep,latlonlst,pardict,pointslst)
    # join generations
    df=pd.concat([era5pl1,era5pl2],ignore_index=True)
    # group together same staids and keep dates in order
    df=df.sort_values(by=['pointID','utctime'],ignore_index=True) 
    df.rename({key:key+'-12'}, axis=1, inplace=True)
    print(df)
    for point in pointslst:
        dfpoint = df[df['pointID'] == point]
        dfpoint.to_csv(data_dir+'soilwater/era5/era5-'+key+'-12_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)

### ERA5 data 24h aggregation and rolling cumsums###
tstep='60' # 1h
for pardict in era524h:
    key,value=list(pardict.items())[0]
    # 1
    start='20140901T000000Z'
    end= '20181231T230000Z'
    df1=fcts.smartmet_ts_query_multiplePoints(source,start,end,tstep,latlonlst,pardict,pointslst)
    # 2
    start='20190101T000000Z'
    end= '20230101T000000Z'
    df2=fcts.smartmet_ts_query_multiplePoints(source,start,end,tstep,latlonlst,pardict,pointslst)
    era524h=pd.concat([df1,df2],ignore_index=True)
    era524h=era524h.sort_values(by=['pointID','utctime'],ignore_index=True)
    era524h=era524h.set_index('utctime')
    '''
    ERA5 hourly rain to daily sums per station
    hour 00:00:00 rain always summed to the previous day daily sum:
    "To cover total precipitation for 1st January 2017, we need two days of data. 
    a) 1st January 2017 time = 01 - 23  will give you total precipitation data to cover 00 - 23 UTC for 1st January 2017 
    b) 2nd January 2017 time = 00 will give you total precipitation data to cover 23 - 24 UTC for 1st January 2017"
    '''
    era524h=era524h.groupby(['pointID'])[[key,'latitude','longitude','pointID']].shift(-1, axis = 0) # shift columns 1 up for daily sum calc to be correct
    era524h=era524h.dropna()
    era524h['utctime'] = pd.to_datetime(era524h.index)
    era524h=era524h.groupby(['pointID','latitude','longitude'])[['utctime',key]].resample('D',on='utctime').sum()
    era524h=(era524h.reset_index())
    era524h=era524h.sort_values(by=['pointID','utctime'],ignore_index=True) # probably not necessary
    era524h=era524h.set_index('utctime')
    era524h['utctime']=pd.to_datetime(era524h.index)
    df5=fcts.rolling_cumsum(era524h.copy(),'5d',key)
    df15=fcts.rolling_cumsum(era524h.copy(),'15d',key)
    df60=fcts.rolling_cumsum(era524h.copy(),'60d',key)
    df100=fcts.rolling_cumsum(era524h.copy(),'100d',key)
    era524h=era524h.loc[(era524h['utctime'] >='2015-01-01')]
    print(era524h)
    for point in pointslst:
        dfpoint = [era524h['pointID'] == point]
        dfpoint.to_csv(data_dir+'soilwater/era5/era5-'+key+'_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
        dfpoint = df5[df5['pointID'] == point]
        dfpoint.to_csv(data_dir+'soilwater/era5/era5-'+key+'5d'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
        dfpoint = df15[df15['pointID'] == point]
        dfpoint.to_csv(data_dir+'soilwater/era5/era5-'+key+'15d'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
        dfpoint = df60[df60['pointID'] == point]
        dfpoint.to_csv(data_dir+'soilwater/era5/era5-'+key+'60d'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
        dfpoint = df100[df100['pointID'] == point]
        dfpoint.to_csv(data_dir+'soilwater/era5/era5-'+key+'100d'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))