#!/usr/bin/env python3
import time, warnings,requests,json
import pandas as pd
import warnings
import functions as fcts
warnings.simplefilter(action='ignore', category=FutureWarning)
# SmarMet-server timeseries query to fetch ERA5 training data for ML
# training data preprocessed to match seasonal forecast (sf) data for prediction
# remember to: conda activate xgb 

startTime=time.time()

data_dir='/home/ubuntu/data/ML/training-data/RR/'
era5_dir=data_dir+'era5/' 

eobsinfo=data_dir+'eobs/eobsStaidInfo_reduced.csv'
eobs_df=pd.read_csv(eobsinfo)

lat=eobs_df['LAT'].values.tolist()
lon=eobs_df['LON'].values.tolist()
point_ids=eobs_df['STAID'].values.tolist()
points = [f"{point_id:06d}" for point_id in point_ids]

llpdict={i:[j, k] for i, j, k in zip(points,lat, lon)}
llpdict={k:llpdict[k] for k in list(llpdict.keys())[:-1]}

# static predictors
statics = [
    {'z': 'Z-M2S2:ERA5:5021:1:0:1:0'}, # geopotential in m2 s-2
    {'lsm': 'LC-0TO1:ERA5:5021:1:0:1:0'}, # Land sea mask: 1=land, 0=sea
    {'sdor': 'SDOR-M:ERA5:5021:1:0:1:0'}, # Standard deviation of orography
    {'slor': 'SLOR:ERA5:5021:1:0:1:0'}, # Slope of sub-gridscale orography
    {'anor': 'ANOR-RAD:ERA5:5021:1:0:1:0'}, # Angle of sub-gridscale orography
]

# 00 and 12 UTC predictors
predictors_0012 = [
    #{'u10':'U10-MS:ERA5:5021:1:0:1:0'}, # 10m u-component of wind (6h instantanous)
    #{'v10':'V10-MS:ERA5:5021:1:0:1:0'}, # 10m v-component of wind (6h instantanous)
    #{'td2':'TD2-K:ERA5:5021:1:0:1:0'}, # 2m dewpoint temperature (6h instantanous)
    #{'t2':'T2-K:ERA5:5021:1:0:1:0'}, # 2m temperature (6h instantanous)
    #{'msl':'PSEA-HPA:ERA5:5021:1:0:1:0'}, # mean sea level pressure (6h instantanous)
    #{'tsea':'TSEA-K:ERA5:5021:1:0:1'}, # sea surface temperature (6h instantanous)
    #{'tcc':'N-0TO1:ERA5:5021:1:0:1:0'}, # total cloud cover (6h instantanous)
    #{'kx': 'KX:ERA5:5021:1:0:0'}, # K index
    #{'sd': 'SD-M:ERA5:5021:1:0:1:0'}, # snow depth
    #{'t850': 'T-K:ERA5:5021:2:850:1:0'}, # temperature in K        
    #{'t700': 'T-K:ERA5:5021:2:700:1:0'},  
    #{'t500': 'T-K:ERA5:5021:2:500:1:0'},
    #{'q850': 'Q-KGKG:ERA5:5021:2:850:1:0'}, # specific humidity in kg/kg
    #{'q700': 'Q-KGKG:ERA5:5021:2:700:1:0'},
    #{'q500': 'Q-KGKG:ERA5:5021:2:500:1:0'},
    #{'u850': 'U-MS:ERA5:5021:2:850:1:0'}, # U comp of wind in m/s
    #{'u700': 'U-MS:ERA5:5021:2:700:1:0'},
    #{'u500': 'U-MS:ERA5:5021:2:500:1:0'},
    #{'v850': 'V-MS:ERA5:5021:2:850:1:0'}, # V comp of wind in m/s
    #{'v700': 'V-MS:ERA5:5021:2:700:1:0'},
    #{'v500': 'V-MS:ERA5:5021:2:500:1:0'},
    #{'z850': 'Z-M2S2:ERA5:5021:2:850:1:0'}, # geopotential in m2 s-2
    #{'z700': 'Z-M2S2:ERA5:5021:2:700:1:0'},
    #{'z500': 'Z-M2S2:ERA5:5021:2:500:1:0'},    
    ]

# 00 predictors 
predictors_00 = [
    #{'tlwc':'TCLW-KGM2:ERA5:5021:1:0:1:0'}, # total column cloud liquid water (24h instantanous) 
    #{'tcwv':'TOTCWV-KGM2:ERA5:5021:1:0:1:0'}, # total column water vapor here
    #{'swvl1':'SOILWET-M3M3:ERA5:5021:9:7:1:0'}, #
    #{'swvl2':'SWVL2-M3M3:ERA5:5021:9:1820:1:0'}, #
    #{'swvl3':'SWVL3-M3M3:ERA5:5021:9:7268:1:0'}, #
    #{'swvl4':'SWVL4-M3M3:ERA5:5021:9:25855:1:0'} #
]

# previous day 24h sums 
predictors_24hAgg = [
    #{'ewss':'sum_t(EWSS-NM2S:ERA5:5021:1:0:1:0/24h/0h)'}, # eastward turbulent surface stress (24h aggregation since beginning of forecast)
    #{'e':'sum_t(EVAP-M:ERA5:5021:1:0:1:0/24h/0h)'}, # evaporation (24h aggregation since beginning of forecast)
    #{'nsss':'sum_t(NSSS-NM2S:ERA5:5021:1:0:1:0/24h/0h)'}, # northward turbulent surface stress (24h aggregation since beginning of forecast)
    #{'slhf':'sum_t(FLLAT-JM2:ERA5:5021:1:0:1:0/24h/0h)'}, # surface latent heat flux (24h aggregation since beginning of forecast)
    #{'ssr':'sum_t(RNETSWA-JM2:ERA5:5021:1:0:1:0/24h/0h)'}, # surface net solar radiation (24h aggregation since beginning of forecast)
    #{'str':'sum_t(RNETLWA-JM2:ERA5:5021:1:0:1:0/24h/0h)'}, # surface net thermal radiation (24h aggregation since beginning of forecast)
    #{'sshf':'sum_t(FLSEN-JM2:ERA5:5021:1:0:1:0/24h/0h)'}, # surface sensible heat flux (24h aggregation since beginning of forecast)
    #{'ssrd':'sum_t(RADGLOA-JM2:ERA5:5021:1:0:1:0/24h/0h)'}, # surface solar radiation downwards (24h aggregation since beginning of forecast)
    #{'strd':'sum_t(RADLWA-JM2:ERA5:5021:1:0:1:0/24h/0h)'}, # surface thermal radiation downwards (24h aggregation since beginning of forecast)
    #{'tp':'sum_t(RR-M:ERA5:5021:1:0:1:0/24h/0h)'} # total precipitation (24h aggregation since beginning of forecast)
]
# previous day maximum (or minimum)
predictor_24hmax = [
    #{'fg10':'max_t(FFG-MS:ERA5:5021:1:0:1:0/24h/0h)'}, # 10m wind gust since previous post-processing (24h aggregation: max value of previous day)
    #{'mx2t': 'max_t(TMAX-K:ERA5:5021:1:0:1:0/24h/0h)'}, # Maximum temperature
    #{'mn2t': 'min_t(TMIN-K:ERA5:5021:1:0:1:0/24h/0h)'}, # Minimum temperature

]

source='desm.harvesterseasons.com:8080' # server for timeseries query
y_start,y_end='2000','2020'

# static parameters 
for pardict in statics:
    hour='00'
    start=y_start+'0101T000000Z'
    end=y_end+'1231T000000Z'
    key,value=list(pardict.items())[0]
    df=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
    print(key)
    print(df)
    for point in llpdict.keys():
        dfpoint = df[df['pointID'] == point].reset_index(drop=True)
        dfpoint.to_csv(era5_dir+'era5-'+key+'_'+start+'-'+end+'_'+str(point)+'.csv',index=False)
        if key=='z': # save utctime, lat, lon, pointID for each station to csv
            dfpoint[['utctime','latitude','longitude','pointID']].to_csv(era5_dir+'era5-utctime-lat-lon-pointID-_'+start+'-'+end+'_'+str(point)+'.csv',index=False)

# 00 and 12 parameters
for pardict in predictors_0012:
    hour='00'
    start=y_start+'0101T000000Z'
    end=y_end+'1231T000000Z'
    key,value=list(pardict.items())[0]
    df=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
    print(key)
    print(df)
    df.rename({key:key+'-00'}, axis=1, inplace=True)
    #print(df)
    for point in llpdict.keys():
        dfpoint = df[df['pointID'] == point].reset_index(drop=True)
        dfpoint.to_csv(era5_dir+'era5-'+key+'-00_'+start+'-'+end+'_'+str(point)+'.csv',index=False)
    hour='12'
    start=y_start+'0101T000000Z'
    end=y_end+'1231T000000Z' 
    df=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
    df.rename({key:key+'-12'}, axis=1, inplace=True)
    df['utctime']=df['utctime'].dt.date
    print(df)
    for point in llpdict.keys():
        dfpoint = df[df['pointID'] == point].reset_index(drop=True)
        dfpoint.to_csv(era5_dir+'era5-'+key+'-12_'+start+'-'+end+'_'+str(point)+'.csv',index=False)

# 00 parameters
for pardict in predictors_00:
    hour='00'
    start=y_start+'0101T000000Z'
    end=y_end+'1231T000000Z'
    key,value=list(pardict.items())[0]
    df=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
    print(key)
    print(df)
    for point in llpdict.keys():
        dfpoint = df[df['pointID'] == point].reset_index(drop=True)
        dfpoint.to_csv(era5_dir+'era5-'+key+'_'+start+'-'+end+'_'+str(point)+'.csv',index=False)

# previous day 24h sums
# previous day maximum or minimum

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))