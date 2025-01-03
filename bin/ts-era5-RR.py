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

eobsinfo='/home/ubuntu/data/ML/training-data/EOBS_orography_EU_5441.csv'
eobs_df=pd.read_csv(eobsinfo)

lat=eobs_df['LAT'].values.tolist()
lon=eobs_df['LON'].values.tolist()
point_ids=eobs_df['STAID'].values.tolist()
points = [f"{point_id:06d}" for point_id in point_ids]

llpdict={i:[j, k] for i, j, k in zip(points,lat, lon)}
llpdict={k:llpdict[k] for k in list(llpdict.keys())[:4000]}

y_start,y_end='2000','2020'

# static predictors
statics = [
    #{'z': 'Z-M2S2:ERA5:5021:1:0:1:0'}, # geopotential in m2 s-2
    #{'lsm': 'LC-0TO1:ERA5:5021:1:0:1:0'}, # Land sea mask: 1=land, 0=sea
    #{'sdor': 'SDOR-M:ERA5:5021:1:0:1:0'}, # Standard deviation of orography
    #{'slor': 'SLOR:ERA5:5021:1:0:1:0'}, # Slope of sub-gridscale orography
    #{'anor': 'ANOR-RAD:ERA5:5021:1:0:1:0'}, # Angle of sub-gridscale orography
]

# 00 and 12 UTC predictors
predictors_0012 = [
    #{'u10':'U10-MS:ERA5:5021:1:0:1:0'}, # 10m u-component of wind (6h instantanous)
    {'v10':'V10-MS:ERA5:5021:1:0:1:0'}, # 10m v-component of wind (6h instantanous)
    {'td2':'TD2-K:ERA5:5021:1:0:1:0'}, # 2m dewpoint temperature (6h instantanous)
    {'t2':'T2-K:ERA5:5021:1:0:1:0'}, # 2m temperature (6h instantanous)
    {'msl':'PSEA-HPA:ERA5:5021:1:0:1:0'}, # mean sea level pressure (6h instantanous)
    #{'tsea':'TSEA-K:ERA5:5021:1:0:1'}, # sea surface temperature (6h instantanous)
    {'tcc':'N-0TO1:ERA5:5021:1:0:1:0'}, # total cloud cover (6h instantanous)
    {'kx': 'KX:ERA5:5021:1:0:0'}, # K index
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
    {'z850': 'Z-M2S2:ERA5:5021:2:850:1:0'}, # geopotential in m2 s-2
    {'z700': 'Z-M2S2:ERA5:5021:2:700:1:0'},
    {'z500': 'Z-M2S2:ERA5:5021:2:500:1:0'},    
    ]

# 00 predictors 
predictors_00 = [
    {'tclw':'TCLW-KGM2:ERA5:5021:1:0:1:0'}, # total column cloud liquid water (24h instantanous) 
    {'tcwv':'TOTCWV-KGM2:ERA5:5021:1:0:1:0'}, # total column water vapor here
    {'swvl1':'SOILWET-M3M3:ERA5:5021:9:7:1:0'}, # volumetric soil water layer 1 (0-7cm) (24h instantanous)
    {'swvl2':'SWVL2-M3M3:ERA5:5021:9:1820:1:0'}, # volumetric soil water layer 2 (7-28cm) (24h instantanous)
    {'swvl3':'SWVL3-M3M3:ERA5:5021:9:7268:1:0'}, # volumetric soil water layer 3 (28-100cm) (24h instantanous)
    {'swvl4':'SWVL4-M3M3:ERA5:5021:9:25855:1:0'} # volumetric soil water layer 4 (100-289cm) (24h instantanous)
]

# previous day 24h sums 
predictors_24hAgg = [
    {'ewss':'sum_t(EWSS-NM2S:ERA5:5021:1:0:1:0/24h/0h)'}, # eastward turbulent surface stress (24h aggregation since beginning of forecast)
    {'e':'sum_t(EVAP-M:ERA5:5021:1:0:1:0/24h/0h)'}, # evaporation (24h aggregation since beginning of forecast)
    {'nsss':'sum_t(NSSS-NM2S:ERA5:5021:1:0:1:0/24h/0h)'}, # northward turbulent surface stress (24h aggregation since beginning of forecast)
    {'slhf':'sum_t(FLLAT-JM2:ERA5:5021:1:0:1:0/24h/0h)'}, # surface latent heat flux (24h aggregation since beginning of forecast)
    {'ssr':'sum_t(RNETSWA-JM2:ERA5:5021:1:0:1:0/24h/0h)'}, # surface net solar radiation (24h aggregation since beginning of forecast)
    {'str':'sum_t(RNETLWA-JM2:ERA5:5021:1:0:1:0/24h/0h)'}, # surface net thermal radiation (24h aggregation since beginning of forecast)
    {'sshf':'sum_t(FLSEN-JM2:ERA5:5021:1:0:1:0/24h/0h)'}, # surface sensible heat flux (24h aggregation since beginning of forecast)
    {'ssrd':'sum_t(RADGLOA-JM2:ERA5:5021:1:0:1:0/24h/0h)'}, # surface solar radiation downwards (24h aggregation since beginning of forecast)
    {'strd':'sum_t(RADLWA-JM2:ERA5:5021:1:0:1:0/24h/0h)'}, # surface thermal radiation downwards (24h aggregation since beginning of forecast)
    {'tp':'sum_t(RR-M:ERA5:5021:1:0:1:0/24h/0h)'}, # total precipitation (24h aggregation since beginning of forecast)
    {'ttr':'sum_t(RTOPLWA-JM2:ERA5:5021:1:0:1:0/24h/0h)'}, # top net thermal radiation (24h aggregation since beginning of forecast)
    {'fg10':'max_t(FFG-MS:ERA5:5021:1:0:1:0/24h/0h)'}, # 10m wind gust since previous post-processing (24h aggregation: max value of previous day)
    {'mx2t': 'max_t(TMAX-K:ERA5:5021:1:0:1:0/24h/0h)'}, # Maximum temperature in the last 24h
    {'mn2t': 'min_t(TMIN-K:ERA5:5021:1:0:1:0/24h/0h)'} # Minimum temperature in the last 24h
]

source='desm.harvesterseasons.com:8080' # server for timeseries query

# static parameters 
for pardict in statics:
    hour='00'
    key,value=list(pardict.items())[0]
    print(key)
    dir=key+'/'
    df_all_years=pd.DataFrame()

    for year in range(int(y_start),int(y_end)+1):
        start=str(year)+'0101T000000Z'
        end=str(year)+'1231T000000Z'
        df_year=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
        df_all_years = pd.concat([df_all_years, df_year], ignore_index=True)
        print(df_all_years)

    for point in llpdict.keys():
        dfpoint = df_all_years[df_all_years['pointID'] == point].reset_index(drop=True)
        dfpoint.to_csv(era5_dir+dir+'era5-'+key+'_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)
        if key=='z': # save utctime, lat, lon, pointID for each station to csv
            dfpoint[['utctime','latitude','longitude','pointID']].to_csv(era5_dir+'locationInfo/era5-utctime-lat-lon-pointID-_'+start+'-'+end+'_'+str(point)+'.csv',index=False)


# 00 and 12 parameters
for pardict in predictors_0012:
    key,value=list(pardict.items())[0]
    dir1=key+'_00/'
    dir2=key+'_12/'
    
    hour='00'
    df_all_years=pd.DataFrame()
    for year in range(int(y_start),int(y_end)+1):
        start=str(year)+'0101T000000Z'
        end=str(year)+'1231T000000Z'
        df_year=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
        df_year.rename({key:key+'-00'}, axis=1, inplace=True)
        df_all_years = pd.concat([df_all_years, df_year], ignore_index=True)
        print(df_all_years)
    for point in llpdict.keys():
        dfpoint = df_all_years[df_all_years['pointID'] == point].reset_index(drop=True)
        dfpoint.to_csv(era5_dir+dir1+'era5-'+key+'-00_'+start+'-'+end+'_'+str(point)+'.csv',index=False)
    
    hour='12'
    df_all_years=pd.DataFrame()
    for year in range(int(y_start),int(y_end)+1):
        start=str(year)+'0101T000000Z'
        end=str(year)+'1231T000000Z'
        df_year=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
        df_year.rename({key:key+'-12'}, axis=1, inplace=True)
        df_all_years = pd.concat([df_all_years, df_year], ignore_index=True)
        print(df_all_years)
    for point in llpdict.keys():
        dfpoint = df_all_years[df_all_years['pointID'] == point].reset_index(drop=True)
        dfpoint.to_csv(era5_dir+dir2+'era5-'+key+'-12_'+start+'-'+end+'_'+str(point)+'.csv',index=False)

# 00 parameters
for pardict in predictors_00:
    hour='00'
    key,value=list(pardict.items())[0]
    print(key)
    dir=key+'/'
    df_all_years=pd.DataFrame()
    for year in range(int(y_start),int(y_end)+1):
        start=str(year)+'0101T000000Z'
        end=str(year)+'1231T000000Z'
        df_year=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
        df_all_years = pd.concat([df_all_years, df_year], ignore_index=True)
        print(df_all_years)
    for point in llpdict.keys():
        dfpoint = df_all_years[df_all_years['pointID'] == point].reset_index(drop=True)
        dfpoint.to_csv(era5_dir+dir+'era5-'+key+'_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)


# previous day 24h sums
for pardict in predictors_24hAgg:
    hour='00'
    key,value=list(pardict.items())[0]
    print(key)
    df_all_years=pd.DataFrame()
        
    for year in range(int(y_start),int(y_end)+1):
        start=str(year)+'0101T000000Z'
        end=str(year)+'1231T000000Z'
        df_year=fcts.smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
        df_all_years = pd.concat([df_all_years, df_year], ignore_index=True)
        print(df_all_years)
    
    for point in llpdict.keys():
        dfpoint = df_all_years[df_all_years['pointID'] == point].reset_index(drop=True)
        dfpoint.to_csv(era5_dir+'era5-'+key+'_'+y_start+'-'+y_end+'_'+str(point)+'.csv',index=False)

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))