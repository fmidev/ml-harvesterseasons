#!/usr/bin/env python3
import time, warnings,requests,json,os,warnings
import pandas as pd
import numpy as np  # Add this import for numpy
warnings.simplefilter(action='ignore', category=FutureWarning)
# SmarMet-server timeseries query to fetch ERA5 training data for precipitation ML
# remember to: conda activate xgb 

startTime=time.time()

def smartmet_ts_query_multiplePointsByID_hour(source,start,end,h,pardict,llpdict):
    ''' timeseries query to smartmet-server
    start date, end date, hour, list of lats&lons, parameters as dictionary (colname:smartmet-fmikey)
    returns dataframe
    '''
    latlons=[]
    for pair in llpdict.values():
        for i in pair.values():
            latlons.append(i)
    
    # Timeseries query
    query='http://'+source+'/timeseries?latlon='
    for nro in latlons:
        query+=str(nro)+','
    query=query[0:-1]
    query+='&param=utctime,latitude,longitude,'
    for par in pardict.values():
        query+=par+','
    query=query[0:-1]
    query+='&starttime='+start+'&endtime='+end+'&hour='+h+'&format=json&precision=full&tz=utc&timeformat=sql&grouplocations=1'
    print(query)

    # Response to dataframe
    response=requests.get(url=query)
    results_json=json.loads(response.content)
    #print(results_json)
    for i in range(len(results_json)):
        res1=results_json[i]
        for key,val in res1.items():
            if key!='utctime':   
                res1[key]=val.strip('[]').split()
    df=pd.DataFrame(results_json)  
    df.columns = ['utctime', 'latitude', 'longitude'] + list(pardict.keys())  # Correct column headers
    df['utctime']=pd.to_datetime(df['utctime'])
    expl_cols=['latitude','longitude']+list(pardict.keys())
    df=df.explode(expl_cols)
    #print(df)
    return df

#data_dir='/home/ubuntu/data/ML/training-data/RR/'
data_dir='/home/ubuntu/data/ML/training-data/RR-fin/'
era5_dir=data_dir+'era5/' 

#eobsinfo='/home/ubuntu/data/ML/training-data/EOBS_orography_EU_5441.csv'
eobsinfo='/home/ubuntu/data/ML/training-data/RR-fin/DEM_10000_training_LUCASlocs-fix.csv'
eobs_df = pd.read_csv(eobsinfo)
llpdict = eobs_df.set_index('POINT_ID')[['TH_LAT', 'TH_LONG']].to_dict(orient='index')
llpdict = dict(list(llpdict.items())[6000:6500])
reverse_lookup = {
    (v['TH_LAT'], v['TH_LONG']): k
    for k, v in llpdict.items()
}
#print(llpdict)
y_start,y_end='2000','2024'

# static and 00 utc predictors
predictors_00 = [
    {'z': 'Z-M2S2:ERA5:5021:1:0:1:0'}, # geopotential in m2 s-2
    {'lsm': 'LC-0TO1:ERA5:5021:1:0:1:0'}, # Land sea mask: 1=land, 0=sea
    {'sdor': 'SDOR-M:ERA5:5021:1:0:1:0'}, # Standard deviation of orography
    {'slor': 'SLOR:ERA5:5021:1:0:1:0'}, # Slope of sub-gridscale orography
    {'anor': 'ANOR-RAD:ERA5:5021:1:0:1:0'}, # Angle of sub-gridscale orography
    {'tclw':'TCLW-KGM2:ERA5:5021:1:0:1:0'}, # total column cloud liquid water (24h instantanous) 
    {'tcwv':'TOTCWV-KGM2:ERA5:5021:1:0:1:0'}, # total column water vapor here
    {'swvl1':'SOILWET-M3M3:ERA5:5021:9:7:1:0'}, # volumetric soil water layer 1 (0-7cm) (24h instantanous)
    {'swvl2':'SWVL2-M3M3:ERA5:5021:9:1820:1:0'}, # volumetric soil water layer 2 (7-28cm) (24h instantanous)
    {'swvl3':'SWVL3-M3M3:ERA5:5021:9:7268:1:0'}, # volumetric soil water layer 3 (28-100cm) (24h instantanous)
    {'swvl4':'SWVL4-M3M3:ERA5:5021:9:25855:1:0'} # volumetric soil water layer 4 (100-289cm) (24h instantanous)
]

# 00 and 12 UTC predictors
predictors_0012 = [
    {'u10':'U10-MS:ERA5:5021:1:0:1:0'}, # 10m u-component of wind (6h instantanous)
    {'v10':'V10-MS:ERA5:5021:1:0:1:0'}, # 10m v-component of wind (6h instantanous)
    {'td2':'TD2-K:ERA5:5021:1:0:1:0'}, # 2m dewpoint temperature (6h instantanous)
    {'t2':'T2-K:ERA5:5021:1:0:1:0'}, # 2m temperature (6h instantanous)
    {'msl':'PSEA-HPA:ERA5:5021:1:0:1:0'}, # mean sea level pressure (6h instantanous)
    {'tsea':'TSEA-K:ERA5:5021:1:0:1'}, # sea surface temperature (6h instantanous)
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

# era5d predictors
predictors_24hAgg = [    
    {'ewss':'EWSS-NM2S:ERA5D:5021:1:0:1'}, # Previous day sum Eastward turbulent surface stress
    {'e':'EVAP-M:ERA5D:5021:1:0:1'}, # Previous day sum Evaporation
    {'nsss':'NSSS-NM2S:ERA5D:5021:1:0:1'}, # Previous day sum Northward turbulent surface stress
    {'slhf':'FLLAT-JM2:ERA5D:5021:1:0:0'}, # Previous day sum Surface latent heat flux
    {'ssr':'RNETSWA-JM2:ERA5D:5021:1:0:0'}, # Previous day sum Surface net solar radiation
    {'str':'RNETLWA-JM2:ERA5D:5021:1:0:0'}, # Previous day sum Surface net thermal radiation
    {'sshf':'FLSEN-JM2:ERA5D:5021:1:0:0'}, # Previous day sum Surface sensible heat flux
    {'ssrd':'RADGLOA-JM2:ERA5D:5021:1:0:0'}, # Previous day sum Surface solar radiation downwards
    {'strd':'RADLWA-JM2:ERA5D:5021:1:0:0'}, # Previous day sum Surface thermal radiation downwards
    {'tp':'RR-M:ERA5D:5021:1:0:1'}, # Previous day sum Total precipitation
    {'ttr':'RTOPLWA-JM2:ERA5D:5021:1:0:1'}, # Previous day sum Top net thermal radiation
    {'fg10':'FFG-MS:ERA5D:5021:1:0:1'}, # Previous day maximum 10m wind gust since previous post-processing (24h aggregation: max value of previous day)
    {'mx2t':'TMAX-K:ERA5D:5021:1:0:1'}, # Previous day maximum Maximum temperature in the last 24h
    {'mn2t':'TMIN-K:ERA5D:5021:1:0:1'} # Previous day minimum Minimum temperature in the last 24h
]

source='desm.harvesterseasons.com:8080' # server for timeseries query

# 00 parameters
for pardict in predictors_00:
    hour='00'
    key,value=list(pardict.items())[0]
    print(key)
    dir=key+'/'
    if not os.path.exists(f'{era5_dir}{dir}'):
        os.makedirs(f'{era5_dir}{dir}')
    df_all_years=pd.DataFrame()
    for year in range(int(y_start),int(y_end)+1):
        start=str(year)+'0101T000000Z'
        end=str(year)+'1231T000000Z'
        df_year=smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
        df_all_years = pd.concat([df_all_years, df_year], ignore_index=True)
        print(df_all_years)
    
    # group by location
    grouped = df_all_years.groupby(['latitude', 'longitude'])
    for (lat, lon), group in grouped:
        point_id = reverse_lookup.get((float(lat), float(lon)),f"lat{float(lat):.2f}_lon{float(lon):.2f}")
        filename = f'{era5_dir}{dir}era5-{key}_{y_start}-{y_end}_{str(int(point_id))}.csv'
        group.to_csv(filename, index=False)

# 00 and 12 parameters
for pardict in predictors_0012:
    key,value=list(pardict.items())[0]
    
    dir1=key+'_00/'
    dir2=key+'_12/'
    if not os.path.exists(f'{era5_dir}{dir1}'):
        os.makedirs(f'{era5_dir}{dir1}')
    if not os.path.exists(f'{era5_dir}{dir2}'):
        os.makedirs(f'{era5_dir}{dir2}')    
    
    hour='00'
    df_all_years=pd.DataFrame()
    for year in range(int(y_start),int(y_end)+1):
        start=str(year)+'0101T000000Z'
        end=str(year)+'1231T000000Z'
        df_year=smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
        df_year.rename({key:key+'-00'}, axis=1, inplace=True)
        df_all_years = pd.concat([df_all_years, df_year], ignore_index=True)
        print(df_all_years)
    
    # group by location
    grouped = df_all_years.groupby(['latitude', 'longitude'])
    for (lat, lon), group in grouped:
        point_id = reverse_lookup.get((float(lat), float(lon)),f"lat{float(lat):.2f}_lon{float(lon):.2f}")
        filename = f'{era5_dir}{dir1}era5-{key}-00_{y_start}-{y_end}_{str(int(point_id))}.csv'
        group.to_csv(filename, index=False)
    hour='12'
    df_all_years=pd.DataFrame()
    for year in range(int(y_start),int(y_end)+1):
        start=str(year)+'0101T000000Z'
        end=str(year)+'1231T000000Z'
        df_year=smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
        df_year.rename({key:key+'-12'}, axis=1, inplace=True)
        df_all_years = pd.concat([df_all_years, df_year], ignore_index=True)
        print(df_all_years)
    
    # group by location
    grouped = df_all_years.groupby(['latitude', 'longitude'])
    for (lat, lon), group in grouped:
        point_id = reverse_lookup.get((float(lat), float(lon)),f"lat{float(lat):.2f}_lon{float(lon):.2f}")
        filename = f'{era5_dir}{dir2}era5-{key}-12_{y_start}-{y_end}_{str(int(point_id))}.csv'
        group.to_csv(filename, index=False)

# previous day 24h sums
for pardict in predictors_24hAgg:
    hour='00'
    key,value=list(pardict.items())[0]
    dir=key+'/'
    if not os.path.exists(f'{era5_dir}{dir}'):
        os.makedirs(f'{era5_dir}{dir}')

    print(key)
    df_all_years=pd.DataFrame()
        
    for year in range(int(y_start),int(y_end)+1):
        start=str(year)+'0101T000000Z'
        end=str(year)+'1231T000000Z'
        df_year=smartmet_ts_query_multiplePointsByID_hour(source,start,end,hour,pardict,llpdict)
        df_all_years = pd.concat([df_all_years, df_year], ignore_index=True)
        print(df_all_years)
    
    # group by location
    grouped = df_all_years.groupby(['latitude', 'longitude'])
    for (lat, lon), group in grouped:
        point_id = reverse_lookup.get((float(lat), float(lon)),f"lat{float(lat):.2f}_lon{float(lon):.2f}")
        filename = f'{era5_dir}{dir}era5-{key}_{y_start}-{y_end}_{str(int(point_id))}.csv'
        group.to_csv(filename, index=False)

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))