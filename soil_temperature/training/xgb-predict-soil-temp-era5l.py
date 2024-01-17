import xarray as xr
import cfgrib,time,sys
import pandas as pd
import dask.distributed
import numpy as np
import xgboost as xgb
startTime=time.time()
# Prediction for swi2
pd.set_option('mode.chained_assignment', None) # turn off SettingWithCopyWarning 
if __name__ == "__main__":
        cluster=dask.distributed.LocalCluster()
        client=dask.distributed.Client(cluster)
        #mod_dir='/home/ubuntu/data/ML/models/soilwater/' # saved mdl
        #data_dir='/home/ubuntu/data/ens/'

        # input files 
        swvls_ecsf=sys.argv[1] # vsw (swvl layers)
        sl00_ecsf=sys.argv[2] # d2m,t2m,rsn,sde,stl1,u10,v10,laihv,lailv
        sl_runsum=sys.argv[3] # 15-day runnins sums for disaccumulated tp,e,ro
        sl_disacc=sys.argv[4] # disaccumulated tp,e,slhf,sshf,ro,str,strd,ssr,ssrd
        laihv=sys.argv[5]
        lailv=sys.argv[6]
        sde_day=sys.argv[7]
        sktd_day=sys.argv[8]
        rsn_day=sys.argv[9]
        stl1_day=sys.argv[10]
        swvl2_day=sys.argv[11]
        u10_day=sys.argv[12]
        v10_day=sys.argv[13]
        t2_day=sys.argv[14]
        td2_day=sys.argv[15]
        sktn_night=sys.argv[16]
        # swi2c=sys.argv[7] # swi2clim
        dtm_aspect='grib/COPERNICUS_20000101T000000_20110701T000000_anor-dtm-aspect-avg_eu-era5l.grb' 
        dtm_slope='grib/COPERNICUS_20000101T000000_20110701T000000_slor-dtm-slope-avg_eu-era5l.grb'
        dtm_height='grib/COPERNICUS_20000101T000000_20110701T000000_h-dtm-height-avg_eu-era5l.grb'
        # soilgrids='grib/SG_20200501T000000_soilgrids-0-200cm-eu-era5l.grib' # sand ssfr, silt soilp, clay scfr, soc stf
        # lakecov='grib/ECC_20000101T000000_ilwaterc-frac-eu-9km-fix.grib' # lake cover
        # urbancov='grib/ECC_20000101T000000_urbanc-frac-eu-9km-fix.grib' # urban cover
        # highveg='grib/ECC_20000101T000000_hveg-frac-eu-9km-fix.grib' # high vegetation cover
        # lowveg='grib/ECC_20000101T000000_lveg-frac-eu-9km-fix.grib' # low vegetation cover 
        # lakedepth='grib/ECC_20000101T000000_ilwater-depth-eu-9km-fix.grib' # lake depth
        # landcov='grib/ECC_20000101T000000_lc-frac-eu-9km-fix.grib' # land cover
        # soiltype='grib/ECC_20000101T000000_soiltype-eu-9km-fix.grib' # soil type
        # typehv='grib/ECC_20000101T000000_hveg-type-eu-9km-fix.grib' # type of high vegetation
        # typelv='grib/ECC_20000101T000000_lveg-type-eu-9km-fix.grib' # type of low vegetation 
        # output file
        outfile=sys.argv[17]

        # read in data
        sl_UTC00_var = ['u10','v10','stl1','rsn','sde','stl1','laihv','lailv','t2','td2'] 
        names00UTC={'d2m':'td2-00','t2m':'t2-00','rsn':'rsn-00','sde':'sd-00','stl1':'stl1-00','u10':'u10-00','v10':'v10-00',
                    'laihv':'laihv-00','lailv':'lailv-00'}
        dtm_var=['p3008','slor','anor']
        # sg_var=['clay_0-5cm','sand_0-5cm','silt_0-5cm','soc_0-5cm',
        # 'clay_5-15cm','sand_5-15cm','silt_5-15cm','soc_5-15cm',
        # 'clay_15-30cm','sand_15-30cm','silt_15-30cm','soc_15-30cm']
        sl_disacc_var=['tp','e','slhf','sshf','ro','str','ssr','ssrd','strd']
        sl_runsum_var=['tp','e','ro','sro','ssro']
        namesRS={'tp':'tp5d','e':'evapp5d','ro':'ro5d','sro':'sro5d','ssro':'ssro5d'}
        ecc_var=['cl','cvh','cvl','dl','lsm','slt','tvh','tvl','cur']
        swvls=xr.open_dataset(swvls_ecsf, engine='cfgrib', chunks={'valid_time':1},
                        backend_kwargs=dict(time_dims=('valid_time','verifying_time'),indexpath=''))
        sl00=xr.open_dataset(sl00_ecsf, engine='cfgrib',  chunks={'valid_time':1},
                        backend_kwargs=dict(time_dims=('valid_time','verifying_time'),indexpath=''))[sl_UTC00_var].rename_vars(names00UTC)
        height=xr.open_dataset(dtm_height, engine='cfgrib', chunks={'valid_time':1}, 
                        backend_kwargs=dict(time_dims=('valid_time','verifying_time'),indexpath='')).rename_vars({'p3008':'DTM_height'})
        slope=xr.open_dataset(dtm_slope, engine='cfgrib',  chunks={'valid_time':1},
                        backend_kwargs=dict(time_dims=('valid_time','verifying_time'),indexpath='')).rename_vars({'slor':'DTM_slope'})
        aspect=xr.open_dataset(dtm_aspect, engine='cfgrib',  chunks={'valid_time':1},
                        backend_kwargs=dict(time_dims=('valid_time','verifying_time'),indexpath='')).rename_vars({'anor':'DTM_aspect'})
        sldisacc=xr.open_dataset(sl_disacc, engine='cfgrib', chunks={'valid_time':1},
                        backend_kwargs=dict(time_dims=('valid_time','verifying_time'),indexpath=''))[sl_disacc_var].rename_vars({'e':'evapp'})
        slrunsum=xr.open_dataset(sl_runsum, engine='cfgrib', chunks={'valid_time':1},
                        backend_kwargs=dict(time_dims=('valid_time','verifying_time'),indexpath=''))[sl_runsum_var].rename_vars(namesRS)
        laihv_ds=xr.open_dataset(laihv, engine='cfgrib', chunks={'valid_time':1},
                        backend_kwargs=dict(time_dims=('valid_time','verifying_time'),indexpath='')).rename_vars({'lai_hv':'laihv-12'})
        lailv_ds=xr.open_dataset(lailv, engine='cfgrib', chunks={'valid_time':1},
                        backend_kwargs=dict(time_dims=('valid_time','verifying_time'),indexpath='')).rename_vars({'lai_lv':'lailv-12'})
        #extra added
        sde_ds=xr.open_dataset(sde_day, engine='cfgrib', chunks={'valid_time':1},
                        backend_kwargs=dict(time_dims=('valid_time','verifying_time'),indexpath='')).rename_vars({'sde':'sde-12'})
        sktd_ds=xr.open_dataset(sktd_day, engine='cfgrib', chunks={'valid_time':1},
                        backend_kwargs=dict(time_dims=('valid_time','verifying_time'),indexpath='')).rename_vars({'skt':'sktd-12'})
        rsn_ds=xr.open_dataset(rsn_day, engine='cfgrib', chunks={'valid_time':1},
                        backend_kwargs=dict(time_dims=('valid_time','verifying_time'),indexpath='')).rename_vars({'rsn':'rsn-12'})
        stl1_ds=xr.open_dataset(stl1_day, engine='cfgrib', chunks={'valid_time':1},
                        backend_kwargs=dict(time_dims=('valid_time','verifying_time'),indexpath='')).rename_vars({'stl1':'stl1-12'})
        swvl2_ds=xr.open_dataset(swvl2_day, engine='cfgrib', chunks={'valid_time':1},
                        backend_kwargs=dict(time_dims=('valid_time','verifying_time'),indexpath='')).rename_vars({'vsw':'swvl2-12'})
        u10_ds=xr.open_dataset(u10_day, engine='cfgrib', chunks={'valid_time':1},
                        backend_kwargs=dict(time_dims=('valid_time','verifying_time'),indexpath='')).rename_vars({'10u':'u10-12'})
        v10_ds=xr.open_dataset(v10_day, engine='cfgrib', chunks={'valid_time':1},
                        backend_kwargs=dict(time_dims=('valid_time','verifying_time'),indexpath='')).rename_vars({'10v':'v10-12'})
        t2_ds=xr.open_dataset(t2_day, engine='cfgrib', chunks={'valid_time':1},
                        backend_kwargs=dict(time_dims=('valid_time','verifying_time'),indexpath='')).rename_vars({'2t':'t2-12'})
        td2_ds=xr.open_dataset(td2_day, engine='cfgrib', chunks={'valid_time':1},
                        backend_kwargs=dict(time_dims=('valid_time','verifying_time'),indexpath='')).rename_vars({'2d':'td2-12'})
        sktn_ds=xr.open_dataset(sktn_night, engine='cfgrib', chunks={'valid_time':1},
                        backend_kwargs=dict(time_dims=('valid_time','verifying_time'),indexpath='')).rename_vars({'skt':'sktn-00'})      
        # swi2clim=xr.open_dataset(swi2c, engine='cfgrib', chunks={'valid_time':1},
        #                 backend_kwargs=dict(time_dims=('valid_time','verifying_time'),indexpath=''))['swi2'].to_dataset().rename_vars({'swi2':'swi2clim'})
        date_first=str(swvls.valid_time[0].data)[:10]
        date_last=str(swvls.valid_time[-1].data)[:10]        
        # swi2clim=swi2clim.sel(valid_time=slice(date_first,date_last))
        laihv_ds=laihv_ds.sel(valid_time=slice(date_first,date_last))
        lailv_ds=lailv_ds.sel(valid_time=slice(date_first,date_last))
        swvls=swvls.where((swvls.depthBelowLandLayer<=0.20) & (swvls.depthBelowLandLayer>=0.06), drop=True).rename_vars({'vsw':'swvl2-00'}).squeeze(["depthBelowLandLayer"], drop=True) # use layer 0.07 m for swvl2
        swvls['dayOfYear']=swvls.valid_time.dt.dayofyear
        # soilg_ds5=soilg_ds.where((soilg_ds.depthBelowLand<=0.10), drop=True).rename_vars(namesSG5).squeeze(["depthBelowLand"], drop=True) # use layers 0-30cm for swvl2
        # soilg_ds15=soilg_ds.where((soilg_ds.depthBelowLand<=0.20) & (soilg_ds.depthBelowLand>=0.10), drop=True).rename_vars(namesSG15).squeeze(["depthBelowLand"], drop=True) # use layers 0-30cm for swvl2
        # soilg_ds30=soilg_ds.where((soilg_ds.depthBelowLand<=0.40) & (soilg_ds.depthBelowLand>=0.20), drop=True).rename_vars(namesSG30).squeeze(["depthBelowLand"], drop=True) # use layers 0-30cm for swvl2
        ds1=xr.merge([swvls,sl00,height,slope,aspect,#soilg_ds5,soilg_ds15,soilg_ds30,
                      #lakecov_ds,hvc_ds,hlc_ds,lakedepth_ds,landcov_ds,soilty_ds,tvh_ds,tvl_ds,ecc_ucov,
                      sldisacc,slrunsum,laihv_ds,lailv_ds,#swi2clim
                      ],compat='override')
        ds1=ds1.drop_vars(['number','surface','depthBelowLandLayer'])
        ds1=ds1.sel(valid_time=slice(date_first,date_last))
        
        df=ds1.to_dataframe() # pandas
        #ds1=ds1.unify_chunks() # dask
        #df=ds1.to_dask_dataframe()[preds] #dask
        #print(df)
        df=df.reset_index() # pandas
        
        # store grid for final result
        df_grid=df[['valid_time','latitude','longitude']]
        df_grid['clim_ts_value'] = np.nan
        df_grid=df_grid.set_index(['valid_time', 'latitude','longitude'])
        
        df=df.dropna()
        preds=['slhf','sshf','ssrd','strd','str','ssr','laihv-00','laihv-12','lailv-00','lailv-12','sd-00','sd-12','sktn','sktd-12',
          'rsn-00','rsn-12','stl1-00','stl1-12','swvl2-00','swvl2-12','t2-00',
          't2-12','td2-00','td2-12','u10-00','u10-12','v10-00','v10-12','ro5d',
          'sro5d','ssro5d','evapp5d','tp5d','ro',
          'longitude','latitude','DTM_height','DTM_slope','DTM_aspect','dayOfYear']
        df_preds = df[preds]
        soilcols=['valid_time','latitude','longitude']
        df=df[soilcols]
        
        #print(df.compute()) # dask
        
        ### Predict with XGBoost fitted model 
        mdl_name='MLmodels/mdl_soiltemp_2015-2022_2000points.txt'
        fitted_mdl=xgb.XGBRegressor() # pandas
        #fitted_mdl=xgb.Booster() # dask
        fitted_mdl.load_model(mdl_name)

        #print('start fit')
        result=fitted_mdl.predict(df_preds) # pandas
        #result = xgb.dask.predict(client, fitted_mdl, df) # dask, super slow! don't even know how long to execute
        #print('end fit')
        df_preds=[]

        # result df to ds, and nc file as output
        df['clim_ts_value']=result.tolist()
        df=df.set_index(['valid_time', 'latitude','longitude'])
        #print(df)
        result=df_grid.fillna(df)
        #print(result)
        ds=result.to_xarray()
        #print(ds)
        nc=ds.to_netcdf(outfile)
        
        executionTime=(time.time()-startTime)
        print('Fitting execution time per member in minutes: %.2f'%(executionTime/60))
