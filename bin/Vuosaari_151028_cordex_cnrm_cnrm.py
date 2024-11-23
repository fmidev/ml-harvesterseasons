bbox='24.9459,60.45867,25.4459,59.95867'
harbor='Vuosaari'
FMISID='151028'
lat=60.20867
lon=25.1959
pred='WG_PT24H_MAX'
qpred='max_t('+pred+'/24h/0h)'
start='20130701T000000Z'
end='20240101T000000Z' 
starty=start[0:4]
endy=end[0:4]

fname = 'training_data_oceanids_cnrm-cnrm-Vuosaari-cordex.csv' # training input data file
obsfile='obs-oceanids-'+start+'-'+end+'-all-'+harbor+'-cnrm-cnrm-daymean-daymax.csv'
fscorepic='Fscore_'+pred+'-cnrm-cnrm-'+harbor+'.png'
mdl_name='mdl_WGPT24MAX_2013-2023_cnrm-cnrm.txt'
xgbstudy='xgb-'+pred+'-Vuosaari-cnrm-cnrm'

test_y=[2019,2024]
train_y=[2013, 2014, 2015, 2016, 2017, 2018, 2020, 2021, 2022, 2023]

#droplist=['utctime',]
cols_own=['utctime',
          #'lat-2','lon-2','lat-3','lon-3','lat-4','lon-4','lat-1','lon-1',
          'pr-1','pr-2','pr-3','pr-4',
          'sfcWind-1','sfcWind-2','sfcWind-3','sfcWind-4',
          'tasmax-1','tasmax-2','tasmax-3','tasmax-4',
          'tasmin-1','tasmin-2','tasmin-3','tasmin-4',
          'maxWind-1','maxWind-2','maxWind-3','maxWind-4',
          'dayofyear',pred]
