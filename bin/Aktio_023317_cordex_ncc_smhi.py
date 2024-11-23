bbox='20.0,38.0,21.5,39.5'
harbor='Aktio'
FMISID='023317'
lat=38.92
lon=20.77
pred='WS_PT24H_AVG'
qpred='max_t('+pred+'/24h/0h)'
start='20060101T000000Z'
end='20201231T000000Z' 
starty=start[0:4]
endy=end[0:4]

fname = 'training_data_oceanids_ncc-dmi-Aktio-cordex.csv' # training input data file
obsfile='obs-oceanids-'+start+'-'+end+'-all-'+harbor+'-ncc-smhi-daymean-daymax.csv'
fscorepic='Fscore_'+pred+'-ncc-smhi-'+harbor+'.png'
mdl_name='mdl_WGPT24MAX_2013-2023_ncc-smhi.txt'
xgbstudy='xgb-'+pred+'-Aktio-ncc-smhi'

test_y=[2014,2017,2020]
train_y=[2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2015, 2016, 2018, 2019]

#droplist=['utctime',]
cols_own=['utctime',
          #'lat-2','lon-2','lat-3','lon-3','lat-4','lon-4','lat-1','lon-1',
          'pr-1','pr-2','pr-3','pr-4',
          'sfcWind-1','sfcWind-2','sfcWind-3','sfcWind-4',
          'tasmax-1','tasmax-2','tasmax-3','tasmax-4',
          'tasmin-1','tasmin-2','tasmin-3','tasmin-4',
          'maxWind-1','maxWind-2','maxWind-3','maxWind-4',
          'dayofyear',pred]
