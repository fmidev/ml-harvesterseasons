bbox='7.8261,52.7832,9.3261,54.2832'
harbor='Bremerhaven'
FMISID='004885'
lat=53.53321
lon=8.576089
pred='WS_PT24H_AVG'
qpred='max_t('+pred+'/24h/0h)'
start='20000101T000000Z'
end='20230831T000000Z' 
starty=start[0:4]
endy=end[0:4]

fname = 'ece3-Bremerhaven-updated.csv' # training input data file
mdl_name='mdl_'+pred+'_2000-2023_ece3_Bremerhaven-updated.txt'
fscorepic='Fscore_'+pred+'-ece3-Bremerhaven-updated.png'
xgbstudy='xgb-'+pred+'-ece3-Bremerhaven-updated'
obsfile='obs-oceanids-'+start+'-'+end+'-'+pred+'-'+harbor+'-ece3-updated-daymax.csv'
test_y=[2014, 2016, 2018, 2021, 2022]
train_y= [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2015, 2017, 2019, 2020, 2023]

cols_own=['utctime',pred,'dayofyear',#'hour',
#'lat-1','lon-1','lat-2','lon-2','lat-3','lon-3','lat-4','lon-4',
'pr-1','pr-2','pr-3','pr-4',
'sfcWind-1','sfcWind-2','sfcWind-3','sfcWind-4',
'tasmax-1','tasmax-2','tasmax-3','tasmax-4',
'tasmin-1','tasmin-2','tasmin-3','tasmin-4',
'month','sfcWind_sum','sfcWind_sum_mean','sfcWind_sum_min','sfcWind_sum_max',
'WS_PT24H_AVG_mean','WS_PT24H_AVG_min','WS_PT24H_AVG_max','sfcWind_WS_diff_mean','sfcWind_WS_diff_min','sfcWind_WS_diff_max'
]
