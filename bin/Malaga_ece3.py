bbox='-4.23222,36.41611,-4.73222,36.91611'
harbor='Malaga'
FMISID='000231'
lat=36.66611
lon=-4.48222
pred='WG_PT1H_MAX'
qpred='max_t('+pred+'/24h/0h)'
start='20000901T000000Z'
end='20230831T000000Z' 
starty=start[0:4]
endy=end[0:4]

fname = 'ece3-Malaga.csv' # training input data file
mdl_name='mdl_'+pred+'_2000-2023_ece3_Malaga.txt'
fscorepic='Fscore_'+pred+'-ece3-Malaga.png'
xgbstudy='xgb-'+pred+'-ece3-Malaga'
obsfile='obs-oceanids-'+start+'-'+end+'-'+pred+'-'+harbor+'-ece3-daymax.csv'
test_y=[2000, 2002, 2006, 2007, 2011]
train_y= [2001, 2003, 2004, 2005, 2008, 2009, 2010, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

cols_own=['utctime',pred,#'dayOfYear',#'hour',
#'lat-1','lon-1','lat-2','lon-2','lat-3','lon-3','lat-4','lon-4',
'pr-1','pr-2','pr-3','pr-4',
'sfcWind-1','sfcWind-2','sfcWind-3','sfcWind-4',
'tasmax-1','tasmax-2','tasmax-3','tasmax-4',
'tasmin-1','tasmin-2','tasmin-3','tasmin-4'
]