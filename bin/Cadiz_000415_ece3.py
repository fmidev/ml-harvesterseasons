bbox='-6.32812,36.1403,-5.625,36.842'
harbor='Cadiz'
FMISID='000415'
lat=36.4997
lon=-6.2577
pred='WS_PT24H_AVG'
qpred='max_t('+pred+'/24h/0h)'
start='20000101T000000Z'
end='20230831T000000Z' 
starty=start[0:4]
endy=end[0:4]

fname = 'ece3-Cadiz.csv' # training input data file
mdl_name='mdl_'+pred+'_2000-2023_ece3_Cadiz.txt'
fscorepic='Fscore_'+pred+'-ece3-Cadiz.png'
xgbstudy='xgb-'+pred+'-ece3-Cadiz'
obsfile='obs-oceanids-'+start+'-'+end+'-'+pred+'-'+harbor+'-ece3-daymax.csv'
test_y=[2014, 2016, 2018, 2021, 2022]
train_y= [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2015, 2017, 2019, 2020, 2023]

cols_own=['utctime',pred,'dayofyear',#'hour',
#'lat-1','lon-1','lat-2','lon-2','lat-3','lon-3','lat-4','lon-4',
'pr-1','pr-2','pr-3','pr-4',
'sfcWind-1','sfcWind-2','sfcWind-3','sfcWind-4',
'tasmax-1','tasmax-2','tasmax-3','tasmax-4',
'tasmin-1','tasmin-2','tasmin-3','tasmin-4'
]