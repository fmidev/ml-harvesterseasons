bbox='8.3261,53.2832,8.8261,53.7832'
harbor='Bremerhaven'
FMISID='004885'
lat=53.53321
lon=8.576089
pred='WG_PT24H_MAX'
qpred='max_t('+pred+'/24h/0h)'
start='20000101T000000Z'
end='20230831T000000Z' 
starty=start[0:4]
endy=end[0:4]

fname = 'training_data_oceanids-Bremerhaven-sf_2000-2023.csv' # training input data file
mdl_name='mdl_'+pred+'_2000-2023_sf_Bremerhaven.txt'
fscorepic='Fscore_'+pred+'-sf-Bremerhaven.png'
xgbstudy='xgb-'+pred+'-Bremerhaven-sf'
obsfile='obs-oceanids-'+start+'-'+end+'-'+pred+'-'+harbor+'-sf-daymax.csv'
test_y=[2000, 2002, 2006, 2007, 2011]
train_y= [2001, 2003, 2004, 2005, 2008, 2009, 2010, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

cols_own=['utctime',pred,#'dayOfYear',#'hour',
#'lat-1','lon-1','lat-2','lon-2','lat-3','lon-3','lat-4','lon-4',
'e-1','e-2','e-3','e-4',
'ewss-1','ewss-2','ewss-3','ewss-4',
'fg10-1','fg10-2','fg10-3','fg10-4',
'lsm-1','lsm-2','lsm-3','lsm-4',
'msl-1','msl-2','msl-3','msl-4',
'nsss-1','nsss-2','nsss-3','nsss-4',
'slhf-1','slhf-2','slhf-3','slhf-4','sshf-1','sshf-2','sshf-3','sshf-4',
'ssr-1','ssr-2','ssr-3','ssr-4','ssrd-1','ssrd-2','ssrd-3','ssrd-4',
'str-1','str-2','str-3','str-4',
'strd-1','strd-2','strd-3','strd-4',
't2-1','t2-2','t2-3','t2-4',
'tcc-1','tcc-2','tcc-3','tcc-4',
'td2-1','td2-2','td2-3','td2-4',
'tlwc-1','tlwc-2','tlwc-3','tlwc-4',
'tp-1','tp-2','tp-3','tp-4', #tsea cut out of dataset completely due to being mostly nan
'u10-1','u10-2','u10-3','u10-4','v10-1','v10-2','v10-3','v10-4'
]
