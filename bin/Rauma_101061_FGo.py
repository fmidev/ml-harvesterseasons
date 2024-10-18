bbox='21.05273,61.39475,21.55273,60.89475'
harbor='Rauma'
FMISID='101061'
lat=61.14475
lon=21.30273 
pred='WG_PT1H_MAX'
qpred='max_t('+pred+'/24h/0h)'
start='20130701T000000Z'
starty=start[0:4]
end='20240101T000000Z' 
endy=end[0:4]

fname = 'training_data_oceanids_Rauma-sf_2013-2023.csv' # training input data file
mdl_name='mdl_'+pred+'_2013-2023_sf_Rauma_FGo.txt'
fscorepic='Fscore_'+pred+'-sf-Rauma-FGo.png'
xgbstudy='xgb-'+pred+'-Rauma'
obsfile='obs-oceanids-'+start+'-'+end+'-'+pred+'-'+harbor+'-daymax.csv'
test_y=[2016,2023]
train_y=[2013,2014,2015,2017,2018,2019,2020,2021,2022]

cols_own=['utctime',pred,#'dayOfYear',#'hour',
#'lat-1','lon-1','lat-2','lon-2','lat-3','lon-3','lat-4','lon-4',
#'e-1','e-2','e-3','e-4',
#'ewss-1','ewss-2','ewss-3','ewss-4',
'fg10-1','fg10-2','fg10-3','fg10-4',
#'lsm-1','lsm-2','lsm-3','lsm-4',
#'msl-1','msl-2','msl-3','msl-4',
#'nsss-1','nsss-2','nsss-3','nsss-4',
#'slhf-1','slhf-2','slhf-3','slhf-4','sshf-1','sshf-2','sshf-3','sshf-4',
#'ssr-1','ssr-2','ssr-3','ssr-4',
#'ssrd-1','ssrd-2','ssrd-3','ssrd-4',
#'str-1','str-2','str-3','str-4',
#'strd-1','strd-2','strd-3','strd-4',
#'t2-1','t2-2','t2-3','t2-4',
#'tcc-1','tcc-2','tcc-3','tcc-4',
#'td2-1','td2-2','td2-3','td2-4',
#'tlwc-1','tlwc-2','tlwc-3','tlwc-4',
#'tp-1','tp-2','tp-3','tp-4',
#'tsea-1','tsea-2','tsea-3','tsea-4', # in sf was nan everywhere
#'u10-1','u10-2','u10-3','u10-4','v10-1','v10-2','v10-3','v10-4'
]
droplist=['utctime','utctime.1','latitude', 'longitude', 'FMISID', pred,'lat-1', 'lon-1', 'lat-2', 'lon-2', 'lat-3', 'lon-3', 'lat-4', 'lon-4','hour','tlwc-1','tlwc-2','tlwc-3','tlwc-4']#,'dayOfYear','e-1','e-2','e-3','e-4','msl-1','msl-2','msl-3','msl-4','str-1','str-2','str-3','str-4','tcc-1','tcc-2','tcc-3','tcc-4','lsm-1','lsm-2','lsm-3','lsm-4']
