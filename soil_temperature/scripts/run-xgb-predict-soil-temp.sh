#!/bin/bash

# monthly script for XGBoost prediction for soil temparature in ERA5-Land grid
# give year month as cmd

source ~/.smart
eval "$(conda shell.bash hook)"

conda activate xgb

year=$1
month=$2

cd /home/smartmet/data

grid='era5l'
echo $year $month $grid

# REQUIRED PREDICTORS :
#'t2-00','td2-00','td2-12','u10-00','swvl2-00','sd-00','stl1-00','v10-00','laihv-00','lailv-00'

echo 'remap sl00'
# u10,v10,d2m,t2m,rsn-00,sd,stl1 ens/ec-sf_${year}${month}_all-24h-eu-50.grib remapped to era5l
[ -f ens/ec-sf_${year}${month}_all+sde-24h-eu-50.grib ] && ! [ -f ens/ec-sf_${grid}_${year}${month}_sl00utc-24h-eu-50.grib ] && \
seq 0 50 | parallel grib_copy -w shortName=2t/2d/stl1/rsn/sde/10u/10v/lai_hv/lai_lv ens/ec-BSF_${year}${month}_unbound-24h-eu-{}-fixed.grib ens/ec-BSF_${year}${month}_snow-24h-eu-{}-fixed.grib  \
    ens/ec-sf_${grid}_${year}${month}_sl00utc-24h-eu-{}.grib || echo 'not remappig 00utc - already done or no input file'
#cdo -b P12 -O --eccodes merge -selname,2d,2t,stl1 ens/ec-BSF_${year}${month}_unbound-24h-eu-{}.grib -selname,rsn,sde ens/ec-BSF_${year}${month}_snow-24h-eu-{}.grib 

# REQUIRED PREDICTORS :'slhf','sshf','ssrd','strd','str','ssr'

echo 'remap accumulated'
# tp,e,slhf,sshf,ro,str,strd,ssr,ssrd,sf ens/disacc-${year}${month}-50.grib remapped to era5l
[ -f ens/disacc-${year}${month}-50.grib ] && ! [ -f ens/disacc_${grid}_${year}${month}-50.grib ] && \
 seq 0 50 | parallel cdo -b P12 -O --eccodes remap,$grid-$abr-grid,ec-sf-$grid-$abr-weights.nc -shifttime,1day \
 -selname,tp,slhf,sshf,ro,str,strd,ssr,ssrd ens/disacc-${year}${month}-{}.grib ens/disacc_${grid}_${year}${month}-{}.grib \
|| echo 'not remappig disacc - already done or no input files'
! [ -f ens/disacc-${year}${month}-50-fixed.grib ] && \
 seq 0 50 | parallel grib_set -s jScansPositively=0 ens/disacc_${grid}_${year}${month}-{}.grib ens/disacc_${grid}_${year}${month}-{}-fixed.grib \

# REQUIRED PREDICTORS :'ro5d','sro5d','ssro5d','evapp5d','tp5d'

echo 'runsums'
# rolling cumsums cdo 
[ -f ens/disacc_${grid}_${year}${month}-50.grib ] && ! [ -f ens/ec-sf_runsums_${grid}_${year}${month}-50.grib ] && \
 seq 0 50 | parallel cdo -b P12 -O --eccodes runsum,5 -selname,tp,ro,e,sro,ssro ens/disacc_${grid}_${year}${month}-{}.grib ens/ec-sf_runsums_${grid}_${year}${month}-{}.grib \
 || echo 'not doing runsums - already done or no input files'
! [ -f ens/ec-sf_runsums_${grid}_${year}${month}-50-fixed.grib ] && \
 seq 0 50 | parallel grib_set -s jScansPositively=0 ens/ec-sf_runsums_${grid}_${year}${month}-{}.grib ens/ec-sf_runsums_${grid}_${year}${month}-{}-fixed.grib 

# REQUIRED PREDICTORS :
#'laihv-12','lailv-12',,'sd-12','sktn-00','sktd-12','rsn-12','stl1-12',
#'t2-12','swvl2-12',,'u10-12','v10-12'

# get day laihv lailv
! [ -f ens/ECC_${year}${month}01T120000_laihv-eu-day.grib ] && ! [ -f ens/ECC_${year}${month}01T120000_lailv-eu-day.grib ] && \
    diff=$(($year - 2020)) && \
    cdo -shifttime,${diff}years -shifttime,-12hour grib/ECC_20000101T000000_laihv-eu-day.grib ens/ECC_${year}${month}01T120000_laihv-eu-day.grib && \
    cdo -shifttime,${diff}years -shifttime,-12hour grib/ECC_20000101T000000_lailv-eu-day.grib ens/ECC_${year}${month}01T120000_lailv-eu-day.grib && \

# get day sde sktd 
'''
need grib/ECC_20000101T000000_sde-eu-day.grib ?
'''
! [ -f ens/ECC_${year}${month}01T120000_sde-eu-day.grib ] && ! [ -f ens/ECC_${year}${month}01T120000_sktd-eu-day.grib && ] && \
    diff=$(($year - 2020)) && \
    cdo -shifttime,${diff}years -shifttime,-12hour grib/ECC_20000101T000000_sde-eu-day.grib ens/ECC_${year}${month}01T120000_sde-eu-day.grib && \
    cdo -shifttime,${diff}years -shifttime,-12hour grib/LSASAFC_20000101T120000_ydmean-day-eu.grib ens/ECC_${year}${month}01T120000_sktd-eu-day.grib && \

# get day rsn stl1
'''
need grib/ECC_20000101T000000_rsn-eu-day.grib ?
need grib/ECC_20000101T000000_stl1-eu-day.grib ?
'''
! [ -f ens/ECC_${year}${month}01T120000_rsn-eu-day.grib ] && ! [ -f ens/ECC_${year}${month}01T120000_stl1-eu-day.grib ] && \
    diff=$(($year - 2020)) && \
    cdo -shifttime,${diff}years -shifttime,-12hour grib/ECC_20000101T000000_rsn-eu-day.grib ens/ECC_${year}${month}01T120000_rsn-eu-day.grib && \
    cdo -shifttime,${diff}years -shifttime,-12hour grib/ECC_20000101T000000_stl1-eu-day.grib ens/ECC_${year}${month}01T120000_stl1-eu-day.grib && \

# get day swvl2 night sktn
'''
need grib/ECC_20000101T000000_swvl2-eu-day.grib ?
'''
! [ -f ens/ECC_${year}${month}01T120000_swvl2-eu-day.grib ] && ! [ -f ens/ECC_${year}${month}01T000000_sktn-eu-night.grib && ] && \
    diff=$(($year - 2020)) && \
    cdo -shifttime,${diff}years -shifttime,-12hour grib/ECC_20000101T000000_swvl2-eu-day.grib ens/ECC_${year}${month}01T120000_swvl2-eu-day.grib && \
    cdo -shifttime,${diff}years -shifttime,-12hour grib/LSASAFC_20000101T000000_ydmean_nights-eu.grib ens/ECC_${year}${month}01T000000_sktn-eu-night.grib && \

# get day u10-12 v10-12
'''
need grib/ECC_20000101T000000_u10-eu-day.grib ?
need grib/ECC_20000101T000000_v10-eu-day.grib ?
'''
! [ -f ens/ECC_${year}${month}01T120000_u10-eu-day.grib ] && ! [ -f ens/ECC_${year}${month}01T120000_v10-eu-day.grib ] && \
    diff=$(($year - 2020)) && \
    cdo -shifttime,${diff}years -shifttime,-12hour grib/ECC_20000101T000000_u10-eu-day.grib ens/ECC_${year}${month}01T120000_u10-eu-day.grib && \
    cdo -shifttime,${diff}years -shifttime,-12hour grib/ECC_20000101T000000_v10-eu-day.grib ens/ECC_${year}${month}01T120000_v10-eu-day.grib && \

# get day t2 td2-12
'''
need grib/ECC_20000101T000000_t2-eu-day.grib ?
need grib/ECC_20000101T000000_td2-eu-day.grib ?
'''
! [ -f ens/ECC_${year}${month}01T120000_t2-eu-day.grib ] && ! [ -f ens/ECC_${year}${month}01T120000_td2-eu-day.grib ] && \
    diff=$(($year - 2020)) && \
    cdo -shifttime,${diff}years -shifttime,-12hour grib/ECC_20000101T000000_t2-eu-day.grib ens/ECC_${year}${month}01T120000_t2-eu-day.grib && \
    cdo -shifttime,${diff}years -shifttime,-12hour grib/ECC_20000101T000000_td2-eu-day.grib ens/ECC_${year}${month}01T120000_td2-eu-day.grib && \


echo 'start xgb predict'
seq 0 50 | parallel -j1 python /home/ubuntu/bin/xgb-predict-soiltemp-${grid}.py ens/ec-BSF_${year}${month}_swvls-24h-eu-{}-fixed.grib ens/ec-sf_${grid}_${year}${month}_sl00utc-24h-eu-{}.grib ens/ec-sf_runsums_${grid}_${year}${month}-{}-fixed.grib ens/disacc_${grid}_${year}${month}-{}-fixed.grib ens/ECC_${year}${month}01T120000_laihv-eu-day.grib ens/ECC_${year}${month}01T120000_lailv-eu-day.grib ens/ECC_${year}${month}01T120000_sde-eu-day.grib ens/ECC_${year}${month}01T120000_sktd-eu-day.grib ens/ECC_${year}${month}01T120000_rsn-eu-day.grib ens/ECC_${year}${month}01T120000_stl1-eu-day.grib ens/ECC_${year}${month}01T120000_swvl2-eu-day.grib ens/ECC_${year}${month}01T120000_u10-eu-day.grib ens/ECC_${year}${month}01T120000_v10-eu-day.grib ens/ECC_${year}${month}01T120000_t2-eu-day.grib ens/ECC_${year}${month}01T120000_td2-eu-day.grib ens/ECC_${year}${month}01T000000_sktn-eu-night.grib ens/ECXSF_${year}${month}_soiltemp_${grid}_out-{}.nc

echo 'netcdf to grib'
# netcdf to grib
seq 0 50 | parallel cdo -b 16 -f grb2 copy -setparam,41.228.192 -setmissval,-9.e38 -seltimestep,9/209 ens/ECXSF_${year}${month}_soiltemp_${grid}_out-{}.nc ens/ECXSF_${year}${month}_soiltemp_${grid}_out-{}.grib #|| echo "NO input or already netcdf to grib1"

echo 'grib fix'
# fix grib attributes
seq 0 50 | parallel grib_set -r -s centre=86,productDefinitionTemplateNumber=1,totalNumber=51,number={} ens/ECXSF_${year}${month}_soiltemp_${grid}_out-{}.grib \
ens/ECXSF_${year}${month}_soiltemp_${grid}_out-{}-fixed.grib
# || echo "NOT fixing swi2 grib attributes - no input or already produced"

echo 'join'
# join ensemble members and move to grib folder
grib_copy ens/ECXSF_${year}${month}_soiltemp_${grid}_out-*-fixed.grib grib/ECXSF_${year}${month}01T000000_soiltemp-24h-eu-${grid}.grib || echo "NOT joining pl-pp ensemble members - no input or already produced"
#wait 
