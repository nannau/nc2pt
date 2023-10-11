echo "PR"
cdo seldate,20001001,20151001 /kenes/downloaded-data/acannon/ERA5_NCAR-RDA_North_America/1hr/pr_1hr_ERA5_fc_RDA-025_1979010106-2019010106.nc ./pr_sliced.nc
echo "TAS"
cdo seldate,20001001,20151001 /kenes/downloaded-data/acannon/ERA5_NCAR-RDA_North_America/1hr/tas_1hr_ERA5_an_RDA-025_1979010100-2018123123.nc ./tas_sliced.nc
echo "UAS"
cdo seldate,20001001,20151001 /kenes/downloaded-data/acannon/ERA5_NCAR-RDA_North_America/1hr/uas_1hr_ERA5_an_RDA-025_1979010100-2018123123.nc ./uas_sliced.nc
echo "VAS"
cdo seldate,20001001,20151001 /kenes/downloaded-data/acannon/ERA5_NCAR-RDA_North_America/1hr/vas_1hr_ERA5_an_RDA-025_1979010100-2018123123.nc ./vas_sliced.nc
