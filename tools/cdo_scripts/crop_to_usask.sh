echo "PR"
cdo -sellonlatbox,208,277,46,73 /kenes/downloaded-data/acannon/ERA5_NCAR-RDA_North_America/1hr/pr_1hr_ERA5_fc_RDA-025_1979010106-2019010106.nc ./pr_cropped.nc
echo "Slicing!"
cdo seldate,20001001,20151001 ./pr_cropped.nc ./pr_sliced_cropped.nc
echo "TAS"
cdo -sellonlatbox,208,277,46,73 /kenes/downloaded-data/acannon/ERA5_NCAR-RDA_North_America/1hr/tas_1hr_ERA5_an_RDA-025_1979010100-2018123123.nc ./tas_cropped.nc
echo "Slicing!"
cdo seldate,20001001,20151001 ./tas_cropped.nc ./tas_sliced_cropped.nc
echo "UAS"
cdo -sellonlatbox,208,277,46,73 /kenes/downloaded-data/acannon/ERA5_NCAR-RDA_North_America/1hr/uas_1hr_ERA5_an_RDA-025_1979010100-2018123123.nc ./uas_cropped.nc
echo "Slicing!"
cdo seldate,20001001,20151001 ./uas_cropped.nc ./uas_sliced_cropped.nc
echo "VAS"
cdo -sellonlatbox,208,277,46,73 /kenes/downloaded-data/acannon/ERA5_NCAR-RDA_North_America/1hr/vas_1hr_ERA5_an_RDA-025_1979010100-2018123123.nc ./vas_cropped.nc
echo "Slicing!"
cdo seldate,20001001,20151001 ./vas_cropped.nc ./vas_sliced_cropped.nc
