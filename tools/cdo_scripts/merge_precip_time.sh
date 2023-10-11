cdo -splitlevel pr_sliced_cropped.nc pr_sliced_cropped_time.

for i in {1..12}; do
    let j=${i}-1;
    printf -v k "%06d" $i
    echo ${k};
    cdo -setlevel,0 -shifttime,${j}hour pr_sliced_cropped_time.${k}.nc pr_sliced_cropped_time.${k}.nc_shift ;
done;

cdo -mergetime *_shift pr_time_merged.nc
