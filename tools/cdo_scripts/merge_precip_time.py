import os
import hydra
import subprocess
import shutil
import logging


@hydra.main(config_name="cdo_config.yaml", config_path=".")
def main(cfg):
    # Initialize logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Check if the 'cdo' command is available
    if shutil.which("cdo") is None:
        raise EnvironmentError(
            "CDO (Climate Data Operators) is not installed or not in your system's PATH. Please install CDO and ensure it's accessible."
        )

    # Define the input file with the specified filename
    input_file = os.path.join(
        cfg.output_dir,
        "pr_1hr_ERA5_fc_RDA-025_1979010106-2019010106_time_sliced_cropped.nc",
    )
    output_prefix = os.path.join(
        cfg.output_dir,
        "pr_1hr_ERA5_fc_RDA-025_1979010106-2019010106_time_sliced_cropped.",
    )

    # Split the input file into time levels
    split_command = f"cdo -splitlevel {input_file} {output_prefix}"
    logging.info(f"Splitting levels: {split_command}")
    subprocess.run(split_command, shell=True, check=True)
    logging.info("Splitting levels completed!")

    # Process each time level
    for i in range(1, 13):
        j = i - 1
        k = f"{i:06d}"
        time_level_input = f"{output_prefix}{k}.nc"
        time_level_output = f"{output_prefix}{k}.nc_shift"

        shift_command = (
            f"cdo -setlevel,0 -shifttime,{j}hour {time_level_input} {time_level_output}"
        )
        logging.info(f"Processing time level {k}: {shift_command}")
        subprocess.run(shift_command, shell=True, check=True)
        logging.info(f"Processing time level {k} completed!")

    # Merge the shifted time levels
    merge_command = f"cdo -mergetime {output_prefix}*.nc_shift {cfg.output_dir}/pr_1hr_ERA5_fc_RDA-025_1979010106-2019010106_time_sliced_cropped_merged_forecast_time.nc"
    logging.info(f"Merging time levels: {merge_command}")
    subprocess.run(merge_command, shell=True, check=True)
    logging.info("Merging time levels completed!")

    # Delete the intermediate files
    for i in range(1, 13):
        k = f"{i:06d}"
        time_level_input = f"{output_prefix}{k}.nc"
        time_level_output = f"{output_prefix}{k}.nc_shift"
        os.remove(time_level_input)
        os.remove(time_level_output)
    logging.info("Intermediate files deleted!")


if __name__ == "__main__":
    main()
