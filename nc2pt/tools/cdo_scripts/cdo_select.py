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
            "CDO (Climate Data Operators) is not installed or not in your system's PATH."
            "Please install CDO and ensure it's accessible."
        )

    for variable in cfg.variables:
        # Check if the input file exists
        input_file = os.path.join(cfg.data_dir, variable.name)
        if not os.path.isfile(input_file):
            logging.warning(
                f"Input file {input_file} not found. Skipping this variable."
            )
            continue

        logging.info(variable.name.split("_")[0].upper())
        intermediate_file = os.path.join(
            cfg.output_dir, f'{variable.name.split(".")[0]}_sellonlatbox_cropped.nc'
        )
        output_file = os.path.join(
            cfg.output_dir, f'{variable.name.split(".")[0]}_time_sliced_cropped.nc'
        )

        # Build the CDO command for sellonlatbox
        cfg_box = f"{cfg.lon_min},{cfg.lon_max},{cfg.lat_min},{cfg.lat_max}"
        sellonlatbox_command = (
            f"cdo -sellonlatbox,{cfg_box} {input_file} {intermediate_file}"
        )

        logging.info(f"Executing sellonlatbox command: {sellonlatbox_command}")

        # try:
        # Execute the sellonlatbox command using subprocess
        subprocess.run(sellonlatbox_command, shell=True, check=True)
        logging.info("sellonlatbox completed!")

        # Build the CDO command for time slicing
        slice_time_command = f"cdo seldate,{variable.start_date},{variable.end_date} {intermediate_file} {output_file}"

        logging.info(f"Executing time slicing command: {slice_time_command}")

        # Execute the time slicing command using subprocess
        subprocess.run(slice_time_command, shell=True, check=True)
        logging.info("Time slicing completed!")

        # Delete the intermediate file
        os.remove(intermediate_file)
        logging.info("Intermediate file deleted!")

        # except subprocess.CalledProcessError as e:
        #     logging.error(f"Error: {e}")


if __name__ == "__main__":
    main()
