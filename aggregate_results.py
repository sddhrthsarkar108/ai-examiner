import argparse
import logging
import os
import re
from pathlib import Path

import pandas as pd


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def find_latest_run_folder(output_dir):
    """Find the run folder with the highest number in a given output directory.

    Args:
        output_dir (Path): Path to the output directory for a specific file

    Returns:
        Path: Path to the latest run folder, or None if no run folders found
    """
    run_folders = [
        d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("run_")
    ]

    if not run_folders:
        logger.warning(f"No run folders found in {output_dir}")
        return None

    # Extract run numbers and find the highest
    run_numbers = []
    for folder in run_folders:
        match = re.match(r"run_(\d+)", folder.name)
        if match:
            run_numbers.append((int(match.group(1)), folder))

    if not run_numbers:
        logger.warning(f"No valid run folders found in {output_dir}")
        return None

    # Sort by run number (first element of tuple) and get the folder with highest number
    latest_run = sorted(run_numbers, key=lambda x: x[0], reverse=True)[0][1]
    logger.info(f"Found latest run folder: {latest_run}")

    return latest_run


def find_csv_file(folder_path, filename="aggregate_evaluation_summary.csv"):
    """Find the specified CSV file in the given folder.

    Args:
        folder_path (Path): Path to the folder to search in
        filename (str): Name of the CSV file to find

    Returns:
        Path: Path to the CSV file, or None if not found
    """
    csv_path = folder_path / filename

    if csv_path.exists():
        return csv_path

    logger.warning(f"CSV file {filename} not found in {folder_path}")
    return None


def aggregate_results(output_dir="output", master_filename="master_evaluation_summary"):
    """Aggregate results from all output folders into a master spreadsheet.

    Args:
        output_dir (str): Path to the main output directory
        master_filename (str): Base name of the output file (without extension)

    Returns:
        bool: True if successful, False otherwise
    """
    output_path = Path(output_dir)

    if not output_path.exists() or not output_path.is_dir():
        logger.error(
            f"Output directory {output_dir} does not exist or is not a directory"
        )
        return False

    # Get all immediate subdirectories in the output directory
    # These should be the file number directories
    file_dirs = [
        d for d in output_path.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]

    if not file_dirs:
        logger.error(f"No output folders found in {output_dir}")
        return False

    logger.info(f"Found {len(file_dirs)} output folders")

    # Collect dataframes from each output folder
    all_dfs = []
    for file_dir in file_dirs:
        # Get file number from directory name
        file_no = file_dir.name
        logger.info(f"Processing output folder for file: {file_no}")

        # Find the latest run folder
        latest_run = find_latest_run_folder(file_dir)
        if not latest_run:
            logger.warning(f"Skipping {file_no} - no run folders found")
            continue

        # Find the aggregate evaluation summary CSV
        csv_path = find_csv_file(latest_run)
        if not csv_path:
            logger.warning(
                f"Skipping {file_no} - no aggregate evaluation summary CSV found"
            )
            continue

        try:
            # Read the CSV and add file number column
            df = pd.read_csv(csv_path)

            # Try to convert file_no to integer if possible
            try:
                file_id = int(file_no)
            except ValueError:
                # If conversion fails, keep as string
                file_id = file_no

            df["ID"] = file_id

            # Move ID to be the first column
            cols = list(df.columns)
            cols.insert(0, cols.pop(cols.index("ID")))
            df = df[cols]

            all_dfs.append(df)
            logger.info(f"Added data from {file_no} to master spreadsheet")
        except Exception as e:
            logger.error(f"Error reading CSV for {file_no}: {e}")
            continue

    if not all_dfs:
        logger.error("No data found to aggregate")
        return False

    # Combine all dataframes
    master_df = pd.concat(all_dfs, ignore_index=True)

    # Always save as CSV (reliable, doesn't require extra dependencies)
    csv_path = output_path / f"{master_filename}.csv"
    try:
        master_df.to_csv(csv_path, index=False)
        logger.info(f"Master CSV saved to {csv_path}")
        successful = True
    except Exception as e:
        logger.error(f"Error saving master CSV: {e}")
        successful = False

    # Try to save as Excel if openpyxl is available
    try:
        excel_path = output_path / f"{master_filename}.xlsx"
        master_df.to_excel(excel_path, index=False, engine="openpyxl")
        logger.info(f"Master Excel spreadsheet saved to {excel_path}")
    except ImportError:
        logger.warning("Could not create Excel file - openpyxl package not installed")
        logger.info(
            "Only CSV file was created. To enable Excel output, install openpyxl: pip install openpyxl"
        )
    except Exception as e:
        logger.error(f"Error saving master Excel spreadsheet: {e}")

    return successful


def main():
    """Main function to parse arguments and run the aggregation."""
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation summaries into a master spreadsheet"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="output",
        help="Path to the main output directory (default: output)",
    )
    parser.add_argument(
        "--filename",
        "-f",
        default="master_evaluation_summary",
        help="Base name of the output file without extension (default: master_evaluation_summary)",
    )

    args = parser.parse_args()

    success = aggregate_results(args.output_dir, args.filename)

    if success:
        print(f"\n{'='*80}")
        print(f"SUCCESS: Master evaluation summary created successfully")
        print(f"CSV file saved to: {os.path.join(args.output_dir, args.filename)}.csv")
        print(f"{'='*80}\n")
        return 0
    else:
        print(f"\n{'='*80}")
        print(f"ERROR: Failed to create master evaluation summary")
        print(f"Please check the logs for details")
        print(f"{'='*80}\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
