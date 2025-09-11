import argparse
import os
import subprocess
import polars as pl

def main():
    """
    Main function to orchestrate the dataset building process for multiple targets.
    It gathers PDB IDs from a metadata file and calls the scaffold_splitter
    script once to process all targets efficiently.
    """
    parser = argparse.ArgumentParser(
        description="Automate the creation of scaffold-split datasets for multiple targets using a pre-generated metadata file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("binding_db_file", type=str, help="Path to the complete BindingDB TSV file.")
    parser.add_argument("pdb_id_file", type=str, default="../data/representative_pdb_ids.txt", help="Path to the PDB ids list file.")
    parser.add_argument("--output_dir", type=str, default="datasets", help="Directory to save the generated dataset subdirectories.")
    parser.add_argument("--activity_percentile", type=float, default=50.0,
                        help="Activity percentile (0-100) to define the active/inactive threshold.\n"
                             "Default is 50.0 (the median).")

    args = parser.parse_args()

    # --- 1. Create Output Directory ---
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # --- 2. Read Target PDB IDs from Metadata File ---
    print(f"Reading target PDB IDs from {args.pdb_id_file}...")
    try:
        index_df = pl.read_csv(args.pdb_id_file, has_header=False, new_columns=['pdb_id'])
        pdb_ids_to_process = index_df['pdb_id'].unique().to_list()
        
    except Exception as e:
        print(f"Error reading metadata file: {e}")
        return

    if not pdb_ids_to_process:
        print("No PDB IDs found in metadata file.")
        return

    print(f"Found {len(pdb_ids_to_process)} unique PDB IDs to process.")

    # --- 3. Call the Worker Script ONCE with all PDB IDs ---
    command = [
        "python", "scaffold_splitter.py",
        args.binding_db_file,
        "--output_dir", args.output_dir,
        "--activity_percentile", str(args.activity_percentile),
        "--pdb_ids", *pdb_ids_to_process  # Unpack the list of PDB IDs
    ]
    
    print("\nCalling the dataset builder to process all targets in a single run...")
    try:
        # Using subprocess.run without capturing output to stream stdout/stderr
        subprocess.run(command, check=True)
        print("\nDataset building process completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: The dataset builder script failed with exit code {e.returncode}.")
        print("See output above for details from the worker script.")

if __name__ == "__main__":
    main()


