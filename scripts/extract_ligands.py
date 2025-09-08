import argparse
import os
import subprocess
import polars as pl
from tqdm.auto import tqdm

def main():
    """
    Main function to orchestrate the dataset building process for multiple targets.
    """
    parser = argparse.ArgumentParser(
        description="Automate the creation of scaffold-split datasets for multiple targets from PDBbind.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("binding_db_file", type=str, help="Path to the complete BindingDB TSV file.")
    parser.add_argument("pdbbind_index_file", type=str, help="Path to the PDBbind index file (e.g., 'INDEX_refined_data.2020').")
    parser.add_argument("--output_dir", type=str, default="datasets", help="Directory to save the generated dataset subdirectories.")
    parser.add_argument("--activity_percentile", type=float, default=50.0,
                        help="Activity percentile (0-100) to define the active/inactive threshold.\n"
                             "For each target, the threshold will be the specified percentile of its activity data.\n"
                             "Default is 50.0 (the median).")

    args = parser.parse_args()

    # --- 1. Create Output Directory ---
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # --- 2. Read Target Names from PDBbind Index ---
    print(f"Reading target names from {args.pdbbind_index_file}...")
    try:
        # Read the index file, skipping comment lines
        with open(args.pdbbind_index_file, 'r') as f:
            lines = [line.strip() for line in f if not line.startswith('#')]
        
        # Extract the full protein name, which starts after the first few columns
        # This is a heuristic that works for the standard PDBbind format.
        target_names = set()
        for line in lines:
            parts = line.split()
            if len(parts) > 4:
                 # The name is the rest of the line after the PDB ID, year, and Uniprot ID
                name = " ".join(parts[3:]).strip()
                target_names.add(name)

    except Exception as e:
        print(f"Error reading PDBbind index file: {e}")
        return

    unique_targets = sorted(list(target_names))
    print(f"Found {len(unique_targets)} unique target names.")

    # --- 3. Iterate and Call the Dataset Builder ---
    for target_name in tqdm(unique_targets, desc="Building Datasets"):
        print(f"\n----- Processing Target: {target_name} -----")
        
        # Create a safe directory name from the target name
        target_dir_name = "".join(c if c.isalnum() else "_" for c in target_name)[:50]
        target_output_path = os.path.join(args.output_dir, target_dir_name)
        
        if not os.path.exists(target_output_path):
            os.makedirs(target_output_path)

        train_file = os.path.join(target_output_path, "train.csv")
        test_file = os.path.join(target_output_path, "test.csv")

        # Construct the command to call scaffold_splitter.py
        command = [
            "python", "scaffold_splitter.py",
            args.binding_db_file,
            "--target_name", target_name,
            "--activity_percentile", str(args.activity_percentile),
            "--train_out", train_file,
            "--test_out", test_file
        ]
        
        # Execute the command
        try:
            # We use capture_output=True to keep the main script's output clean
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(result.stdout) # Print the worker script's summary
        except subprocess.CalledProcessError as e:
            print(f"ERROR: The dataset builder failed for target '{target_name}'.")
            print("--- Stderr from worker ---")
            print(e.stderr)
            print("--------------------------")


if __name__ == "__main__":
    main()

