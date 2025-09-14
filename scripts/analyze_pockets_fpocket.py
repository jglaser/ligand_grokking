import argparse
import subprocess
import pandas as pd
from tqdm.auto import tqdm
import os
import logging
import re
import shutil
import tempfile
import concurrent.futures

# --- Setup Basic Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')

def run_and_parse_fpocket(pdb_path: str, pdb_id: str) -> dict | None:
    """
    Worker function that handles the full process for a single PDB file,
    including robust, case-insensitive parsing of the fpocket output.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_pdb_path = os.path.join(temp_dir, f"{pdb_id.lower()}.pdb")
        shutil.copy(pdb_path, temp_pdb_path)

        try:
            subprocess.run(
                ['fpocket', '-f', temp_pdb_path],
                check=True,
                capture_output=True,
                text=True,
                timeout=120
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e:
            logging.error(f"[{pdb_id}] fpocket command failed. Error: {e}")
            return None

        base_temp_path = os.path.splitext(temp_pdb_path)[0]
        output_dir = f"{base_temp_path}_out"
        
        if not os.path.isdir(output_dir):
            logging.error(f"[{pdb_id}] Output directory was not created at {output_dir}")
            return None

        try:
            info_file_path = os.path.join(output_dir, f'{pdb_id.lower()}_info.txt')
            if not os.path.exists(info_file_path):
                logging.error(f"[{pdb_id}] Info file not found at {info_file_path}")
                return None

            with open(info_file_path, 'r') as f:
                content = f.read()
            
            pocket1_match = re.search(r'Pocket 1 :\s*\n(.*?)(?=\n\s*\n|\Z)', content, re.S)
            if not pocket1_match:
                logging.warning(f"[{pdb_id}] Could not find 'Pocket 1' data block.")
                return None
            
            pocket1_data = pocket1_match.group(1).strip()
            
            # --- THE FIX: Case-insensitive parsing with correct keys ---
            descriptors = {}
            for line in pocket1_data.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    # Standardize the key: lowercase and remove spaces
                    clean_key = key.strip().lower().replace(' ', '')
                    try:
                        descriptors[clean_key] = float(value.strip())
                    except ValueError:
                        pass
            
            # Map the standardized keys to our desired output columns
            result_dict = {
                'pdb_id': pdb_id,
                'fpocket_drug_score': descriptors.get('druggabilityscore'),
                'fpocket_volume': descriptors.get('volume'), # Matches 'Volume'
                'fpocket_hydrophobicity_score': descriptors.get('hydrophobicityscore'), # Matches 'Hydrophobicity score'
                'fpocket_polarity_score': descriptors.get('polarityscore'), # Matches 'Polarity score'
                'fpocket_num_alpha_spheres': descriptors.get('numberofalphaspheres') # Matches 'Number of Alpha Spheres'
            }
            
            return result_dict

        except Exception as e:
            logging.error(f"[{pdb_id}] An unexpected error occurred during parsing: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(
        description="Run fpocket in parallel and extract druggability descriptors.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("metadata_file", type=str, help="Path to a CSV file with a 'pdb_id' or 'run_name' column.")
    parser.add_argument("--output_file", type=str, default="fpocket_descriptors.csv", help="Name for the output CSV file.")
    parser.add_argument("--pdb_cache_dir", type=str, default="pdb_cache", help="Directory where PDB files are stored.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers. Defaults to number of CPU cores.")
    args = parser.parse_args()

    try:
        pdb_df = pd.read_csv(args.metadata_file)
        if 'pdb_id' in pdb_df.columns:
            pdb_ids = pdb_df['pdb_id'].unique().tolist()
        elif 'run_name' in pdb_df.columns:
            pdb_df['pdb_id'] = pdb_df['run_name'].apply(lambda x: x.split('-seed')[0])
            pdb_ids = pdb_df['pdb_id'].unique().tolist()
        else:
            print(f"Error: Input file must contain either a 'pdb_id' or 'run_name' column.")
            return
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.metadata_file}'")
        return

    tasks = []
    for pdb_id in pdb_ids:
        pdb_id_lower = pdb_id.lower()
        possible_paths = [
            os.path.join(args.pdb_cache_dir, f"{pdb_id_lower}.pdb"),
            os.path.join(args.pdb_cache_dir, f"pdb{pdb_id_lower}.ent")
        ]
        pdb_path = next((path for path in possible_paths if os.path.exists(path)), None)
        if pdb_path:
            tasks.append((pdb_path, pdb_id))
        else:
            logging.warning(f"[{pdb_id}] PDB file not found in cache. Skipping.")

    print(f"Found {len(tasks)} valid PDB files to process with fpocket.")
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_task = {executor.submit(run_and_parse_fpocket, task[0], task[1]): task for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks), desc="Processing with fpocket"):
            result = future.result()
            if result:
                results.append(result)

    if results:
        output_df = pd.DataFrame(results)
        output_df.dropna(how='all', subset=[col for col in output_df.columns if col != 'pdb_id'], inplace=True)
        output_df.to_csv(args.output_file, index=False)
        print(f"\nSuccessfully characterized pockets for {len(output_df)} PDB IDs.")
        print(f"Results saved to: {args.output_file}")
    else:
        print("\nCould not characterize any pockets with fpocket. Check the log messages above for details.")

if __name__ == "__main__":
    main()
