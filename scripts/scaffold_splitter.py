import argparse
import os
import warnings
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import pandas as pd
import polars as pl
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm.auto import tqdm

# --- Worker function for parallel processing (must be at top level) ---
def canonicalize_smiles_worker(smiles: str) -> str:
    """
    Worker function to canonicalize a single SMILES string.
    Returns canonical SMILES or None if invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
        return None
    except:
        return None

def parallel_canonicalize(smiles_series: pd.Series) -> list:
    """
    Canonicalizes a pandas Series of SMILES strings in parallel.
    """
    with Pool(cpu_count()) as p:
        # Use imap for memory efficiency and to allow tqdm to track progress
        canonical_smiles_list = list(tqdm(
            p.imap(canonicalize_smiles_worker, smiles_series, chunksize=1000),
            total=len(smiles_series),
            desc="Canonicalizing SMILES"
        ))
    return canonical_smiles_list


def get_scaffold(smiles_string: str) -> str:
    """Computes the Murcko scaffold for a given SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold, canonical=True)
        return ""
    except:
        return ""

def create_scaffold_split(df: pd.DataFrame, test_size: float = 0.2):
    """Splits a DataFrame into training and test sets based on molecular scaffolds."""
    scaffolds = defaultdict(list)
    for idx, row in df.iterrows():
        scaffold = get_scaffold(row['smiles'])
        if scaffold:
            scaffolds[scaffold].append(idx)

    scaffold_counts = sorted(scaffolds.items(), key=lambda x: len(x[1]))
    train_indices, test_indices = [], []
    n_total = len(df)
    n_test_target = int(n_total * test_size)

    for scaffold, indices in scaffold_counts:
        if len(test_indices) < n_test_target:
            test_indices.extend(indices)
        else:
            train_indices.extend(indices)

    return df.loc[train_indices].reset_index(drop=True), df.loc[test_indices].reset_index(drop=True)

def main():
    """Filters a large DB for multiple targets and creates individual scaffold-split datasets."""
    parser = argparse.ArgumentParser(description="Filter BindingDB for a list of PDB IDs and create scaffold-based train/test splits.")
    parser.add_argument("input_file", type=str, help="Path to the complete BindingDB TSV file.")
    parser.add_argument("--pdb_ids", type=str, nargs='+', required=True, help="A list of PDB IDs to filter for.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated dataset subdirectories.")
    parser.add_argument("--activity_cols", type=str, default="Ki (nM),IC50 (nM),Kd (nM)", help='Comma-separated list of activity columns.')
    parser.add_argument("--activity_percentile", type=float, default=50.0, help="Activity percentile to define active compounds.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Approximate fraction for the test set.")

    args = parser.parse_args()
    
    warnings.warn("Using robust Polars engine. Malformed rows may be skipped.")

    # --- Pass 1: Map PDB IDs to Target Names ---
    print(f"Pass 1: Mapping {len(args.pdb_ids)} PDB IDs to Target Names...")
    pdb_regex = f"(?i)({'|'.join(args.pdb_ids)})"
    try:
        q_map = (
            pl.scan_csv(args.input_file, separator='\t', ignore_errors=True, truncate_ragged_lines=True, quote_char=None)
            .filter(pl.col("PDB ID(s) for Ligand-Target Complex").str.contains(pdb_regex))
            .select(["PDB ID(s) for Ligand-Target Complex", "Target Name"])
            .unique()
        )
        mapping_df_pd = q_map.collect().to_pandas()
    except Exception as e:
        print(f"Polars scan for mapping failed: {e}")
        return

    pdb_to_name_map = {}
    target_names_to_fetch = set()
    for _, row in mapping_df_pd.iterrows():
        pdb_id_str = row["PDB ID(s) for Ligand-Target Complex"]
        target_name = row["Target Name"]
        for pdb_id in args.pdb_ids:
            if pdb_id.lower() in pdb_id_str.lower():
                pdb_to_name_map[pdb_id] = target_name
                target_names_to_fetch.add(target_name)

    print(f"Mapped {len(pdb_to_name_map)} PDB IDs to {len(target_names_to_fetch)} unique target names.")

    # --- Pass 2: Expand Ligand Set and Process in Polars ---
    print(f"Pass 2: Fetching and processing all ligands for {len(target_names_to_fetch)} target names...")
    target_name_regex = "|".join([pl.escape_regex(name) for name in target_names_to_fetch])
    try:
        activity_cols_list = [col.strip() for col in args.activity_cols.split(',')]
        
        q_expand = (
            pl.scan_csv(args.input_file, separator='\t', ignore_errors=True, truncate_ragged_lines=True, quote_char=None)
            .filter(pl.col("Target Name").str.contains(target_name_regex))
            # Rename columns
            .rename({'BindingDB Ligand Name': 'name', 'Ligand SMILES': 'smiles'})
            # Coalesce activity columns
            .with_columns(
                pl.coalesce([
                    pl.col(col).str.replace_all(r"[<>]", "").cast(pl.Float64, strict=False)
                    for col in activity_cols_list if col in pl.scan_csv(args.input_file, separator='\t', n_rows=1).columns
                ]).alias("activity")
            )
            # Filter rows with no valid SMILES or activity
            .filter(pl.col("smiles").is_not_null() & pl.col("activity").is_not_null())
        )
        
        all_targets_df_pl = q_expand.collect()
        
    except Exception as e:
        print(f"Polars processing failed: {e}")
        return
        
    if all_targets_df_pl.is_empty():
        print("No valid entries found for any of the identified target names.")
        return

    print(f"Found {len(all_targets_df_pl)} raw valid entries across all targets.")

    # --- Deduplication (The only pandas/RDKit step) ---
    canonical_smiles = parallel_canonicalize(all_targets_df_pl['smiles'].to_pandas())
    all_targets_df_pl = all_targets_df_pl.with_columns(pl.Series("canonical_smiles", canonical_smiles)).drop_nulls("canonical_smiles")

    deduplicated_df = all_targets_df_pl.group_by(['Target Name', 'canonical_smiles']).agg(
        pl.col('activity').median().alias('activity'),
        pl.col('name').first().alias('name')
    ).rename({'canonical_smiles': 'smiles'})
    print(f"Processed down to {len(deduplicated_df)} unique molecule-target pairs.")

    # --- Final Loop: Create Individual Datasets ---
    deduplicated_pd = deduplicated_df.to_pandas()
    for pdb_id in tqdm(args.pdb_ids, desc="Generating individual datasets"):
        target_name = pdb_to_name_map.get(pdb_id)
        if not target_name:
            continue

        target_df_pd = deduplicated_pd[deduplicated_pd["Target Name"] == target_name].copy()

        if target_df_pd.empty:
            continue

        threshold = target_df_pd['activity'].quantile(args.activity_percentile / 100.0)
        target_df_pd['active'] = (target_df_pd['activity'] < threshold).astype(int)
        
        final_df = target_df_pd[['name', 'smiles', 'active']]
        train_df, test_df = create_scaffold_split(final_df, test_size=args.test_size)

        if train_df.empty or test_df.empty:
            tqdm.write(f"  - Skipping {pdb_id}: Not enough data for a train/test split.")
            continue

        target_output_path = os.path.join(args.output_dir, pdb_id)
        if not os.path.exists(target_output_path):
            os.makedirs(target_output_path)
        
        train_file = os.path.join(target_output_path, "train.csv")
        test_file = os.path.join(target_output_path, "test.csv")
        train_df.to_csv(train_file, index=False, quoting=1)
        test_df.to_csv(test_file, index=False, quoting=1)

        tqdm.write(f"  - Processed {pdb_id} ({target_name[:30]}...): {len(final_df)} unique molecules. Train/Test: {len(train_df)}/{len(test_df)}. Threshold: {threshold:.2f} nM")

if __name__ == '__main__':
    main()


