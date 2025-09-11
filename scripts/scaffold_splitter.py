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

# --- Helper Functions ---

def canonicalize_smiles_worker(smiles: str) -> tuple[str, str | None]:
    """
    Canonicalizes a single SMILES string for parallel processing.
    Returns a tuple of (original_smiles, canonical_smiles).
    """
    if not smiles:
        return smiles, None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles, None
        return smiles, Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return smiles, None

def get_scaffold(smiles_string: str) -> str | None:
    """Computes the Murcko scaffold for a given SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None: return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True) if scaffold else None
    except Exception:
        return None

def create_scaffold_split(df: pd.DataFrame, test_size: float = 0.2):
    """Splits a DataFrame into training and test sets based on molecular scaffolds."""
    scaffolds = defaultdict(list)
    for idx, row in df.iterrows():
        scaffold = get_scaffold(row['smiles'])
        if scaffold:
            scaffolds[scaffold].append(idx)

    scaffold_counts = sorted(scaffolds.items(), key=lambda x: len(x[1]))
    train_indices, test_indices = [], []
    n_test_target = int(len(df) * test_size)

    for scaffold, indices in scaffold_counts:
        if len(test_indices) < n_test_target:
            test_indices.extend(indices)
        else:
            train_indices.extend(indices)

    return df.loc[train_indices].reset_index(drop=True), df.loc[test_indices].reset_index(drop=True)

def process_and_save_target(args_tuple):
    """
    Worker function to threshold, split, and save a pre-processed DataFrame group.
    """
    pdb_id, target_df, args = args_tuple
    if target_df.empty:
        return None

    threshold = target_df['activity'].quantile(args.activity_percentile / 100.0)

    df_final = target_df.copy()
    df_final['active'] = (df_final['activity'] < threshold).astype(int)
    # Use the canonical smiles column for splitting
    df_final = df_final[['ligand_name', 'smiles', 'active']]

    # Limit the number of ligands per target by random sampling
    if args.max_ligands and len(df_final) > args.max_ligands:
        df_final = df_final.sample(n=args.max_ligands, random_state=42)

    train_df, test_df = create_scaffold_split(df_final, test_size=args.test_size)

    if train_df.empty or test_df.empty:
        return None

    target_output_path = os.path.join(args.output_dir, pdb_id)
    os.makedirs(target_output_path, exist_ok=True)

    train_file = os.path.join(target_output_path, "train.csv")
    test_file = os.path.join(target_output_path, "test.csv")
    train_df.to_csv(train_file, index=False, quoting=1)
    test_df.to_csv(test_file, index=False, quoting=1)

    return {
        'pdb_id': pdb_id,
        'target_name': target_df['Target Name'].iloc[0],
        'activity_threshold_nM': threshold,
        'num_total_molecules': len(df_final),
        'num_train': len(train_df),
        'num_test': len(test_df)
    }

# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(description="Filter BindingDB Delta Lake for PDB IDs and create scaffold-based train/test splits.")
    # (Parser arguments remain the same)
    parser.add_argument("input_file", type=str, help="Path to the BindingDB Delta Lake directory.")
    parser.add_argument("--pdb_ids", type=str, nargs='+', required=True, help="A list of PDB IDs to filter for.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated dataset subdirectories.")
    parser.add_argument("--summary_file", type=str, default="dataset_split_summary.csv", help="Name for the output summary CSV file.")
    parser.add_argument("--activity_cols", type=str, default="Ki (nM),IC50 (nM),Kd (nM)", help='Comma-separated list of activity columns.')
    parser.add_argument("--activity_percentile", type=float, default=50.0, help="Activity percentile to define active compounds.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Approximate fraction for the test set.")
    parser.add_argument("--max_ligands", type=int, default=1000, help="Maximum number of ligands per target.")
    args = parser.parse_args()

    # --- Pass 1: Efficiently map PDB IDs to Target Names (unchanged) ---
    print(f"Pass 1: Mapping {len(args.pdb_ids)} PDB IDs to Target Names...")
    # (This section is unchanged and correct)
    pdb_regex = f"(?i)({'|'.join(args.pdb_ids)})"

    try:
        q_map = (
            pl.scan_delta(args.input_file)
            .filter(pl.col("PDB ID(s) for Ligand-Target Complex").str.contains(pdb_regex))
            .select(["PDB ID(s) for Ligand-Target Complex", "Target Name"])
            .unique()
            .collect()
        )
    except Exception as e:
        print(f"Error during Delta Lake scan for PDB mapping: {e}")
        return

    pdb_to_name_map = {}
    name_to_pdb_map = defaultdict(list)
    for row in q_map.iter_rows(named=True):
        pdb_id_str = row["PDB ID(s) for Ligand-Target Complex"]
        target_name = row["Target Name"]
        for pdb_id in args.pdb_ids:
            if pdb_id.lower() in pdb_id_str.lower():
                pdb_to_name_map[pdb_id] = target_name
                if pdb_id not in name_to_pdb_map[target_name]:
                     name_to_pdb_map[target_name].append(pdb_id)

    target_names_to_fetch = set(pdb_to_name_map.values())
    print(f"Mapped {len(pdb_to_name_map)} PDB IDs to {len(target_names_to_fetch)} unique target names.")

    # --- Pass 2: Lazily Fetch Data and then Pre-process SMILES in Parallel ---
    print(f"Pass 2: Lazily fetching data for {len(target_names_to_fetch)} targets...")
    activity_cols_list = [col.strip() for col in args.activity_cols.split(',')]

    try:
        map_df = pl.DataFrame([
            {"Target Name": name, "pdb_id": pdb}
            for name, pdbs in name_to_pdb_map.items()
            for pdb in pdbs
        ])

        # Define the LAZY query to get all raw data
        q_raw = (
            pl.scan_delta(args.input_file)
            .filter(pl.col("Target Name").is_in(list(target_names_to_fetch)))
            .join(map_df.lazy(), on="Target Name", how="inner")
            .rename({'BindingDB Ligand Name': 'ligand_name', 'Ligand SMILES': 'smiles'})
            .with_columns(
                pl.min_horizontal([
                    pl.col(col).cast(pl.Utf8).str.replace_all(r"[<>]", "").cast(pl.Float64, strict=False)
                    for col in activity_cols_list
                ]).alias("activity")
            )
            .filter(pl.col("smiles").is_not_null() & pl.col("activity").is_not_null())
            .select(["pdb_id", "Target Name", "ligand_name", "smiles", "activity"])
        )

        # Execute a small query to get just the unique SMILES strings
        unique_smiles = q_raw.select("smiles").unique().collect().get_column("smiles").to_list()
        print(f"Found {len(unique_smiles)} unique SMILES to canonicalize...")

        # PARALLELIZE the canonicalization
        with Pool(cpu_count()) as p:
            results = list(tqdm(
                p.imap(canonicalize_smiles_worker, unique_smiles, chunksize=1000),
                total=len(unique_smiles),
                desc="Canonicalizing SMILES in parallel"
            ))

        # Create a mapping from original to canonical SMILES
        smiles_map_df = pl.DataFrame(
            results,
            schema=[("smiles", pl.Utf8), ("canonical_smiles", pl.Utf8)]
        ).drop_nulls("canonical_smiles")

        # --- Efficiently Process the Full Dataset using the SMILES Map ---
        print("Deduplicating and finalizing dataset...")
        q_processed = (
            q_raw
            # Join the canonical smiles back to the main data
            .join(smiles_map_df.lazy(), on="smiles", how="inner")
            # Now perform the efficient group-by and aggregation
            .group_by(['pdb_id', 'Target Name', 'canonical_smiles'])
            .agg(
                pl.col('activity').median().alias('activity'),
                pl.col('ligand_name').first().alias('ligand_name')
            )
            # Rename for consistency in the next step
            .rename({"canonical_smiles": "smiles"})
        )

        processed_df = q_processed.collect().to_pandas()
        print(f"Processed down to {len(processed_df)} unique molecule-target-PDB triplets.")

    except Exception as e:
        print(f"Polars processing failed: {e}")
        return

    # --- Final Parallel Loop: Distribute Clean Data for Splitting ---
    tasks = [
        (pdb_id, group, args)
        for pdb_id, group in processed_df.groupby('pdb_id')
    ]

    print(f"\nGenerating individual datasets for {len(tasks)} PDB IDs in parallel...")
    with Pool(cpu_count()) as p:
        results = list(tqdm(
            p.imap(process_and_save_target, tasks),
            total=len(tasks),
            desc="Generating Datasets"
        ))

    split_metadata = [meta for meta in results if meta]

    # --- Save Metadata Summary File ---
    if split_metadata:
        summary_df = pd.DataFrame(split_metadata)
        summary_file_path = os.path.join(args.output_dir, args.summary_file)
        summary_df.to_csv(summary_file_path, index=False)
        print(f"\nSaved split metadata summary for {len(summary_df)} targets to: {summary_file_path}")
    else:
        print("\nNo datasets were successfully generated.")


if __name__ == '__main__':
    main()
