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

def canonicalize_smiles(smiles: str) -> str:
    """Canonicalizes a single SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

def canonicalize_batch(batch_df: pl.DataFrame) -> pl.DataFrame:
    smiles_list = batch_df["smiles"].to_list()
    canonical_list = [canonicalize_smiles(s) for s in smiles_list]
    return batch_df.with_columns(pl.Series("smiles", canonical_list))

def get_scaffold(smiles_string: str) -> str | None:
    """
    Computes the Murcko scaffold for a given SMILES string.
    Returns the canonical SMILES of the scaffold, or None if invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return None
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return None

def process_and_save_target(args_tuple):
    """
    Worker function to process a single target: filter, threshold, split, and save.
    Returns metadata upon completion.
    """
    pdb_id, target_name, id_target_df, args = args_tuple

    threshold = id_target_df.get_column("activity").quantile(args.activity_percentile / 100.0)
    df_to_split = (
        id_target_df.with_columns(
            (pl.col("activity") < threshold).cast(pl.Int8).alias("active")
        )
        .select(["ligand_name", "smiles", "active"])
        .to_pandas()
    )
    
    train_df, test_df = create_scaffold_split(df_to_split, test_size=args.test_size)

    if train_df.empty or test_df.empty:
        return None # Not enough data for a split

    target_output_path = os.path.join(args.output_dir, pdb_id)
    if not os.path.exists(target_output_path):
        os.makedirs(target_output_path, exist_ok=True)
    
    train_file = os.path.join(target_output_path, "train.csv")
    test_file = os.path.join(target_output_path, "test.csv")
    train_df.to_csv(train_file, index=False, quoting=1)
    test_df.to_csv(test_file, index=False, quoting=1)

    return {
        'pdb_id': pdb_id,
        'target_name': target_name,
        'activity_threshold_nM': threshold,
        'num_total_molecules': len(df_to_split),
        'num_train': len(train_df),
        'num_test': len(test_df)
    }

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
    parser = argparse.ArgumentParser(description="Filter BindingDB for a list of PDB IDs and create scaffold-based train/test splits.")
    # ... (parser arguments remain the same)
    parser.add_argument("input_file", type=str, help="Path to the complete BindingDB TSV file.")
    parser.add_argument("--pdb_ids", type=str, nargs='+', required=True, help="A list of PDB IDs to filter for.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated dataset subdirectories.")
    parser.add_argument("--summary_file", type=str, default="dataset_split_summary.csv", help="Name for the output summary CSV file.")
    parser.add_argument("--activity_cols", type=str, default="Ki (nM),IC50 (nM),Kd (nM)", help='Comma-separated list of activity columns.')
    parser.add_argument("--activity_percentile", type=float, default=50.0, help="Activity percentile to define active compounds.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Approximate fraction for the test set.")
    args = parser.parse_args()
    
    # --- The rest of the main function, now with parallel processing for the final loop ---
    warnings.warn("Using robust Polars engine. Malformed rows may be skipped.")
    print(f"Pass 1: Mapping {len(args.pdb_ids)} PDB IDs to Target Names...")

    activity_cols = tuple(col.strip() for col in args.activity_cols.strip(","))
    pdb_regex = f"(?i)({'|'.join(args.pdb_ids)})"
    dl = pl.scan_delta(args.input_file).with_columns(pl.col("PDB ID(s) for Ligand-Target Complex").alias("pdb_id"))

    id_target_map_lf = (
        dl.filter(pl.col("pdb_id").str.contains(pdb_regex))
        .select(["pdb_id", "Target Name"])
        .unique()
        .with_columns(
            pl.col("pdb_id").str.extract(f"{pdb_regex}", 0)
        )
    )

    lf = (
        id_target_map_lf.join(dl, on="Target Name", how="left")
        .with_columns(
            pl.col("BindingDB Ligand Name").alias("ligand_name"),
            pl.col("Ligand SMILES").alias("smiles"),
            pl.min_horizontal(activity_cols).alias("activity"),
        )
        .select(["pdb_id", "Target Name", "smiles", "ligand_name", "activity"])
        .map_batches(canonicalize_batch)
        .filter(pl.col("smiles").is_not_null())
        .group_by(["pdb_id", "Target Name", "smiles"])
        .agg(
            pl.col("activity").median(),
            pl.col("ligand_name").first()
        )
    )

    df = lf.collect()
    tasks = tuple(
        (pdb_id, target, id_target_df, args)
        for (pdb_id, target), id_target_df in df.group_by(["pdb_id", "Target Name"])
    )

    # --- Parallel Final Loop ---
    split_metadata = []
    print(f"\nGenerating individual datasets for {len(tasks)} targets in parallel...")
    with Pool(cpu_count()) as p:
        results = list(tqdm(
            p.imap(process_and_save_target, tasks),
            total=len(tasks),
            desc="Generating Datasets in Parallel"
        ))
    
    for meta in results:
        if meta:
            split_metadata.append(meta)

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


