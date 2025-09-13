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

# --- Worker functions for parallel processing (must be at top level) ---
def canonicalize_smiles(smiles: str) -> str:
    """Canonicalizes a single SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None

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
    n_test_target = int(len(df) * test_size)

    for _, indices in scaffold_counts:
        if len(test_indices) < n_test_target:
            test_indices.extend(indices)
        else:
            train_indices.extend(indices)
    return df.loc[train_indices].reset_index(drop=True), df.loc[test_indices].reset_index(drop=True)

def process_and_save_target(args_tuple):
    """
    Worker function to process a single target: filter, threshold, split, and save.
    Returns metadata upon completion.
    """
    uniprot_id, target_df_pd, args, uniprot_to_pdb = args_tuple
    if target_df_pd.empty: return None

    threshold = target_df_pd['activity'].quantile(args.activity_percentile / 100.0)
    target_df_pd['active'] = (target_df_pd['activity'] < threshold).astype(int)
    final_df = target_df_pd[['ligand_name', 'smiles', 'active']]

    if args.max_ligands and len(final_df) > args.max_ligands:
        final_df = final_df.sample(n=args.max_ligands, random_state=42)

    train_df, test_df = create_scaffold_split(final_df, test_size=args.test_size)
    if train_df.empty or test_df.empty: return None

    pdb_id = uniprot_to_pdb.get(uniprot_id, uniprot_id)
    target_output_path = os.path.join(args.output_dir, pdb_id)
    os.makedirs(target_output_path, exist_ok=True)

    train_df.to_csv(os.path.join(target_output_path, "train.csv"), index=False, quoting=1)
    test_df.to_csv(os.path.join(target_output_path, "test.csv"), index=False, quoting=1)

    return {
        'pdb_id': pdb_id, 'uniprot_id': uniprot_id,
        'activity_threshold_nM': threshold, 'num_total_molecules': len(final_df),
        'num_train': len(train_df), 'num_test': len(test_df)
    }

def main():
    parser = argparse.ArgumentParser(
        description="Filter BindingDB by UniProt ID and create scaffold-based train/test splits.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("binding_db_file", type=str, help="Path to the complete BindingDB Delta Lake directory.")
    parser.add_argument("metadata_file", type=str, help="Path to the target_pocket_metadata.csv file.")
    parser.add_argument("--output_dir", type=str, default="datasets", help="Directory to save the generated dataset subdirectories.")
    parser.add_argument("--summary_file", type=str, default="dataset_split_summary.csv", help="Name for the output summary CSV file.")
    parser.add_argument("--activity_cols", type=str, default="Ki (nM),IC50 (nM),Kd (nM)", help='Comma-separated list of activity columns.')
    parser.add_argument("--activity_percentile", type=float, default=50.0, help="Activity percentile to define active compounds.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Approximate fraction for the test set.")
    parser.add_argument("--max_ligands", type=int, default=1000, help="Maximum number of ligands per target.")
    parser.add_argument("--max_scaffolds", type=int, default=None, help="Maximum number of Murcko scaffolds to include per target.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    try:
        metadata_df = pd.read_csv(args.metadata_file)
        uniprot_ids_to_process = metadata_df['uniprot_id'].unique().tolist()
        uniprot_to_pdb = dict(zip(metadata_df['uniprot_id'], metadata_df['pdb_id']))
    except Exception as e:
        print(f"Error reading metadata file: {e}"); return
    
    print(f"Found {len(uniprot_ids_to_process)} unique UniProt IDs to process from {args.metadata_file}.")

    # Lazily scan the Delta Lake and filter for our UniProt IDs
    activity_cols_list = [col.strip() for col in args.activity_cols.split(',')]
    lf = (
        pl.scan_delta(args.binding_db_file)
        .filter(pl.col("UniProt (SwissProt) Primary ID of Target Chain 1").is_in(uniprot_ids_to_process))
        .rename({'BindingDB Ligand Name': 'ligand_name', 
                 'Ligand SMILES': 'smiles',
                 'UniProt (SwissProt) Primary ID of Target Chain 1': 'uniprot_id'})
        .with_columns(
            pl.min_horizontal([
                pl.col(col).cast(pl.Utf8).str.replace_all(r"[<>]", "").cast(pl.Float64, strict=False)
                for col in activity_cols_list
            ]).alias("activity")
        )
        .filter(pl.col("smiles").is_not_null() & pl.col("activity").is_not_null())
        .select(["uniprot_id", "ligand_name", "smiles", "activity"])
    )

    # Collect the full filtered dataset into memory
    print("Collecting and processing data from BindingDB...")
    df_collected = lf.collect().to_pandas()
    
    # Canonicalize SMILES in parallel
    unique_smiles = df_collected['smiles'].unique()
    with Pool(cpu_count()) as p:
        smiles_map = dict(zip(unique_smiles, tqdm(p.imap(canonicalize_smiles, unique_smiles), total=len(unique_smiles), desc="Canonicalizing SMILES")))
    
    df_collected['smiles'] = df_collected['smiles'].map(smiles_map)
    df_collected.dropna(subset=['smiles'], inplace=True)

    # Deduplicate based on UniProt ID and canonical SMILES
    deduplicated_pd = df_collected.groupby(['uniprot_id', 'smiles']).agg(
        activity=('activity', 'median'),
        ligand_name=('ligand_name', 'first')
    ).reset_index()

    # Prepare tasks for parallel processing
    tasks = [(uniprot_id, group, args, uniprot_to_pdb) for uniprot_id, group in deduplicated_pd.groupby('uniprot_id')]

    split_metadata = []
    print(f"\nGenerating individual datasets for {len(tasks)} targets in parallel...")
    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(process_and_save_target, tasks), total=len(tasks), desc="Generating datasets"))

    split_metadata = [meta for meta in results if meta]

    if split_metadata:
        summary_df = pd.DataFrame(split_metadata)
        summary_file_path = os.path.join(args.output_dir, args.summary_file)
        summary_df.to_csv(summary_file_path, index=False)
        print(f"\nSaved split metadata summary for {len(summary_df)} targets to: {summary_file_path}")
    else:
        print("\nNo datasets were successfully generated.")

if __name__ == '__main__':
    main()
