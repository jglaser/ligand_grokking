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

# --- Worker functions for parallel processing ---
def canonicalize_smiles(smiles: str) -> str | None:
    """Canonicalizes a single SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except Exception:
        return None

def get_scaffold(smiles_string: str) -> str:
    """Computes the Murcko scaffold for a given SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if not mol: return ""
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True) if scaffold else ""
    except Exception:
        return ""

def create_scaffold_split(df: pd.DataFrame, test_size: float = 0.2):
    """Splits a DataFrame into training and test sets based on molecular scaffolds."""
    if len(df) < 3:
        return df.copy().reset_index(drop=True), pd.DataFrame(columns=df.columns)
    scaffolds = defaultdict(list)
    for idx, row in df.iterrows():
        scaffold = get_scaffold(row['smiles'])
        if scaffold:
            scaffolds[scaffold].append(idx)
    scaffold_counts = sorted(scaffolds.items(), key=lambda x: len(x[1]))
    train_indices, test_indices = [], []
    n_test_target = int(len(df) * test_size)
    if n_test_target == 0 and len(df) > 1:
        n_test_target = 1
    for _, indices in scaffold_counts:
        if len(test_indices) < n_test_target:
            test_indices.extend(indices)
        else:
            train_indices.extend(indices)
    if not train_indices or not test_indices:
        return df.copy().reset_index(drop=True), pd.DataFrame(columns=df.columns)
    return df.loc[train_indices].reset_index(drop=True), df.loc[test_indices].reset_index(drop=True)

def process_and_save_target(args_tuple):
    """
    Worker function to process a single UniProt ID target.
    """
    uniprot_id, target_df_pd, args = args_tuple
    if target_df_pd.empty: return None

    if len(target_df_pd) < 2:
        threshold = target_df_pd['activity'].iloc[0]
    else:
        threshold = target_df_pd['activity'].quantile(args.activity_percentile / 100.0)
        
    target_df_pd['active'] = (target_df_pd['activity'] < threshold).astype(int)
    final_df = target_df_pd[['ligand_name', 'smiles', 'active']]

    if args.max_ligands and len(final_df) > args.max_ligands:
        final_df = final_df.sample(n=args.max_ligands, random_state=42)

    train_df, test_df = create_scaffold_split(final_df, test_size=args.test_size)
    
    if train_df.empty and test_df.empty: return None

    # --- THE CHANGE: Output directory is now the UniProt ID ---
    target_output_path = os.path.join(args.output_dir, uniprot_id)
    os.makedirs(target_output_path, exist_ok=True)

    train_df.to_csv(os.path.join(target_output_path, "train.csv"), index=False, quoting=1)
    test_df.to_csv(os.path.join(target_output_path, "test.csv"), index=False, quoting=1)

    return {
        'uniprot_id': uniprot_id,
        'activity_threshold_nM': threshold,
        'num_total_molecules': len(final_df),
        'num_train': len(train_df),
        'num_test': len(test_df),
        'num_scaffolds': len(pd.concat([train_df, test_df])['smiles'].apply(get_scaffold).unique())
    }

def main():
    parser = argparse.ArgumentParser(
        description="Filter BindingDB by UniProt ID and create scaffold-based train/test splits.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("binding_db_file", type=str, help="Path to the BindingDB Delta Lake directory.")
    parser.add_argument("metadata_file", type=str, help="Path to target_pocket_metadata.csv.")
    parser.add_argument("--output_dir", type=str, default="datasets", help="Directory to save the generated dataset subdirectories.")
    parser.add_argument("--summary_file", type=str, default="dataset_split_summary.csv", help="Name for the output summary CSV file.")
    parser.add_argument("--activity_cols", type=str, default="Ki (nM),IC50 (nM),Kd (nM)", help='Comma-separated list of activity columns.')
    parser.add_argument("--activity_percentile", type=float, default=50.0, help="Activity percentile to define active compounds.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Approximate fraction for the test set.")
    parser.add_argument("--max_ligands", type=int, default=1000, help="Maximum number of ligands per target.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    try:
        meta_df = pl.read_csv(args.metadata_file)
        uniprot_ids_to_process = meta_df['uniprot_id'].unique().to_list()
    except Exception as e:
        print(f"Error reading metadata file: {e}"); return

    print(f"Found {len(uniprot_ids_to_process)} unique UniProt IDs in metadata file to use for filtering.")

    activity_cols_list = [col.strip() for col in args.activity_cols.split(',')]
    lf_raw = pl.scan_delta(args.binding_db_file)

    uniprot_cols = [
        "UniProt (SwissProt) Primary ID of Target Chain",
        "UniProt (SwissProt) Primary ID of Target Chain 1",
        "UniProt (SwissProt) Primary ID of Target Chain 2"
    ]
    uniprot_cols_exist = [col for col in uniprot_cols if col in lf_raw.columns]

    lf_bindingdb = (
        lf_raw
        .with_columns(pl.coalesce(uniprot_cols_exist).alias("uniprot_id"))
        .filter(pl.col("uniprot_id").is_in(uniprot_ids_to_process))
        .rename({
            'BindingDB Ligand Name': 'ligand_name', 
            'Ligand SMILES': 'smiles'
        })
        .with_columns(
            pl.min_horizontal([
                pl.col(col).cast(pl.Utf8).str.replace_all(r"[<>]", "").cast(pl.Float64, strict=False)
                for col in activity_cols_list if col in lf_raw.columns
            ]).alias("activity")
        )
        .filter(pl.col("smiles").is_not_null() & pl.col("activity").is_not_null())
    )

    print("Collecting and processing data from BindingDB...")
    df_collected = lf_bindingdb.select(["uniprot_id", "ligand_name", "smiles", "activity"]).collect().to_pandas()
    
    if df_collected.empty:
        print("No matching entries found in BindingDB for the specified UniProt IDs. Exiting.")
        return
    print(f"Found {len(df_collected)} matching ligand entries in BindingDB.")
    
    unique_smiles = df_collected['smiles'].unique()
    with Pool(cpu_count()) as p:
        smiles_map = dict(zip(unique_smiles, tqdm(p.imap(canonicalize_smiles, unique_smiles), total=len(unique_smiles), desc="Canonicalizing SMILES")))

    df_collected['smiles'] = df_collected['smiles'].map(smiles_map)
    df_collected.dropna(subset=['smiles'], inplace=True)
    
    deduplicated_pd = df_collected.groupby(['uniprot_id', 'smiles']).agg(
        activity=('activity', 'median'),
        ligand_name=('ligand_name', 'first')
    ).reset_index()

    tasks = [(uniprot_id, group_df, args) for uniprot_id, group_df in deduplicated_pd.groupby('uniprot_id')]

    print(f"\nGenerating individual datasets for {len(tasks)} UniProt targets in parallel...")
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

