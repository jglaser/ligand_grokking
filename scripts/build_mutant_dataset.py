import argparse
import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm

def parse_mutation(target_name: str) -> str | None:
    """
    Uses regular expressions to find and extract a single point mutation
    from a protein target name string.
    """
    if not isinstance(target_name, str):
        return None
    match = re.search(r'[\(\[]([A-Z]\d+[A-Z])[\s\w]*[\)\]]', target_name)
    if match:
        return match.group(1)
    return None

def main(args):
    print(f"Loading raw BindingDB data from: {args.binding_db_path}")
    use_cols = [
        'Ligand SMILES', 'UniProt (SwissProt) Primary ID of Target Chain 1',
        'Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)', 'Target Name',
        'PDB ID(s) for Ligand-Target Complex'
    ]
    df = pd.read_csv(args.binding_db_path, sep='\t', usecols=use_cols, low_memory=False)
    
    # --- Data Cleaning and Preprocessing ---
    print("Cleaning and preprocessing data...")
    df.rename(columns={
        'UniProt (SwissProt) Primary ID of Target Chain 1': 'uniprot_id',
        'PDB ID(s) for Ligand-Target Complex': 'pdb_id'
    }, inplace=True)
    
    affinity_cols = ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']
    for col in affinity_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('>', '').str.replace('<', ''), errors='coerce')
    df['affinity_nm'] = df[affinity_cols].bfill(axis=1).iloc[:, 0]
    
    # Do NOT require a PDB ID yet. This is the key change.
    df.dropna(subset=['Ligand SMILES', 'uniprot_id', 'affinity_nm', 'Target Name'], inplace=True)
    df = df[df['affinity_nm'] > 0]

    # --- Impute Missing PDB IDs ---
    print("Imputing missing PDB IDs using a representative for each UniProt ID...")
    # Create a map from uniprot_id to the first valid PDB ID found for it.
    pdb_map = df.dropna(subset=['pdb_id']).groupby('uniprot_id')['pdb_id'].first()
    df['pdb_id'] = df['uniprot_id'].map(pdb_map)
    
    # Now, drop any proteins for which NO PDB ID could be found at all.
    df.dropna(subset=['pdb_id'], inplace=True)
    print(f"Data retained after PDB imputation: {len(df)} measurements.")
    
    # --- Parse Mutations ---
    print("Parsing mutations from target names...")
    tqdm.pandas(desc="Parsing Mutations")
    df['mutation'] = df['Target Name'].progress_apply(parse_mutation)
    
    df_wt = df[df['mutation'].isnull()].copy()
    df_mt = df[df['mutation'].notnull()].copy()
    
    print(f"Found {len(df_wt)} wild-type and {len(df_mt)} mutant measurements with associated PDB IDs.")
    
    # --- Create WT Lookup Dictionary ---
    print("Creating wild-type affinity lookup table...")
    wt_lookup = df_wt.groupby(['Ligand SMILES', 'uniprot_id'])['affinity_nm'].median().to_dict()
    
    # --- Match Mutants to Wild-Types and Generate Labels ---
    print("Matching mutant data to wild-type and generating resistance labels...")
    def get_wt_affinity(row):
        return wt_lookup.get((row['Ligand SMILES'], row['uniprot_id']))

    tqdm.pandas(desc="Matching WT Affinities")
    df_mt['wt_affinity_nm'] = df_mt.progress_apply(get_wt_affinity, axis=1)
    df_mt.dropna(subset=['wt_affinity_nm'], inplace=True)
    
    df_mt['fold_change'] = df_mt['affinity_nm'] / df_mt['wt_affinity_nm']
    df_mt['confers_resistance'] = (df_mt['fold_change'] > args.resistance_threshold).astype(int)
    
    # --- Finalize Dataset ---
    output_df = df_mt[[
        'Ligand SMILES', 'uniprot_id', 'mutation', 'pdb_id',
        'confers_resistance', 'wt_affinity_nm', 'affinity_nm'
    ]].rename(columns={'affinity_nm': 'mutant_affinity_nm'})
    
    # Group by the unique combination and aggregate
    final_df = output_df.groupby(['Ligand SMILES', 'uniprot_id', 'mutation']).agg({
        'pdb_id': 'first', # All PDBs for this group should be the same after imputation
        'confers_resistance': lambda x: x.mode()[0],
        'wt_affinity_nm': 'median',
        'mutant_affinity_nm': 'median'
    }).reset_index()
    
    print(f"\nGenerated a final dataset with {len(final_df)} unique ligand-mutant pairs.")
    print(f"Class distribution:\n{final_df['confers_resistance'].value_counts(normalize=True)}")
    
    print(f"Saving final dataset to: {args.output_path}")
    final_df.to_parquet(args.output_path, index=False)
    print("--- Done ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build a drug resistance dataset from BindingDB.")
    parser.add_argument('--binding_db_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='mutant_resistance_dataset_with_pdb.parquet')
    parser.add_argument('--resistance_threshold', type=float, default=10.0)
    args = parser.parse_args()
    main(args)


