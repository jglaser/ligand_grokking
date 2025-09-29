import argparse
import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm

def parse_multiple_mutations_to_string(target_name: str) -> str | None:
    """
    Uses regular expressions to find and extract one or more point mutations
    from a protein target name string and returns them as a comma-separated string.
    """
    if not isinstance(target_name, str):
        return None
    mutation_pattern = r'[A-Z]\d+[A-Z]'
    mutations = re.findall(mutation_pattern, target_name)
    return ",".join(sorted(list(set(mutations)))) if mutations else None

def main(args):
    print(f"Loading raw BindingDB data from: {args.binding_db_path}")
    # --- CHANGE: Added 'Ligand InChI Key' to the columns to load ---
    use_cols = [
        'Ligand SMILES', 'Ligand InChI Key', 'UniProt (SwissProt) Primary ID of Target Chain 1',
        'Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)', 'Target Name',
        'PDB ID(s) for Ligand-Target Complex'
    ]
    df = pd.read_csv(args.binding_db_path, sep='\t', usecols=use_cols, low_memory=False)
    
    # --- Data Cleaning and Preprocessing ---
    df.rename(columns={
        'UniProt (SwissProt) Primary ID of Target Chain 1': 'uniprot_id',
        'PDB ID(s) for Ligand-Target Complex': 'pdb_id'
    }, inplace=True)

    # --- CHANGE: Drop rows missing the InChI Key ---
    df.dropna(subset=['Ligand SMILES', 'Ligand InChI Key', 'uniprot_id'], inplace=True)

    # Consolidate multiple affinity measurements into one column
    affinity_cols = ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']
    for col in affinity_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('>', '').str.replace('<', ''), errors='coerce')
    df['affinity_nm'] = df[affinity_cols].median(axis=1)
    df.dropna(subset=['affinity_nm'], inplace=True)

    # --- Separate Wild-Type and Mutant Data ---
    df['mutation'] = df['Target Name'].apply(parse_multiple_mutations_to_string)
    df_wt = df[df['mutation'].isnull()].copy()
    df_mt = df[df['mutation'].notnull()].copy()
    
    print(f"Separated into {len(df_wt)} wild-type and {len(df_mt)} mutant entries.")

    # --- Join WT and MT Data using InChIKey for robustness ---
    print("Joining mutant data with corresponding wild-type data using InChIKey...")

    # --- CHANGE: Aggregate WT data by InChIKey before merging ---
    wt_agg = df_wt.groupby(['uniprot_id', 'Ligand InChI Key']).agg(
        wt_affinity_nm=('affinity_nm', 'median')
    ).reset_index()

    # --- CHANGE: Perform the merge using 'Ligand InChI Key' ---
    merged_df = pd.merge(
        df_mt,
        wt_agg,
        on=['uniprot_id', 'Ligand InChI Key']
    )

    print(f"Found {len(merged_df)} paired mutant-wild-type measurements after join.")
    if merged_df.empty:
        print("Could not find any paired measurements. Exiting.")
        return
        
    # --- Create Resistance Label ---
    merged_df['fold_change'] = merged_df['affinity_nm'] / merged_df['wt_affinity_nm']
    merged_df['confers_resistance'] = (merged_df['fold_change'] > args.resistance_threshold).astype(int)
    
    # --- Finalize Dataset ---
    output_df = merged_df[[
        'Ligand SMILES', 'uniprot_id', 'mutation', 'pdb_id',
        'confers_resistance', 'wt_affinity_nm', 'affinity_nm'
    ]].rename(columns={'affinity_nm': 'mutant_affinity_nm'})
    
    # Group by the original unique combination and aggregate results
    final_df = output_df.groupby(['Ligand SMILES', 'uniprot_id', 'mutation']).agg({
        'pdb_id': 'first',
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
    parser.add_argument('--binding_db_path', type=str, required=True, help='Path to the BindingDB TSV data file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the final Parquet dataset.')
    parser.add_argument('--resistance_threshold', type=float, default=2.0, help='Fold-change in affinity to define resistance.')
    args = parser.parse_args()
    main(args)
