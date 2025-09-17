import argparse
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import os
from collections import defaultdict
from tqdm.auto import tqdm
import concurrent.futures
import numpy as np

def get_scaffold(smiles_string: str) -> str | None:
    """Computes the Murcko scaffold for a given SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None: return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True) if scaffold else None
    except Exception:
        return None

def stratified_scaffold_split(df, stratify_col='active', test_size=0.2, random_state=42):
    """
    Performs a scaffold-based split that is stratified by the activity class.
    """
    scaffolds = defaultdict(list)
    df = df.reset_index(drop=True)
    for idx, row in df.iterrows():
        scaffold = get_scaffold(row['canonical_smiles'])
        if scaffold:
            scaffolds[scaffold].append(idx)

    scaffold_labels = []
    scaffold_keys = list(scaffolds.keys())
    for scaffold in scaffold_keys:
        majority_class = df.loc[scaffolds[scaffold], stratify_col].mode()[0]
        scaffold_labels.append(majority_class)

    if len(scaffold_keys) < 2:
        return df, pd.DataFrame(columns=df.columns)

    try:
        train_scaffolds, test_scaffolds = train_test_split(
            scaffold_keys,
            test_size=test_size,
            stratify=scaffold_labels,
            random_state=random_state
        )
    except ValueError:
        train_scaffolds, test_scaffolds = train_test_split(scaffold_keys, test_size=test_size, random_state=random_state)
    
    train_indices = [idx for scaffold in train_scaffolds for idx in scaffolds[scaffold]]
    test_indices = [idx for scaffold in test_scaffolds for idx in scaffolds[scaffold]]
    
    return df.loc[train_indices], df.loc[test_indices]


def canonicalize_smiles_worker(smiles: str) -> tuple[str, str | None]:
    """
    Safely canonicalizes a single SMILES string for parallel processing.
    Returns the original and the canonical SMILES.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return smiles, Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        pass
    return smiles, None

def main():
    parser = argparse.ArgumentParser(description="Construct 'easy' or 'hard' multi-target datasets based on SAR discordance.")
    parser.add_argument("bindingdb_delta_path", type=str, help="Path to the BindingDB Delta Lake directory.")
    parser.add_argument("uniprot_pocket_clusters", type=str, help="Path to uniprot_pocket_clusters.csv.")
    parser.add_argument("ligand_clusters", type=str, help="Path to ligand_clusters.csv.")
    parser.add_argument("--mode", type=str, choices=['easy', 'hard'], required=True, help="Dataset construction mode.")
    parser.add_argument("--n_total_actives", type=int, default=10000, help="Total number of active compounds to include in the final dataset.")
    parser.add_argument("--output_dir", type=str, default='.', help="Directory to save the generated dataset files.")
    # --- NEW: Argument to limit pocket cluster diversity ---
    parser.add_argument("--max_pocket_clusters", type=int, default=None, help="Randomly select a subset of this many pocket clusters to build datasets from.")
    parser.add_argument('--random_seed', type=int, default=42, help="Random number generator seed")
    
    args = parser.parse_args()

    # --- 1. Load and Merge All Data using a Hybrid Polars/Pandas Approach ---
    try:
        print("Building lazy query plan with Polars...")
        
        uniprot_pocket_clusters_df = pl.read_csv(args.uniprot_pocket_clusters).lazy().with_columns(
            pl.col('uniprot_id').str.strip_chars()
        )
        ligand_clusters_df = pl.read_csv(args.ligand_clusters).lazy()
        
        binding_lazy = pl.scan_delta(args.bindingdb_delta_path)
        lazy_schema_names = binding_lazy.collect_schema().names()

        print("Collecting unique raw SMILES for parallel canonicalization...")
        raw_smiles_list = binding_lazy.select(pl.col("Ligand SMILES")).collect()['Ligand SMILES'].drop_nulls().unique().to_list()

        print(f"Canonicalizing {len(raw_smiles_list)} unique raw SMILES in parallel...")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(canonicalize_smiles_worker, raw_smiles_list), total=len(raw_smiles_list), desc="Canonicalizing"))
        
        canonical_map_df = pl.DataFrame(
            [{"raw_smiles": raw, "canonical_smiles": can} for raw, can in results if can is not None]
        ).lazy()
        
        base_cols = {
            'uniprot_id': 'UniProt (SwissProt) Primary ID of Target Chain',
            'sequence': 'BindingDB Target Chain Sequence'
        }
        for key, base_name in base_cols.items():
            chain_cols = [base_name] + [f"{base_name} {i}" for i in range(1, 5)]
            existing_chain_cols = [col for col in chain_cols if col in lazy_schema_names]
            if not existing_chain_cols: raise ValueError(f"CRITICAL ERROR: Could not find any columns for '{key}'.")
            binding_lazy = binding_lazy.with_columns(pl.coalesce(existing_chain_cols).alias(key))
            
        activity_cols = ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)']
        existing_activity_cols = [col for col in activity_cols if col in lazy_schema_names]
        activity_exprs = [
            pl.col(c).cast(pl.Utf8).str.replace_all(r"[<>]", "").cast(pl.Float64, strict=False) for c in existing_activity_cols
        ]
        binding_lazy = binding_lazy.with_columns(pl.min_horizontal(activity_exprs).alias("activity"))
        
        binding_lazy = binding_lazy.rename({"Ligand SMILES": "raw_smiles"})
        binding_lazy = binding_lazy.join(canonical_map_df, on="raw_smiles", how="inner").with_columns(
            pl.col('uniprot_id').str.strip_chars()
        )
        
        master_lazy = binding_lazy.join(uniprot_pocket_clusters_df, on='uniprot_id', how='left')
        master_lazy = master_lazy.join(ligand_clusters_df, on='canonical_smiles', how='left')
        
        master_lazy = master_lazy.drop_nulls(subset=['sequence', 'canonical_smiles', 'activity', 'uniprot_id'])
        
        print("Executing query plan and collecting data into memory...")
        master_df = master_lazy.collect().to_pandas()
        print(f"Master dataframe with full and partial cluster information created with {len(master_df)} entries.")

    except Exception as e:
        print(f"Error loading or merging files: {e}")
        return

    # --- 2. Build the Multi-Target Dataset ---
    print(f"\nConstructing '{args.mode}' dataset with ~{args.n_total_actives} actives...")
    
    master_df['active'] = master_df.groupby('uniprot_id')['activity'].transform(
        lambda x: (x < x.quantile(0.5)).astype(int)
    )

    actives_df = master_df[master_df['active'] == 1]
    inactives_df = master_df[master_df['active'] == 0]

    if actives_df.empty:
        print("Error: No active compounds found after filtering. Cannot proceed.")
        return

    sampled_actives = pd.DataFrame()
    
    actives_with_clusters = actives_df.dropna(subset=['pocket_cluster_id'])

    # --- THE CHANGE: Limit the number of pocket clusters if requested ---
    available_pocket_clusters = actives_with_clusters['pocket_cluster_id'].unique()
    if args.max_pocket_clusters and len(available_pocket_clusters) > args.max_pocket_clusters:
        print(f"Randomly selecting {args.max_pocket_clusters} out of {len(available_pocket_clusters)} available pocket clusters.")
        rng = np.random.default_rng(seed=args.random_seed)
        selected_clusters = rng.choice(available_pocket_clusters, size=args.max_pocket_clusters, replace=False)
        actives_with_clusters = actives_with_clusters[actives_with_clusters['pocket_cluster_id'].isin(selected_clusters)]

    grouped_pockets = actives_with_clusters.groupby('pocket_cluster_id')
    
    if len(grouped_pockets) == 0:
        print("Error: No active compounds have pocket cluster information. Cannot build structured dataset.")
        return
        
    samples_per_pocket_cluster = args.n_total_actives // len(grouped_pockets)

    if args.mode == 'easy':
        for pocket_cluster_id, pocket_group in tqdm(grouped_pockets, desc="Sampling Easy Actives"):
            pocket_group_with_lig_clusters = pocket_group.dropna(subset=['ligand_cluster_id'])
            if pocket_group_with_lig_clusters.empty: continue
            
            top_ligand_cluster = pocket_group_with_lig_clusters['ligand_cluster_id'].mode()[0]
            final_subset = pocket_group_with_lig_clusters[pocket_group_with_lig_clusters['ligand_cluster_id'] == top_ligand_cluster]
            if final_subset.empty: continue
            
            n_to_sample = min(samples_per_pocket_cluster, len(final_subset))
            sampled_actives = pd.concat([
                sampled_actives, 
                final_subset.sample(n=n_to_sample, random_state=args.random_seed)
            ])

    elif args.mode == 'hard':
        for pocket_cluster_id, pocket_group in tqdm(grouped_pockets, desc="Sampling Hard Actives"):
            pocket_group_with_lig_clusters = pocket_group.dropna(subset=['ligand_cluster_id'])
            if pocket_group_with_lig_clusters.empty: continue
            
            n_ligand_clusters = pocket_group_with_lig_clusters['ligand_cluster_id'].nunique()
            if n_ligand_clusters == 0: continue
            
            samples_per_ligand_cluster = max(1, samples_per_pocket_cluster // n_ligand_clusters)
            
            diverse_samples = pocket_group_with_lig_clusters.groupby('ligand_cluster_id').sample(
                n=samples_per_ligand_cluster, 
                replace=True, 
                random_state=args.random_seed,
            )
            sampled_actives = pd.concat([sampled_actives, diverse_samples])

    print(f"Sampled {len(sampled_actives)} active compounds.")

    if sampled_actives.empty:
        print("Could not sample any active compounds based on the specified mode. Aborting.")
        return

    n_inactives_to_sample = len(sampled_actives)
    if len(inactives_df) < n_inactives_to_sample:
        print(f"Warning: Not enough inactive compounds ({len(inactives_df)}) to create a fully balanced dataset. Using all available inactives.")
        n_inactives_to_sample = len(inactives_df)

    sampled_inactives = inactives_df.sample(n=n_inactives_to_sample, random_state=args.random_seed)
    print(f"Sampled {len(sampled_inactives)} inactive compounds.")
    
    final_dataset = pd.concat([sampled_actives, sampled_inactives])

    # --- 3. Create Final Train/Test Splits ---
    print("\nCreating final train/test splits...")
    train_df, test_df = stratified_scaffold_split(final_dataset, random_state=args.random_seed)
    
    if train_df.empty or test_df.empty:
        print("Error: Could not create valid train/test splits.")
        return
            
    final_cols = ['canonical_smiles', 'active', 'sequence']
    train_output = train_df[final_cols].rename(columns={'canonical_smiles': 'smiles'})
    test_output = test_df[final_cols].rename(columns={'canonical_smiles': 'smiles'})
    
    output_subdir = os.path.join(args.output_dir, args.mode)
    os.makedirs(output_subdir, exist_ok=True)
    
    train_output.to_csv(os.path.join(output_subdir, "train.csv"), index=False)
    test_output.to_csv(os.path.join(output_subdir, "test.csv"), index=False)
    
    print(f"\nDataset construction complete. Saved datasets to '{output_subdir}'")
    print(f"  - Train set size: {len(train_output)}")
    print(f"  - Test set size:  {len(test_output)}")

if __name__ == "__main__":
    main()


