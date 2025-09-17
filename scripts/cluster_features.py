import argparse
import pandas as pd
import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from tqdm.auto import tqdm
import concurrent.futures

def cluster_uniprot_pockets(pocket_features_df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """
    Aggregates pocket features by UniProt ID and then clusters these aggregated
    feature vectors to group entire protein families by their pocket characteristics.
    """
    print(f"\nAggregating pocket features for {pocket_features_df['uniprot_id'].nunique()} unique UniProt IDs...")
    
    feature_cols = pocket_features_df.select_dtypes(include=np.number).columns.tolist()
    if 'pdb_id' in feature_cols: # Don't aggregate the PDB ID itself
        feature_cols.remove('pdb_id')
        
    # Define aggregation functions: mean and standard deviation for each feature
    agg_funs = {col: ['mean', 'std'] for col in feature_cols}
    uniprot_features = pocket_features_df.groupby('uniprot_id').agg(agg_funs).reset_index()

    # Flatten the multi-level column names (e.g., ('volume_A3', 'mean') -> 'volume_A3_mean')
    uniprot_features.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in uniprot_features.columns.values]
    uniprot_features.rename(columns={'uniprot_id_': 'uniprot_id'}, inplace=True)
    
    # Fill NaN for std dev in single-entry groups with 0 (no variation)
    std_cols = [col for col in uniprot_features.columns if col.endswith('_std')]
    uniprot_features[std_cols] = uniprot_features[std_cols].fillna(0)
    
    print(f"\nClustering {len(uniprot_features)} UniProt targets into {n_clusters} clusters...")
    
    # Select only the aggregated feature columns for clustering
    agg_feature_cols = [col for col in uniprot_features.columns if col != 'uniprot_id']
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(uniprot_features[agg_feature_cols])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    uniprot_features['pocket_cluster_id'] = kmeans.fit_predict(features_scaled)
    
    print("UniProt pocket clustering complete.")
    return uniprot_features[['uniprot_id', 'pocket_cluster_id']]

# (The ligand clustering functions and main logic for loading data remain the same)
# ... [rest of the script is unchanged but included for completeness] ...

def _calculate_fingerprint_worker(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            gen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
            fp_array = np.array(gen.GetFingerprint(mol), dtype=np.int8)
            return smiles, fp_array, None
        else:
            return smiles, None, "RDKit.Chem.MolFromSmiles returned None"
    except Exception as e:
        return smiles, None, str(e)

def cluster_ligands_kmeans(smiles_list: list, n_clusters: int) -> pd.DataFrame:
    print(f"\nGenerating fingerprints for {len(smiles_list)} unique ligands in parallel...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(_calculate_fingerprint_worker, smiles_list), total=len(smiles_list), desc="Generating Fingerprints"))

    valid_results = [res for res in results if res[1] is not None]
    valid_smiles = [smi for smi, fp, err in valid_results]
    valid_fps_np = np.array([fp for smi, fp, err in valid_results], dtype=np.float32)

    print(f"\nClustering {len(valid_fps_np)} valid ligands into {n_clusters} clusters using MiniBatchKMeans...")
    num_cores = os.cpu_count() or 1
    recommended_batch_size = 256 * num_cores
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=42, batch_size=recommended_batch_size,
        init='random', n_init=20, max_iter=300, verbose=True
    )
    cluster_ids = kmeans.fit_predict(valid_fps_np)
    smiles_to_cluster_df = pd.DataFrame({'canonical_smiles': valid_smiles, 'ligand_cluster_id': cluster_ids})
    print(f"Ligand clustering complete.")
    return smiles_to_cluster_df

def canonicalize_smiles_worker(smiles: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        pass
    return None

def main():
    parser = argparse.ArgumentParser(description="Cluster pockets and ligands to create a feature space for dataset construction.")
    parser.add_argument("pocket_metadata", type=str, help="Path to the original target_pocket_metadata.csv file.")
    parser.add_argument("fpocket_descriptors", type=str, help="Path to the fpocket_descriptors.csv file.")
    parser.add_argument("bindingdb_delta_path", type=str, help="Path to the BindingDB Delta Lake directory.")
    parser.add_argument("--n_pocket_clusters", type=int, default=100, help="Number of clusters to create for UniProt pocket feature distributions.")
    parser.add_argument("--n_ligand_clusters", type=int, default=5000, help="Number of clusters to create for ligands using K-Means.")
    
    args = parser.parse_args()

    # --- 1. UniProt Pocket Clustering ---
    try:
        pocket_df_initial = pd.read_csv(args.pocket_metadata)
        fpocket_df = pd.read_csv(args.fpocket_descriptors)
        features_df = pd.merge(pocket_df_initial, fpocket_df, on='pdb_id', how='inner')
    except FileNotFoundError as e:
        print(f"Error loading feature files: {e}")
        return

    uniprot_pocket_clusters_df = cluster_uniprot_pockets(features_df, args.n_pocket_clusters)
    uniprot_pocket_clusters_df.to_csv("uniprot_pocket_clusters.csv", index=False)
    print("Saved UniProt pocket cluster assignments to uniprot_pocket_clusters.csv")

    # --- 2. Ligand Clustering (Unchanged) ---
    try:
        print(f"\nLoading raw SMILES from {args.bindingdb_delta_path}...")
        binding_lazy = pl.scan_delta(args.bindingdb_delta_path)
        raw_smiles_list = binding_lazy.select(pl.col('Ligand SMILES')).collect()['Ligand SMILES'].drop_nulls().unique().to_list()
        
        print("Canonicalizing SMILES in parallel...")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(canonicalize_smiles_worker, raw_smiles_list), total=len(raw_smiles_list), desc="Canonicalizing SMILES"))
        
        unique_canonical_smiles = {smi for smi in results if smi is not None}
        print(f"Found {len(unique_canonical_smiles)} unique, canonical SMILES strings to be clustered.")
        
    except Exception as e:
        print(f"Error loading or processing ligand file from Delta Lake: {e}")
        return
        
    ligand_clusters_df = cluster_ligands_kmeans(list(unique_canonical_smiles), args.n_ligand_clusters)
    ligand_clusters_df.to_csv("ligand_clusters.csv", index=False)
    print("Saved ligand cluster assignments to ligand_clusters.csv")

if __name__ == "__main__":
    main()


