import argparse
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import numpy as np

def create_protein_hierarchy_labels(
    pocket_features_path: Path,
    metadata_path: Path,
    output_path: Path,
    max_clusters: int = 20
):
    """
    Performs hierarchical clustering at the UniProt ID level by aggregating
    pocket features from multiple PDB structures.

    This creates a lookup table of cluster memberships for multiple hierarchy
    levels (k=2 to max_clusters) for each unique protein.

    Args:
        pocket_features_path: Path to the fpocket_descriptors.csv file.
        metadata_path: Path to the target_pocket_metadata.csv file.
        output_path: Path to save the output CSV file with protein hierarchy labels.
        max_clusters: The maximum number of clusters to generate labels for.
    """
    print("Loading pocket features and metadata...")
    pocket_df = pd.read_csv(pocket_features_path)
    meta_df = pd.read_csv(metadata_path)

    print("Merging features with UniProt ID metadata...")
    merged_df = pd.merge(pocket_df, meta_df[['pdb_id', 'uniprot_id']], on='pdb_id')

    # Drop rows where uniprot_id is missing, as they cannot be clustered
    merged_df.dropna(subset=['uniprot_id'], inplace=True)
    
    # Identify feature columns (all columns except pdb_id and uniprot_id)
    feature_cols = [col for col in pocket_df.columns if col != 'pdb_id']
    
    print("Aggregating pocket features by UniProt ID (using mean)...")
    # Group by uniprot_id and calculate the mean for each feature column
    protein_features_df = merged_df.groupby('uniprot_id')[feature_cols].mean().reset_index()

    print(f"Aggregated {len(merged_df)} PDBs into {len(protein_features_df)} unique UniProt IDs.")

    # Prepare data for clustering
    uniprot_ids = protein_features_df['uniprot_id'].values
    features = protein_features_df[feature_cols].values

    print("Standardizing aggregated features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    print("Performing hierarchical clustering on UniProt IDs...")
    Z = linkage(scaled_features, method='ward')

    # Create a DataFrame to store the final results
    hierarchy_df = pd.DataFrame({'uniprot_id': uniprot_ids})

    print(f"Generating cluster labels for k=2 to k={max_clusters}...")
    for k in range(2, max_clusters + 1):
        labels = fcluster(Z, t=k, criterion='maxclust')
        hierarchy_df[f'k{k}_cluster'] = labels

    # Save the hierarchy to the specified output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hierarchy_df.to_csv(output_path, index=False)
    print(f"✅ Protein hierarchy labels saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate hierarchical cluster labels for UniProt IDs from pocket features."
    )
    parser.add_argument(
        "--pocket_features_path", type=Path, default="../data/fpocket_descriptors.csv"
    )
    parser.add_argument(
        "--metadata_path", type=Path, default="../data/target_pocket_metadata.csv"
    )
    parser.add_argument(
        "--output_path", type=Path, default="protein_hierarchy_labels.csv"
    )
    parser.add_argument(
        "--max_clusters", type=int, default=20
    )
    args = parser.parse_args()

    create_protein_hierarchy_labels(**vars(args))
