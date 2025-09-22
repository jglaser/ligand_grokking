import argparse
import polars as pl
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from loguru import logger
from pathlib import Path

def create_protein_hierarchy_labels(
    pocket_features_path: Path,
    metadata_path: Path,
    output_path: Path,
    max_clusters: int = 50
):
    """
    Performs hierarchical clustering at the UniProt ID level and saves a single,
    compact, semicolon-delimited hierarchy path string for each protein.
    """
    logger.info("Loading pocket features and metadata...")
    # Assuming the first column of pocket_features_path is the target identifier
    pocket_df = pl.read_csv(pocket_features_path)
    meta_df = pl.read_csv(metadata_path)

    logger.info("Merging features with UniProt ID metadata...")
    merged_df = pocket_df.join(meta_df.select(["pdb_id", "uniprot_id"]), on='pdb_id', how="inner")
    feature_cols = [col for col in pocket_df.columns if col != 'pdb_id']
    
    logger.info("Aggregating pocket features by UniProt ID...")
    protein_features_df = merged_df.group_by('uniprot_id').agg(
        [pl.mean(col) for col in feature_cols]
    ).sort("uniprot_id")

    uniprot_ids = protein_features_df['uniprot_id']
    features = protein_features_df.drop('uniprot_id').to_numpy()

    logger.info("Standardizing and clustering features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    Z = linkage(scaled_features, method='ward')

    logger.info(f"Generating hierarchy paths for k=2 to k={max_clusters}...")
    # Generate all cluster labels into a matrix
    # Each column corresponds to a k-level (k=2, k=3, ...)
    all_labels_matrix = [
        fcluster(Z, t=k, criterion='maxclust')
        for k in range(2, max_clusters + 1)
    ]

    # Transpose and convert to strings to create the path for each protein
    path_strings = [
        ";".join(map(str, row))
        for row in zip(*all_labels_matrix)
    ]

    # Create the final, lean DataFrame
    hierarchy_df = pl.DataFrame({
        "uniprot_id": uniprot_ids,
        "hierarchy_path": path_strings
    })

    hierarchy_df.write_csv(output_path)
    logger.info(f"✅ Lean protein hierarchy labels saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create lean protein hierarchy labels.")
    parser.add_argument("pocket_features_path", type=Path)
    parser.add_argument("metadata_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--max_clusters", type=int, default=5000)
    args = parser.parse_args()
    create_protein_hierarchy_labels(**vars(args))
