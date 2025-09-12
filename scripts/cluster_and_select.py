import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import argparse

def cluster_and_select_representatives(
    metadata_file, n_clusters, output_file
):
    """
    Clusters binding pockets based on their physicochemical properties and selects
    a representative PDB ID from each cluster.

    Args:
        metadata_file (str): Path to the target_pocket_metadata.csv file.
        n_clusters (int): The number of representative pockets to select.
        output_file (str): Path to save the list of representative PDB IDs.
    """
    # Load the metadata
    df = pd.read_csv(metadata_file)

    # Select features for clustering
    # These are the numerical columns describing the pocket properties
    features = df.select_dtypes(include=np.number)

    # Store PDB IDs for later
    pdb_ids = df["pdb_id"]

    # Preprocessing: Fill NaN values and scale the features
    features = features.fillna(features.mean())
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Dimensionality Reduction with PCA
    # Reduce to a number of components that explains a good portion of the variance
    pca = PCA(n_components=0.95) # Retain 95% of the variance
    features_pca = pca.fit_transform(features_scaled)

    print(f"Original number of features: {features_scaled.shape[1]}")
    print(f"Reduced to {features_pca.shape[1]} components via PCA.")

    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_pca)

    # Find the representative PDB ID for each cluster (closest to the centroid)
    representatives, _ = pairwise_distances_argmin_min(
        kmeans.cluster_centers_, features_pca
    )

    representative_pdb_ids = pdb_ids.iloc[representatives].tolist()

    # Save the representative PDB IDs to a file
    with open(output_file, "w") as f:
        for pdb_id in representative_pdb_ids:
            f.write(f"{pdb_id}\n")

    print(f"\nSelected {len(representative_pdb_ids)} representative PDB IDs.")
    print(f"List of representatives saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster PDB pockets and select representatives."
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="../data/target_pocket_metadata.csv",
        help="Path to the target_pocket_metadata.csv file.",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=200,
        help="The number of representative clusters to generate.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="../data/representative_pdb_ids.txt",
        help="Path to save the output list of PDB IDs.",
    )

    args = parser.parse_args()

    cluster_and_select_representatives(
        args.metadata_file, args.n_clusters, args.output_file
    )
