import argparse
import json
import os
from pathlib import Path

import polars as pl
from loguru import logger
from tqdm import tqdm


def get_protein_hierarchy_splits(hierarchy_df, k, n_folds):
    """
    Generates train/test UniProt ID splits for a given hierarchy level (k) and number of folds.
    """
    cluster_col = f"k{k}_cluster"
    if cluster_col not in hierarchy_df.columns:
        raise ValueError(f"Hierarchy for k={k} not found.")

    unique_clusters = hierarchy_df[cluster_col].unique().sort()
    cluster_to_fold = {
        cluster_id: (i % n_folds) for i, cluster_id in enumerate(unique_clusters)
    }

    for fold_idx in range(n_folds):
        test_clusters = [
            cid for cid, f_idx in cluster_to_fold.items() if f_idx == fold_idx
        ]
        if not test_clusters:
            continue

        test_uniprots = hierarchy_df.filter(pl.col(cluster_col).is_in(test_clusters))[
            "uniprot_id"
        ].to_list()
        train_uniprots = hierarchy_df.filter(
            ~pl.col(cluster_col).is_in(test_clusters)
        )["uniprot_id"].to_list()

        yield (fold_idx, train_uniprots, test_uniprots)


def define_and_deduplicate(df: pl.DataFrame, affinity_cols: list) -> pl.DataFrame:
    """
    Computes pchembl_value, defines a binary is_active label based on a
    per-UniProt ID threshold, and removes duplicate SMILES for each target.
    """
    logger.info("Defining activity labels and de-duplicating...")

    df = df.with_columns(
        pl.coalesce(affinity_cols).alias("affinity_nM")
    ).filter(pl.col("affinity_nM").is_not_null() & (pl.col("affinity_nM") > 0))

    df = df.with_columns(
        (-pl.col("affinity_nM").log10() + 9).alias("pchembl_value")
    )

    uniprot_thresholds = df.group_by("uniprot_id").agg(
        pl.col("pchembl_value").median().alias("activity_threshold")
    )

    df = df.join(uniprot_thresholds, on="uniprot_id", how="left")
    df = df.with_columns(
        (pl.col("pchembl_value") >= pl.col("activity_threshold")).alias("is_active")
    )
    
    n_before = len(df)
    df = df.unique(subset=["uniprot_id", "canonical_smiles"], keep="first")
    n_after = len(df)
    logger.info(f"Removed {n_before - n_after} duplicate SMILES per target.")

    return df


def main(
    config_path: Path,
    delta_lake_path: Path,
    output_path: Path,
    hierarchy_labels_path: Path,
):
    """
    Main function to build experimental datasets based on a config file.
    """
    with open(config_path) as f:
        configs = json.load(f)

    logger.info(f"Reading main data from Delta Lake at {delta_lake_path}")
    main_df = pl.read_delta(delta_lake_path)

    if "uniprot_id" not in main_df.columns:
        raise ValueError("Input Delta Lake table must contain a 'uniprot_id' column.")
    main_df = main_df.filter(pl.col("uniprot_id").is_not_null())

    affinity_cols = [
        "Ki (nM)", "IC50 (nM)", "Kd (nM)", "EC50 (nM)", "kon (M-1-s-1)", "koff (s-1)"
    ]
    main_df = define_and_deduplicate(main_df, affinity_cols)

    for config in configs:
        split_method = config["split_method"]
        max_actives_per_target = config.get("max_actives_per_target", None)

        if split_method == "protein_hierarchy":
            logger.info("Starting 'protein_hierarchy' split method.")
            if max_actives_per_target:
                logger.info(f"Applying limit of {max_actives_per_target} actives/inactives per target for training sets.")

            hierarchy_df = pl.read_csv(hierarchy_labels_path)
            hierarchy_levels = config.get("hierarchy_levels", [])
            n_folds_config = config.get("n_folds", 5)

            for k in hierarchy_levels:
                actual_n_folds = min(n_folds_config, k)
                if actual_n_folds < n_folds_config:
                    logger.warning(f"For k={k}, number of folds is capped at {k}.")
                
                logger.info(f"--- Generating splits for k={k} with {actual_n_folds} folds ---")
                split_generator = get_protein_hierarchy_splits(hierarchy_df, k, actual_n_folds)
                fold_iterator = tqdm(split_generator, total=actual_n_folds, desc=f"k={k} Folds")

                for fold_idx, train_uniprots, test_uniprots in fold_iterator:
                    split_name = f"protein_hierarchy_k{k}_fold{fold_idx+1}of{actual_n_folds}"
                    split_output_path = output_path / split_name

                    if split_output_path.exists():
                        logger.warning(f"Path {split_output_path} exists. Skipping.")
                        continue
                    os.makedirs(split_output_path)

                    test_df = main_df.filter(pl.col("uniprot_id").is_in(test_uniprots))
                    if len(test_df) > 0:
                        test_df.write_parquet(split_output_path / "test.parquet")
                    else:
                        logger.warning(f"Test set is empty for {split_name}.")

                    train_pool = main_df.filter(pl.col("uniprot_id").is_in(train_uniprots))
                    
                    if max_actives_per_target is None:
                        train_df = train_pool
                    else:
                        train_dfs_per_target = []
                        for target_id in train_uniprots:
                            target_df = train_pool.filter(pl.col("uniprot_id") == target_id)
                            actives = target_df.filter(pl.col("is_active"))
                            inactives = target_df.filter(~pl.col("is_active"))

                            sampled_actives = actives.sample(n=min(len(actives), max_actives_per_target), shuffle=True)
                            n_to_sample = len(sampled_actives)
                            sampled_inactives = inactives.sample(n=min(len(inactives), n_to_sample), shuffle=True)

                            if len(sampled_actives) > 0:
                                train_dfs_per_target.append(sampled_actives)
                            if len(sampled_inactives) > 0:
                                train_dfs_per_target.append(sampled_inactives)
                        
                        if not train_dfs_per_target:
                            train_df = pl.DataFrame()
                        else:
                            train_df = pl.concat(train_dfs_per_target)

                    if len(train_df) > 0:
                        train_df.write_parquet(split_output_path / "train.parquet")
                    else:
                        logger.warning(f"Train set is empty for {split_name}.")

                    logger.info(f"✅ Saved split: {split_name}")
        else:
            raise ValueError(f"Unknown split_method: '{split_method}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build experimental datasets from a Delta Lake table.")
    parser.add_argument("--config_path", type=Path, required=True, help="Path to the JSON config file.")
    parser.add_argument("--delta_lake_path", type=Path, required=True, help="Path to the input Delta Lake directory.")
    parser.add_argument("--output_path", type=Path, required=True, help="Root directory for output splits.")
    parser.add_argument("--hierarchy_labels_path", type=Path, default="protein_hierarchy_labels.csv", help="Path to the protein hierarchy labels file.")
    args = parser.parse_args()
    main(**vars(args))
