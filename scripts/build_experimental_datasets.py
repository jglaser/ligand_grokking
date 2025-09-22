import argparse
import json
import os
from pathlib import Path

import polars as pl
from loguru import logger
from tqdm import tqdm
import numpy as np
import shutil

# Suppress RDKit warnings, as we handle them
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def define_and_deduplicate(df: pl.DataFrame, affinity_cols: list) -> pl.DataFrame:
    """Defines activity labels and removes duplicate SMILES."""
    logger.info("Defining activity labels and de-duplicating...")
    df = df.with_columns(pl.coalesce(affinity_cols).alias("affinity_nM")).filter(pl.col("affinity_nM").is_not_null() & (pl.col("affinity_nM") > 0))
    df = df.with_columns((-pl.col("affinity_nM").log10() + 9).alias("pchembl_value"))
    thresholds = df.group_by("uniprot_id").agg(pl.col("pchembl_value").median().alias("activity_threshold"))
    df = df.join(thresholds, on="uniprot_id", how="left")
    df = df.with_columns((pl.col("pchembl_value") >= pl.col("activity_threshold")).alias("is_active"))
    df = df.unique(subset=["uniprot_id", "canonical_smiles"], keep="first")
    return df

def sample_set(df_pool: pl.DataFrame, max_per_target: int | float | None, use_scaffolds: bool) -> pl.DataFrame:
    """Samples a dataframe to a max number of actives/inactives per target."""
    if max_per_target is None:
        return df_pool

    sampled_dfs = []
    original_columns = df_pool.columns

    for _, target_df in df_pool.group_by("uniprot_id"):
        actives = target_df.filter(pl.col("is_active"))
        inactives = target_df.filter(~pl.col("is_active"))
        
        sampled_actives = pl.DataFrame()
        if len(actives) > 0:
            if isinstance(max_per_target, float):
                n_samples = int(len(actives) * max_per_target)
                with_replacement = True
            else: # is int
                n_samples = min(len(actives), max_per_target)
                with_replacement = False

            if use_scaffolds and not with_replacement:
                scaffold_reps = actives.group_by("scaffold").head(1)
                remaining_actives = actives.join(scaffold_reps.select("canonical_smiles"), on="canonical_smiles", how="anti")
                
                n_from_scaffolds = min(len(scaffold_reps), n_samples)
                sampled_scaffold_actives = scaffold_reps.sample(n=n_from_scaffolds, shuffle=True)
                
                n_remaining = n_samples - n_from_scaffolds
                if n_remaining > 0 and len(remaining_actives) > 0:
                    additional_sample = remaining_actives.sample(n=min(len(remaining_actives), n_remaining), shuffle=True)
                    sampled_actives = pl.concat([
                        sampled_scaffold_actives.select(original_columns),
                        additional_sample.select(original_columns)
                    ])
                else:
                    sampled_actives = sampled_scaffold_actives
            else:
                sampled_actives = actives.sample(n=n_samples, with_replacement=with_replacement, shuffle=True)
        
        n_to_sample_inactives = len(sampled_actives)
        sampled_inactives = inactives.sample(n=min(len(inactives), n_to_sample_inactives), shuffle=True)
        
        if len(sampled_actives) > 0: sampled_dfs.append(sampled_actives.select(original_columns))
        if len(sampled_inactives) > 0: sampled_dfs.append(sampled_inactives.select(original_columns))

    return pl.concat(sampled_dfs) if sampled_dfs else pl.DataFrame()

def main(config_path: Path, delta_lake_path: Path, output_path: Path, hierarchy_labels_path: Path):
    with open(config_path) as f: configs = json.load(f)

    logger.info(f"Reading main data from {delta_lake_path}")
    main_df = pl.read_delta(delta_lake_path).filter(pl.col("uniprot_id").is_not_null())
    main_df = define_and_deduplicate(main_df, ["Ki (nM)", "IC50 (nM)", "Kd (nM)", "EC50 (nM)"])

    logger.info(f"Reading protein hierarchy from {hierarchy_labels_path}")
    hierarchy_df = pl.read_csv(hierarchy_labels_path)

    logger.info("Parsing hierarchy path string into columns...")
    path_splits = hierarchy_df["hierarchy_path"].str.split(";")
    max_level_in_file = len(path_splits[0]) + 1

    cluster_cols = [
        path_splits.list.get(k-2).cast(pl.Int32).alias(f"k{k}_cluster")
        for k in range(2, max_level_in_file + 1)
    ]
    hierarchy_df = hierarchy_df.with_columns(cluster_cols).drop("hierarchy_path")

    main_df = main_df.join(hierarchy_df, on="uniprot_id", how="inner")

    for config in configs:
        max_train = config.get("max_actives_per_target", None)
        max_test = config.get("max_test_actives_per_target", None)
        output_levels = set(config.get("hierarchy_levels", []))
        max_level_config = max(output_levels) if output_levels else 0

        max_level_to_cull = min(max_level_config, max_level_in_file)
        if max_level_config > max_level_in_file:
            logger.warning(
                f"Config requests culling up to k={max_level_config}, but file only contains levels up to k={max_level_in_file}. Culling will stop there."
            )

        n_folds_config = config.get("n_folds", 5)
        n_stochastic_splits = config.get("n_stochastic_splits", 1)

        logger.info(f"Generating {n_stochastic_splits} stochastic splits, culling up to k={max_level_to_cull}.")

        rng = np.random.default_rng(seed=42)

        for i in range(n_stochastic_splits):
            split_name = f"stochastic_split_{i+1}"
            split_path = output_path / split_name
            if split_path.exists(): continue
            os.makedirs(split_path)

            logger.info(f"--- Creating {split_name} ---")

            current_pool = main_df
            last_successful_test_pool = None
            last_successful_train_pool = None
            last_successful_k = -1

            for k in tqdm(range(2, max_level_to_cull + 1), desc="Culling Hierarchy"):
                if current_pool.is_empty(): break

                cluster_col = f"k{k}_cluster"

                available_clusters = current_pool[cluster_col].unique().drop_nulls().to_list()

                if len(available_clusters) > 1:
                    rng.shuffle(available_clusters)
                    n_to_hold_out = max(1, len(available_clusters) // n_folds_config)
                    test_cluster_ids = available_clusters[:n_to_hold_out]

                    test_pool = current_pool.filter(pl.col(cluster_col).is_in(test_cluster_ids))

                    uniprots_to_exclude = test_pool.select("uniprot_id")
                    train_pool = current_pool.join(uniprots_to_exclude, on="uniprot_id", how="anti")

                    last_successful_test_pool = test_pool
                    last_successful_train_pool = train_pool
                    last_successful_k = k

                    current_pool = train_pool

                if k in output_levels:
                    if last_successful_k != -1:
                        # --- DEFINITIVE FIX: Use explicit filenames for carried-over splits ---
                        from_k_str = f"_from_k{last_successful_k}" if last_successful_k != k else ""
                        if from_k_str:
                             logger.info(f"  - Level k={k} is unsplittable, using last successful split from k={last_successful_k}.")

                        test_df = sample_set(last_successful_test_pool, max_test, use_scaffolds=False)
                        train_df_level = sample_set(last_successful_train_pool, max_train, use_scaffolds=True)

                        if len(test_df) > 0:
                            test_df.write_parquet(split_path / f"k{k}{from_k_str}_test.parquet")

                        if len(train_df_level) > 0:
                            train_df_level.write_parquet(split_path / f"k{k}{from_k_str}_train.parquet")
                    else:
                        logger.warning(f"  - No successful splits occurred yet, cannot write output for k={k}.")

            # Save the final training set
            if last_successful_k != -1:
                final_train_df = sample_set(last_successful_train_pool, max_train, use_scaffolds=True)
                if len(final_train_df) > 0:
                    final_train_df.write_parquet(split_path / "train.parquet")
                    logger.info(f"  - Final train.parquet is the remainder from the k={last_successful_k} split.")

            logger.info(f"✅ Finished {split_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build stochastic, manifestly leak-proof datasets.")
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--delta_lake_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--hierarchy_labels_path", type=Path, default="protein_hierarchy_labels.csv")
    args = parser.parse_args()
    main(**vars(args))
