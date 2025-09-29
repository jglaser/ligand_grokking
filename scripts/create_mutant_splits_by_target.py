import argparse
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import os
import numpy as np

def print_split_stats(df, name):
    """Calculates and prints statistics for a given dataframe split."""
    if df.empty:
        print(f"  {name} set is empty.")
        return
        
    num_measurements = len(df)
    # Corrected typo: uniprot_id
    num_proteins = df['uniprot_id'].nunique()
    
    # --- FIX: Handle integer (1/0) or boolean (True/False) labels ---
    class_counts = df['confers_resistance'].value_counts()
    # Get counts for 1 (True) and 0 (False) to handle integer-based booleans
    resistant_count = class_counts.get(1, class_counts.get(True, 0))
    non_resistant_count = class_counts.get(0, class_counts.get(False, 0))
    # --- END FIX ---
    
    if num_measurements > 0:
        resistance_ratio = resistant_count / num_measurements * 100
    else:
        resistance_ratio = 0

    print(f"  - {name} Set Stats:")
    print(f"    - Total Measurements: {num_measurements}")
    print(f"    - Unique Proteins:    {num_proteins}")
    print(f"    - Class Balance:      {resistance_ratio:.2f}% Resistant ({resistant_count} Resistant / {non_resistant_count} Non-Resistant)")

def main(args):
    """
    Creates disjoint train/test splits for the mutant resistance dataset,
    grouping by the wild-type uniprot_id to test generalization.
    First, it filters the dataset to include only proteins present in the feature file.
    """
    print("--- Creating Disjoint Mutant Dataset Splits by Target ---")

    # --- 1. Load Feature File to Get Valid Targets ---
    print(f"Loading feature keys from: {args.feature_file}")
    with np.load(args.feature_file) as feature_data:
        feature_keys = set(feature_data.keys())
    
    valid_uniprot_ids = set(key.split('_')[0] for key in feature_keys)
    print(f"Found {len(valid_uniprot_ids)} unique protein targets with features.")

    # --- 2. Load and Filter Input Data ---
    print(f"Loading mutant dataset from: {args.mutant_data_path}")
    mutant_df = pd.read_parquet(args.mutant_data_path)
    print(f"Loaded {len(mutant_df)} total measurements for {mutant_df['uniprot_id'].nunique()} targets.")
    
    initial_rows = len(mutant_df)
    mutant_df = mutant_df[mutant_df['uniprot_id'].isin(valid_uniprot_ids)].copy()
    print(f"Filtered dataset to {len(mutant_df)} measurements (from {initial_rows}) based on feature file.")
    
    if len(mutant_df) == 0:
        print("\n❌ Error: No data remains after filtering. Please check for mismatches in UniProt IDs between your data and feature files.")
        return
        
    unique_targets = mutant_df['uniprot_id'].nunique()
    print(f"Now proceeding with {unique_targets} unique protein targets.")

    # --- 3. Prepare Data for Splitting ---
    X = mutant_df
    y = mutant_df['confers_resistance']
    groups = mutant_df['uniprot_id'] 

    # --- 4. Create Splits using StratifiedGroupKFold ---
    sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nGenerating {args.n_splits} stratified, grouped folds...")
    print("-" * 60)
    
    for fold_counter, (train_index, test_index) in enumerate(sgkf.split(X, y, groups)):
        print(f"▶ Fold {fold_counter}:")
        
        train_df = X.iloc[train_index]
        test_df = X.iloc[test_index]
        
        print_split_stats(train_df, "Train")
        print_split_stats(test_df, "Test")
        
        train_targets = set(train_df['uniprot_id'])
        test_targets = set(test_df['uniprot_id'])
        assert train_targets.isdisjoint(test_targets), f"Fold {fold_counter} has overlapping targets!"
        
        train_path = os.path.join(args.output_dir, f"fold_{fold_counter}_train.parquet")
        test_path = os.path.join(args.output_dir, f"fold_{fold_counter}_test.parquet")
        
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        print("-" * 60)
        
    print("\n✅ Splitting Complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create disjoint train/test splits for the mutant dataset based on UniProt ID.")
    parser.add_argument('--mutant_data_path', type=str, required=True, help='Path to the mutant_resistance_dataset.parquet file.')
    parser.add_argument('--feature_file', type=str, required=True, help='Path to the .npz feature file used for filtering targets.')
    parser.add_argument('--output_dir', type=str, default='./mutant_splits_by_target', help='Directory to save the train/test split files.')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of folds to generate (K in K-fold).')
    
    args = parser.parse_args()
    main(args)
