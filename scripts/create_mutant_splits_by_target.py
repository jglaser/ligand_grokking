import argparse
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import os

def main(args):
    """
    Creates disjoint train/test splits for the mutant resistance dataset,
    grouping by the wild-type uniprot_id to test generalization.
    """
    print("--- Creating Disjoint Mutant Dataset Splits by Target ---")

    # --- 1. Load Input Data ---
    print(f"Loading mutant dataset from: {args.mutant_data_path}")
    mutant_df = pd.read_parquet(args.mutant_data_path)
    print(f"Loaded {len(mutant_df)} total measurements.")
    
    unique_targets = mutant_df['uniprot_id'].nunique()
    print(f"Found {unique_targets} unique protein targets (UniProt IDs).")

    # --- 2. Prepare Data for Splitting ---
    # The dataframe itself can be considered X, as we only need indices.
    # y is the label we want to stratify on.
    # groups are the uniprot_ids that must not be split across train/test.
    X = mutant_df
    y = mutant_df['confers_resistance']
    groups = mutant_df['uniprot_id'] 

    # --- 3. Create Splits using StratifiedGroupKFold ---
    # This splitter ensures that all members of a protein group (uniprot_id)
    # stay in the same set (train or test) while also preserving the
    # percentage of samples for each class (stratification).
    sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_seed)

    print(f"\nGenerating {args.n_splits} disjoint splits, grouped by 'uniprot_id'...")
    
    os.makedirs(args.output_dir, exist_ok=True)

    fold_counter = 0
    for train_idx, test_idx in sgkf.split(X, y, groups):
        fold_counter += 1
        print(f"--- Fold {fold_counter}/{args.n_splits} ---")
        
        train_df = mutant_df.iloc[train_idx]
        test_df = mutant_df.iloc[test_idx]
        
        # --- 4. Report and Save Splits ---
        print(f"  Train set size: {len(train_df)} ({train_df['uniprot_id'].nunique()} unique proteins)")
        print(f"  Test set size: {len(test_df)} ({test_df['uniprot_id'].nunique()} unique proteins)")
        print(f"  Train class distribution:\n{train_df['confers_resistance'].value_counts(normalize=True).to_string()}")
        print(f"  Test class distribution:\n{test_df['confers_resistance'].value_counts(normalize=True).to_string()}")

        # Verify that there is no overlap in protein uniprot_ids between train and test
        train_targets = set(train_df['uniprot_id'])
        test_targets = set(test_df['uniprot_id'])
        assert train_targets.isdisjoint(test_targets), f"Fold {fold_counter} has overlapping targets!"
        print(f"  Validation successful: No UniProt ID overlap between train and test sets.")
        
        # Save the splits to parquet files
        train_path = os.path.join(args.output_dir, f"fold_{fold_counter}_train.parquet")
        test_path = os.path.join(args.output_dir, f"fold_{fold_counter}_test.parquet")
        
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        print(f"  Saved splits to '{train_path}' and '{test_path}'")
        
    print("\n--- Splitting Complete ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create disjoint train/test splits for the mutant dataset based on UniProt ID.")
    parser.add_argument('--mutant_data_path', type=str, required=True, help='Path to the mutant_resistance_dataset.parquet file.')
    parser.add_argument('--output_dir', type=str, default='./mutant_splits_by_target', help='Directory to save the train/test split files.')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of folds to generate (K in K-fold).')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility.')
    
    args = parser.parse_args()
    main(args)

