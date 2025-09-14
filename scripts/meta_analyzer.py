import argparse
import pandas as pd
import numpy as np

def main():
    """
    Performs a meta-analysis with a robust two-stage aggregation to correctly
    calculate pocket statistics for each UniProt target family.
    """
    parser = argparse.ArgumentParser(
        description="Analyze grokking and correlate it with pocket feature statistics.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("grokking_summary", type=str, help="Path to the grokking_analysis_summary.csv file.")
    parser.add_argument("pocket_metadata", type=str, help="Path to the original target_pocket_metadata.csv file.")
    parser.add_argument("fpocket_descriptors", type=str, help="Path to the fpocket_descriptors.csv file.")
    parser.add_argument("--output_file", type=str, default="grokking_meta_analysis.csv", help="Name for the final output CSV file.")

    args = parser.parse_args()

    # --- 1. Load All Data Sources ---
    try:
        grokking_df = pd.read_csv(args.grokking_summary)
        pocket_df_initial = pd.read_csv(args.pocket_metadata)
        fpocket_df = pd.read_csv(args.fpocket_descriptors)

        print(f"Loaded {len(grokking_df)} run entries from grokking summary.")
        print(f"Loaded {len(pocket_df_initial)} entries from initial pocket metadata.")
        print(f"Loaded {len(fpocket_df)} entries from fpocket descriptors.")

    except FileNotFoundError as e:
        print(f"Error: Could not find a required file: {e.filename}")
        return

    # --- 2. Correctly Calculate Pocket Statistics for Target Families ---

    # First, create a comprehensive feature table for ALL PDBs by merging the two feature files.
    # An inner join ensures we only consider PDBs that were successfully characterized by both methods.
    features_df = pd.merge(pocket_df_initial, fpocket_df, on='pdb_id', how='inner')
    print(f"Created a comprehensive feature set for {len(features_df)} PDBs with complete features.")

    # Second, identify the unique UniProt IDs that are actually in our experiment.
    grokking_df['pdb_id'] = grokking_df['run_name'].apply(lambda x: x.split('-seed')[0])
    # Map the PDBs from the grokking runs to their UniProt IDs using the initial metadata file.
    run_uniprot_map = pd.merge(grokking_df[['pdb_id']], pocket_df_initial[['pdb_id', 'uniprot_id']], on='pdb_id', how='left')
    uniprot_ids_in_runs = run_uniprot_map['uniprot_id'].dropna().unique()
    print(f"Identified {len(uniprot_ids_in_runs)} unique UniProt targets present in the grokking runs.")

    # Third, filter our comprehensive feature set to include all PDBs belonging to these target families.
    relevant_features_df = features_df[features_df['uniprot_id'].isin(uniprot_ids_in_runs)]

    # --- Data Sparsity Diagnostic ---
    pdb_counts_per_uniprot = relevant_features_df.groupby('uniprot_id')['pdb_id'].nunique()
    print("\n--- Data Sparsity Diagnostic ---")
    print("Number of PDBs per UniProt target (for all relevant targets):")
    print(pdb_counts_per_uniprot.value_counts().sort_index().to_string())
    print("--------------------------------\n")

    # Fourth, aggregate the features for these complete target families.
    feature_cols = [
        'num_residues', 'volume_A3', 'hydrophobic_sasa_nm2',
        'hbond_donors', 'hbond_acceptors', 'net_charge', 'polarity_score',
        'fpocket_drug_score', 'fpocket_volume', 'fpocket_hydrophobicity_score',
        'fpocket_polarity_score', 'fpocket_num_alpha_spheres'
    ]
    for col in feature_cols:
        if col in relevant_features_df.columns:
            relevant_features_df[col] = pd.to_numeric(relevant_features_df[col], errors='coerce')

    agg_funs = {col: ['mean', 'std'] for col in feature_cols}
    pocket_statistics = relevant_features_df.groupby('uniprot_id').agg(agg_funs).reset_index()

    pocket_statistics.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in pocket_statistics.columns.values]
    pocket_statistics.rename(columns={'uniprot_id_': 'uniprot_id'}, inplace=True)
    
    std_cols = [col for col in pocket_statistics.columns if col.endswith('_std')]
    pocket_statistics[std_cols] = pocket_statistics[std_cols].fillna(0)
    print(f"Calculated pocket statistics for {len(pocket_statistics)} UniProt families.")

    # --- 3. Separately, Aggregate Grokking Results ---
    # We use the previously created run_uniprot_map to ensure we group by the correct UniProt ID for each run.
    grokking_with_uniprot = pd.merge(grokking_df, run_uniprot_map.drop_duplicates(), on='pdb_id', how='left').dropna(subset=['uniprot_id'])
    
    grokking_summary = grokking_with_uniprot.groupby('uniprot_id').agg(
        grokking_frequency=('grokking_detected', 'mean'),
        average_grokking_delay=('grokking_delay', lambda x: x[x != -1].mean()),
        average_initial_generalization_epoch=('initial_generalization_epoch', lambda x: x[x != -1].mean())
    ).reset_index()
    
    # --- 4. Final Merge for Analysis ---
    meta_df = pd.merge(grokking_summary, pocket_statistics, on='uniprot_id')
    print(f"Final merged analysis contains {len(meta_df)} UniProt targets.")

    # --- 5. Perform Correlation Analysis ---
    mean_feature_cols = [col for col in meta_df.columns if '_mean' in col]
    std_feature_cols = [col for col in std_cols if meta_df[col].var() > 1e-9]

    if not std_feature_cols:
        print("\nWarning: No standard deviation columns have enough variance for correlation analysis.")
    
    analysis_feature_cols = mean_feature_cols + std_feature_cols

    # Part 1: Grokking Frequency
    print("\n--- Meta-Analysis Part 1: Grokking Frequency (All Targets) ---")
    if len(meta_df) > 1:
        freq_correlations = meta_df[['grokking_frequency'] + analysis_feature_cols].corr(numeric_only=True)['grokking_frequency'].drop('grokking_frequency')
        print("\nCorrelation with Grokking Frequency:")
        print(freq_correlations.sort_values(ascending=False).to_string())

    # Part 2: Grokking Delay
    print("\n--- Meta-Analysis Part 2: Grokking Delay (Grokking Subset) ---")
    grokking_subset_df = meta_df[meta_df['grokking_frequency'] > 0].dropna(subset=['average_grokking_delay'])
    if len(grokking_subset_df) > 1:
        delay_correlations = grokking_subset_df[['average_grokking_delay'] + analysis_feature_cols].corr(numeric_only=True)['average_grokking_delay'].drop('average_grokking_delay')
        print(f"\nAnalyzing {len(grokking_subset_df)} targets that grokked at least once.")
        print("\nCorrelation with Average Grokking Delay:")
        print(delay_correlations.sort_values(ascending=False).to_string())

    # Part 3: Initial Generalization Speed
    print("\n--- Meta-Analysis Part 3: Initial Generalization Speed (Generalizing Subset) ---")
    generalizing_subset_df = meta_df.dropna(subset=['average_initial_generalization_epoch'])
    if len(generalizing_subset_df) > 1:
        gen_correlations = generalizing_subset_df[['average_initial_generalization_epoch'] + analysis_feature_cols].corr(numeric_only=True)['average_initial_generalization_epoch'].drop('average_initial_generalization_epoch')
        print(f"\nAnalyzing {len(generalizing_subset_df)} targets that generalized.")
        print("\nCorrelation with Average Initial Generalization Epoch:")
        print(gen_correlations.sort_values(ascending=False).to_string())

    # --- 6. Save Full Data Table ---
    meta_df.to_csv(args.output_file, index=False)
    print(f"\nFull merged analysis table saved to: {args.output_file}")


if __name__ == "__main__":
    main()
