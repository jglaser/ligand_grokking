import argparse
import pandas as pd

def main():
    """
    Performs a two-stage meta-analysis by first correlating grokking frequency
    across all targets, and then correlating grokking delay only for the
    subset of targets that were observed to grok.
    """
    parser = argparse.ArgumentParser(
        description="Analyze grokking frequency and delay, and find predictive pocket features.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("grokking_summary", type=str, help="Path to the grokking_analysis_summary.csv file.")
    parser.add_argument("pocket_metadata", type=str, help="Path to the target_pocket_metadata.csv file.")
    parser.add_argument("--output_file", type=str, default="grokking_meta_analysis.csv", help="Name for the final output CSV file.")

    args = parser.parse_args()

    # --- 1. Load and Process Grokking Results ---
    try:
        grokking_df = pd.read_csv(args.grokking_summary)
    except FileNotFoundError:
        print(f"Error: Grokking summary file not found at '{args.grokking_summary}'")
        return

    grokking_df['pdb_id'] = grokking_df['run_name'].apply(lambda x: x.split('-seed')[0])

    grokking_summary = grokking_df.groupby('pdb_id').agg(
        grokking_frequency=('grokking_detected', 'mean'),
        # The mean of an empty series will be NaN for non-grokking targets
        average_grokking_delay=('grokking_delay', lambda x: x[x != -1].mean())
    ).reset_index()
    
    print(f"Summarized results for {len(grokking_summary)} unique targets.")

    # --- 2. Load and Merge Pocket Metadata ---
    try:
        metadata_df = pd.read_csv(args.pocket_metadata).rename(columns={'target_name': 'pdb_id'})
    except FileNotFoundError:
        print(f"Error: Pocket metadata file not found at '{args.pocket_metadata}'")
        return

    meta_df = pd.merge(grokking_summary, metadata_df, on='pdb_id')
    if meta_df.empty:
        print("Error: Merge resulted in an empty DataFrame. Check that PDB IDs match.")
        return

    # --- 3. First Analysis: Grokking Frequency (All Targets) ---
    print("\n--- Meta-Analysis Part 1: Grokking Frequency (All Targets) ---")
    
    # Define the columns we want to correlate against
    feature_cols = ['num_residues', 'volume_A3', 'hydrophobic_sasa_nm2', 
                    'hbond_donors', 'hbond_acceptors', 'net_charge', 'polarity_score']

    # Ensure all feature columns are numeric for correlation
    df_for_freq_analysis = meta_df.copy()
    for col in feature_cols:
        if col in df_for_freq_analysis.columns:
            df_for_freq_analysis[col] = pd.to_numeric(df_for_freq_analysis[col], errors='coerce')
    df_for_freq_analysis.dropna(subset=feature_cols, inplace=True)
    
    print(f"Analyzing {len(df_for_freq_analysis)} targets with complete metadata.")
    
    if len(df_for_freq_analysis) > 1:
        freq_correlations = df_for_freq_analysis[['grokking_frequency'] + feature_cols].corr()['grokking_frequency'].drop('grokking_frequency')
        print("\nCorrelation with Grokking Frequency:")
        print(freq_correlations.to_string())
    else:
        print("Not enough data for frequency correlation analysis.")


    # --- 4. Second Analysis: Grokking Delay (Grokking Subset Only) ---
    print("\n--- Meta-Analysis Part 2: Grokking Delay (Grokking Subset) ---")
    
    # Create a subset of targets that grokked at least once
    grokking_subset_df = meta_df[meta_df['grokking_frequency'] > 0].copy()
    
    # Clean and prepare this smaller subset for correlation
    for col in feature_cols:
         if col in grokking_subset_df.columns:
            grokking_subset_df[col] = pd.to_numeric(grokking_subset_df[col], errors='coerce')
    grokking_subset_df.dropna(subset=['average_grokking_delay'] + feature_cols, inplace=True)

    print(f"Analyzing {len(grokking_subset_df)} targets that grokked at least once.")
    
    if len(grokking_subset_df) > 1:
        delay_correlations = grokking_subset_df[['average_grokking_delay'] + feature_cols].corr()['average_grokking_delay'].drop('average_grokking_delay')
        print("\nCorrelation with Average Grokking Delay:")
        print(delay_correlations.to_string())
    else:
        print("Not enough data for delay correlation analysis (fewer than 2 targets grokked).")


    # --- 5. Save Full Data Table ---
    meta_df.to_csv(args.output_file, index=False)
    print(f"\nFull merged analysis table saved to: {args.output_file}")


if __name__ == "__main__":
    main()


