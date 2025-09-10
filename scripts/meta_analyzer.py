import argparse
import pandas as pd

def main():
    """
    Performs a meta-analysis by combining grokking results with pocket metadata
    to find features predictive of the grokking phenomenon.
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

    grokking_df['pdb_id'] = grokking_df['run_name'].apply(lambda x: x.split('-seed-')[0])

    grokking_summary = grokking_df.groupby('pdb_id').agg(
        grokking_frequency=('grokking_detected', 'mean'),
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

    # --- 3. Correlation Analysis ---
    numeric_cols = ['grokking_frequency', 'average_grokking_delay', 'num_residues', 'volume_A3', 
                    'hydrophobic_sasa_nm2', 'hbond_donors', 'hbond_acceptors', 'net_charge', 'polarity_score']
    
    for col in numeric_cols:
        if col in meta_df.columns:
            meta_df[col] = pd.to_numeric(meta_df[col], errors='coerce')
    meta_df.dropna(subset=numeric_cols, inplace=True)

    if len(meta_df) < 2:
        print("Not enough overlapping data for correlation analysis.")
        return

    correlations = meta_df[numeric_cols].corr()

    # --- 4. Save and Display Results ---
    meta_df.to_csv(args.output_file, index=False)
    
    print("\n--- Meta-Analysis Summary ---")
    print("\nCorrelation with Grokking Frequency:")
    print(correlations['grokking_frequency'].drop('grokking_frequency').to_string())
    
    print("\nCorrelation with Average Grokking Delay:")
    print(correlations['average_grokking_delay'].drop('average_grokking_delay').to_string())

    print(f"\nFull merged analysis table saved to: {args.output_file}")

if __name__ == "__main__":
    main()


