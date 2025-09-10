import argparse
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count

def find_grokking_point(epochs, train_acc, val_acc,
                        overfit_threshold=0.9, plateau_threshold=0.7,
                        jump_threshold=0.05, window_size=50):
    """
    Analyzes a training trajectory to find memorization and grokking events.

    Returns:
        A tuple: (memorization_epoch, grokking_epoch). Returns (-1, -1) if events are not found.
    """
    if len(val_acc) < window_size * 2:
        return -1, -1

    # Smooth the validation accuracy to find a stable jump
    val_acc_smooth = pd.Series(val_acc).rolling(window=window_size, min_periods=1, center=True).mean().values
    
    # --- Find Memorization Point ---
    memorization_epoch = -1
    # Find the first epoch where training accuracy is sustainably high
    sustained_high_train = (pd.Series(train_acc).rolling(window=window_size).mean() > overfit_threshold)
    first_high_train_idx = sustained_high_train.idxmax()
    if sustained_high_train[first_high_train_idx]: # Check if it's a true positive
        memorization_epoch = epochs[first_high_train_idx]

    if memorization_epoch == -1:
        return -1, -1 # Can't grok if it never memorizes

    # --- Find Grokking Point (must occur after memorization) ---
    in_plateau = False
    for i in range(first_high_train_idx, len(epochs) - window_size):
        # Condition 1: Is validation accuracy in the "confused" plateau?
        if val_acc_smooth[i] < plateau_threshold:
            in_plateau = True

        # Condition 2: If in the plateau, look for a significant, sustained jump
        if in_plateau:
            jump_size = val_acc_smooth[i + window_size] - val_acc_smooth[i]
            if jump_size > jump_threshold:
                grokking_epoch = epochs[i]
                return memorization_epoch, grokking_epoch
                
    return memorization_epoch, -1

def read_tensorboard_log(log_dir: str) -> pd.DataFrame:
    """Reads scalar data from a TensorBoard log directory into a pandas DataFrame."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(log_dir, size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        required_tags = {'train_accuracy', 'validation_accuracy'}
        if not required_tags.issubset(ea.Tags()['scalars']): return None
        
        df_train = pd.DataFrame([(e.step, e.value) for e in ea.Scalars('train_accuracy')], columns=['epoch', 'train_accuracy'])
        df_val = pd.DataFrame([(e.step, e.value) for e in ea.Scalars('validation_accuracy')], columns=['epoch', 'validation_accuracy'])
        
        history_df = pd.merge(df_train, df_val, on='epoch', how='outer').sort_values('epoch').ffill().dropna()
        return history_df
    except Exception:
        return None

def analyze_log_worker(run_dir_path: str):
    """Worker function to analyze a single TensorBoard run."""
    run_name = os.path.basename(run_dir_path)
    try:
        history = read_tensorboard_log(run_dir_path)
        if history is None or history.empty or len(history) < 100:
            return None

        mem_epoch, grok_epoch = find_grokking_point(
            history["epoch"].values,
            history["train_accuracy"].values,
            history["validation_accuracy"].values
        )
        
        delay = grok_epoch - mem_epoch if grok_epoch != -1 and mem_epoch != -1 else -1

        return {
            'run_name': run_name,
            'memorization_epoch': mem_epoch,
            'grokking_epoch': grok_epoch,
            'grokking_delay': delay,
            'grokking_detected': grok_epoch != -1
        }
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Analyze grokking frequency and delay, and find predictive pocket features.")
    parser.add_argument("log_dir", type=str, help="Path to the local TensorBoard log directory ('logs/').")
    parser.add_argument("pocket_metadata", type=str, help="Path to the target_pocket_metadata.csv file.")
    parser.add_argument("--output_file", type=str, default="grokking_meta_analysis.csv", help="Name for the final output CSV file.")
    args = parser.parse_args()

    # --- 1. Analyze all runs in parallel ---
    run_dirs = [os.path.join(args.log_dir, d) for d in sorted(os.listdir(args.log_dir)) if os.path.isdir(os.path.join(args.log_dir, d))]
    print(f"Found {len(run_dirs)} potential runs in '{args.log_dir}'. Analyzing in parallel...")
    
    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(analyze_log_worker, run_dirs), total=len(run_dirs), desc="Analyzing Logs"))
    
    analysis_results = [r for r in results if r is not None]
    if not analysis_results:
        print("No valid runs with sufficient data were found to analyze.")
        return

    grokking_df = pd.DataFrame(analysis_results)
    grokking_df['pdb_id'] = grokking_df['run_name'].apply(lambda x: x.split('-seed-')[0])

    # --- 2. Aggregate Results per Target ---
    grokking_summary = grokking_df.groupby('pdb_id').agg(
        grokking_frequency=('grokking_detected', 'mean'),
        # Calculate average delay only for runs that actually grokked
        average_grokking_delay=('grokking_delay', lambda x: x[x != -1].mean())
    ).reset_index()

    print(f"Summarized results for {len(grokking_summary)} unique targets.")

    # --- 3. Load and Merge Pocket Metadata ---
    try:
        metadata_df = pd.read_csv(args.pocket_metadata).rename(columns={'target_name': 'pdb_id'})
    except FileNotFoundError:
        print(f"Error: Pocket metadata file not found at '{args.pocket_metadata}'")
        return

    meta_df = pd.merge(grokking_summary, metadata_df, on='pdb_id')
    if meta_df.empty:
        print("Error: Merge resulted in an empty DataFrame. Check that PDB IDs match.")
        return

    # --- 4. Correlation Analysis ---
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

    # --- 5. Save and Display Results ---
    meta_df.to_csv(args.output_file, index=False)
    
    print("\n--- Meta-Analysis Summary ---")
    print("\nCorrelation with Grokking Frequency:")
    print(correlations['grokking_frequency'].drop('grokking_frequency').to_string())
    
    print("\nCorrelation with Average Grokking Delay:")
    print(correlations['average_grokking_delay'].drop('average_grokking_delay').to_string())

    print(f"\nFull merged analysis table saved to: {args.output_file}")

if __name__ == "__main__":
    main()


