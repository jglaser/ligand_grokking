import argparse
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
from scipy.stats import pearsonr

LOG_FREQ = 100

def find_initial_generalization_epoch(epochs, val_acc, generalization_threshold=0.75, window_size=20):
    window_steps = max(1, window_size // LOG_FREQ)
    if len(val_acc) < window_steps:
        return -1
    sustained_generalization = pd.Series(val_acc).rolling(window=window_steps).mean() > generalization_threshold
    true_indices = np.where(sustained_generalization)[0]
    if true_indices.size > 0:
        return int(epochs[true_indices[0]])
    else:
        return -1

def find_grokking_point_vectorized(epochs, train_acc, val_acc,
                                   overfit_threshold=0.9, plateau_threshold=0.75,
                                   jump_threshold=0.05, window_size=500,
                                   min_delay_epochs=500, sustain_window_multiplier=3):
    window_steps = max(1, window_size // LOG_FREQ)
    min_delay_steps = max(1, min_delay_epochs // LOG_FREQ)
    required_len = window_steps * (sustain_window_multiplier + 1)
    if len(val_acc) < required_len:
        return -1, -1

    val_acc_smooth = pd.Series(val_acc).rolling(window=window_steps, min_periods=1, center=True).mean().values
    sustained_high_train = (pd.Series(train_acc).rolling(window=window_steps).mean() > overfit_threshold)
    true_indices = np.where(sustained_high_train)[0]
    if len(true_indices) == 0: return -1, -1

    first_high_train_idx = true_indices[0]
    memorization_epoch = epochs[first_high_train_idx]
    search_start_idx = first_high_train_idx + 1
    search_end_idx = len(epochs) - (window_steps * sustain_window_multiplier)
    if search_start_idx >= search_end_idx: return memorization_epoch, -1

    candidate_indices = np.arange(search_start_idx, search_end_idx)
    in_plateau_mask = val_acc_smooth[candidate_indices] < plateau_threshold
    delay_mask = (candidate_indices - first_high_train_idx) > min_delay_steps
    jump_sizes = val_acc_smooth[candidate_indices + window_steps] - val_acc_smooth[candidate_indices]
    jump_mask = jump_sizes > jump_threshold
    
    post_jump_start_indices = candidate_indices + window_steps
    post_jump_end_indices = candidate_indices + window_steps * sustain_window_multiplier
    sustain_window_len = window_steps * (sustain_window_multiplier - 1)
    cumsum = np.concatenate(([0], np.cumsum(val_acc_smooth)))
    post_jump_sums = cumsum[post_jump_end_indices] - cumsum[post_jump_start_indices]
    avg_post_jump_acc = np.zeros_like(post_jump_sums, dtype=float)
    if sustain_window_len > 0: avg_post_jump_acc = post_jump_sums / sustain_window_len
    sustain_mask = avg_post_jump_acc > plateau_threshold

    full_mask = in_plateau_mask & delay_mask & jump_mask & sustain_mask
    valid_grok_indices = np.where(full_mask)[0]

    if valid_grok_indices.size > 0:
        first_grok_idx = candidate_indices[valid_grok_indices[0]]
        return memorization_epoch, epochs[first_grok_idx]
    else:
        return memorization_epoch, -1

def read_csv_log(log_file: str) -> pd.DataFrame | None:
    """Reads training history from a CSV file."""
    try:
        df = pd.read_csv(log_file)
        required_cols = {'epoch', 'train_auc', 'validation_auc'}
        if not required_cols.issubset(df.columns):
            return None
        return df.sort_values('epoch').ffill().dropna()
    except Exception:
        return None

def analyze_log_worker(log_path: str):
    """
    Worker to analyze a single CSV log file.
    """
    run_name = os.path.basename(log_path).replace('.csv', '')
    history = read_csv_log(log_path)
    
    if history is None: return {'run_name': run_name, 'status': 'failed_read'}
    if history.empty or len(history) < 100: return {'run_name': run_name, 'status': 'too_short'}
    
    epochs = history["epoch"].values
    train_acc = history["train_auc"].values
    val_acc = history["validation_auc"].values

    initial_gen_epoch = find_initial_generalization_epoch(epochs, val_acc)
    mem_epoch, grok_epoch = find_grokking_point_vectorized(epochs, train_acc, val_acc)
    delay = grok_epoch - mem_epoch if grok_epoch != -1 and mem_epoch != -1 else -1
    
    return {
        'run_name': run_name,
        'memorization_epoch': mem_epoch,
        'grokking_epoch': grok_epoch,
        'grokking_delay': delay,
        'grokking_detected': grok_epoch != -1,
        'initial_generalization_epoch': initial_gen_epoch,
        'status': 'success'
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze a directory of CSV logs for grokking events.")
    parser.add_argument("log_dir", type=str, help="Path to the log directory containing run CSV files.")
    parser.add_argument("dataset_summary_file", type=str, help="Path to the dataset_split_summary.csv file.")
    parser.add_argument("--output_file", type=str, default="grokking_analysis_summary.csv", help="Name for the output summary CSV file.")
    args = parser.parse_args()
    
    try:
        run_paths = [os.path.join(args.log_dir, f) for f in sorted(os.listdir(args.log_dir)) 
                     if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"Error: Log directory not found at '{args.log_dir}'")
        return

    print(f"Found {len(run_paths)} potential runs in '{args.log_dir}'. Analyzing in parallel...")
    
    stats = {'total_runs': len(run_paths), 'failed_reads': 0, 'too_short': 0, 'analyzed_runs': 0, 'grokking_runs': 0}
    successful_results = []
    
    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(analyze_log_worker, run_paths), total=len(run_paths), desc="Analyzing Logs"))
        
    for result in results:
        if result is None: stats['failed_reads'] += 1; continue
        status = result.get('status', 'failed_read')
        if status == 'failed_read': stats['failed_reads'] += 1
        elif status == 'too_short': stats['too_short'] += 1
        elif status == 'success':
            stats['analyzed_runs'] += 1
            successful_results.append(result)
            if result.get('grokking_detected', False): stats['grokking_runs'] += 1
            
    if successful_results:
        summary_df = pd.DataFrame(successful_results)
        
        try:
            dataset_summary_df = pd.read_csv(args.dataset_summary_file)
            summary_df['uniprot_id'] = summary_df['run_name'].apply(lambda x: x.split('-')[0])
            summary_df = pd.merge(summary_df, dataset_summary_df[['uniprot_id', 'num_scaffolds']], on='uniprot_id', how='left')
        except FileNotFoundError:
            print(f"Warning: Dataset summary file not found at '{args.dataset_summary_file}'. Scaffold counts will not be included.")
        except Exception as e:
            print(f"An error occurred while merging with the dataset summary: {e}")

        summary_df.to_csv(args.output_file, index=False)
        print(f"\nAnalysis complete. Summary saved to: {args.output_file}")
        
    else:
        print("\nNo valid runs with sufficient data were found to analyze.")
        
    print("\n--- Overall Analysis Statistics ---")
    print(f"Total runs found:              {stats['total_runs']}")
    print(f"  - Failed/Corrupt logs:       {stats['failed_reads']}")
    print(f"  - Logs too short for analysis: {stats['too_short']}")
    print(f"  - Successfully analyzed:     {stats['analyzed_runs']}")
    print(f"  - Grokking events detected:  {stats['grokking_runs']}")
    print("-----------------------------------")

if __name__ == "__main__":
    main()


