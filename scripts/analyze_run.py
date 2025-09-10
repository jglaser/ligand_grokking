import argparse
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count

def find_grokking_point(epochs, train_acc, val_acc,
                        overfit_threshold=0.9, plateau_threshold=0.75,
                        jump_threshold=0.05, window_size=50,
                        min_delay_epochs=500): # New parameter to enforce a minimum delay
    """
    Analyzes a training trajectory for a true "delayed generalization" event.
    Returns a tuple: (memorization_epoch, grokking_epoch).
    """
    if len(val_acc) < window_size * 2:
        return -1, -1

    val_acc_smooth = pd.Series(val_acc).rolling(window=window_size, min_periods=1, center=True).mean().values
    
    # 1. Find the point of sustained memorization
    sustained_high_train = (pd.Series(train_acc).rolling(window=window_size).mean() > overfit_threshold)
    first_high_train_idx = sustained_high_train.idxmax()
    if not sustained_high_train[first_high_train_idx]:
        return -1, -1 # Never memorized
    memorization_epoch = epochs[first_high_train_idx]

    # 2. Search for a grokking jump AFTER the memorization point
    for i in range(first_high_train_idx + 1, len(epochs) - window_size):
        # --- THE CRITICAL NEW CONDITION ---
        # Ensure there's a sustained period of low validation accuracy (the delay)
        # between memorization and the potential jump point.
        is_in_plateau = val_acc_smooth[i] < plateau_threshold
        has_sufficient_delay = (epochs[i] - memorization_epoch) > min_delay_epochs

        if is_in_plateau and has_sufficient_delay:
            jump_size = val_acc_smooth[i + window_size] - val_acc_smooth[i]
            if jump_size > jump_threshold:
                grokking_epoch = epochs[i]
                return memorization_epoch, grokking_epoch
                
    return memorization_epoch, -1 # Memorized but never grokked

# ... (The rest of the script is the same, as its logic is now correct)
def read_tensorboard_log(log_dir: str) -> pd.DataFrame:
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
    run_name = os.path.basename(run_dir_path)
    try:
        history = read_tensorboard_log(run_dir_path)
        if history is None: return {'run_name': run_name, 'status': 'failed_read'}
        if history.empty or len(history) < 100: return {'run_name': run_name, 'status': 'too_short'}
        mem_epoch, grok_epoch = find_grokking_point(history["epoch"].values, history["train_accuracy"].values, history["validation_accuracy"].values)
        delay = grok_epoch - mem_epoch if grok_epoch != -1 and mem_epoch != -1 else -1
        return {'run_name': run_name, 'memorization_epoch': mem_epoch, 'grokking_epoch': grok_epoch,
                'grokking_delay': delay, 'grokking_detected': grok_epoch != -1, 'status': 'success'}
    except Exception:
        return {'run_name': run_name, 'status': 'failed_read'}

def main():
    parser = argparse.ArgumentParser(description="Analyze a directory of TensorBoard logs for grokking events.")
    # ... (parser args are the same)
    parser.add_argument("log_dir", type=str, help="Path to the local TensorBoard log directory ('logs/').")
    parser.add_argument("--output_file", type=str, default="grokking_analysis_summary.csv", help="Name for the output summary CSV file.")
    args = parser.parse_args()
    
    run_dirs = [os.path.join(args.log_dir, d) for d in sorted(os.listdir(args.log_dir)) if os.path.isdir(os.path.join(args.log_dir, d))]
    print(f"Found {len(run_dirs)} potential runs in '{args.log_dir}'. Analyzing in parallel...")
    stats = {'total_runs': len(run_dirs), 'failed_reads': 0, 'too_short': 0, 'analyzed_runs': 0, 'grokking_runs': 0}
    successful_results = []
    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(analyze_log_worker, run_dirs), total=len(run_dirs), desc="Analyzing Logs"))
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


