import argparse
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count

def find_grokking_point_vectorized(epochs, train_acc, val_acc,
                                   overfit_threshold=0.9, plateau_threshold=0.75,
                                   jump_threshold=0.05, window_size=50,
                                   min_delay_epochs=500, sustain_window_multiplier=3):
    """
    Analyzes a training trajectory for a grokking event using vectorized NumPy
    operations for significantly improved performance.
    """
    # --- 1. Initial Checks and Data Smoothing (same as before) ---
    required_len = window_size * (sustain_window_multiplier + 1)
    if len(val_acc) < required_len:
        return -1, -1

    val_acc_smooth = pd.Series(val_acc).rolling(window=window_size, min_periods=1, center=True).mean().values

    # --- 2. Find Memorization Point (same as before) ---
    sustained_high_train = (pd.Series(train_acc).rolling(window=window_size).mean() > overfit_threshold)
    true_indices = np.where(sustained_high_train)[0]
    if len(true_indices) == 0:
        return -1, -1  # Never memorized

    first_high_train_idx = true_indices[0]
    memorization_epoch = epochs[first_high_train_idx]

    # --- 3. Vectorized Search for Grokking ---

    # Define the range of indices 'i' to search over. We must have enough
    # data points *after* each index for the jump and sustain windows.
    search_start_idx = first_high_train_idx + 1
    search_end_idx = len(epochs) - (window_size * sustain_window_multiplier)

    if search_start_idx >= search_end_idx:
        return memorization_epoch, -1 # Not enough data after memorization

    candidate_indices = np.arange(search_start_idx, search_end_idx)

    # Create boolean masks for each condition, applied to all candidates at once.
    # Condition A: Is the validation accuracy in a low plateau?
    in_plateau_mask = val_acc_smooth[candidate_indices] < plateau_threshold

    # Condition B: Has there been a sufficient delay since memorization?
    delay_mask = (epochs[candidate_indices] - memorization_epoch) > min_delay_epochs

    # Condition C: Is there a significant jump in accuracy after `window_size`?
    jump_sizes = val_acc_smooth[candidate_indices + window_size] - val_acc_smooth[candidate_indices]
    jump_mask = jump_sizes > jump_threshold

    # Condition D: Is the new accuracy state *sustained*?
    # We use a cumulative sum trick for a highly efficient rolling average calculation.
    post_jump_start_indices = candidate_indices + window_size
    post_jump_end_indices = candidate_indices + window_size * sustain_window_multiplier
    sustain_window_len = window_size * (sustain_window_multiplier - 1)

    # The cumulative sum array allows calculating the sum over any slice in O(1).
    # Prepending [0] simplifies index handling.
    cumsum = np.concatenate(([0], np.cumsum(val_acc_smooth)))
    post_jump_sums = cumsum[post_jump_end_indices] - cumsum[post_jump_start_indices]

    avg_post_jump_acc = np.zeros_like(post_jump_sums, dtype=float)
    if sustain_window_len > 0:
        avg_post_jump_acc = post_jump_sums / sustain_window_len

    sustain_mask = avg_post_jump_acc > plateau_threshold

    # --- 4. Combine Masks and Find First Event ---

    # Find all indices where every condition is True.
    full_mask = in_plateau_mask & delay_mask & jump_mask & sustain_mask

    # Find the index of the *first* True value in our combined mask.
    valid_grok_indices = np.where(full_mask)[0]

    if valid_grok_indices.size > 0:
        # Get the first valid index from our original `candidate_indices` array
        first_grok_idx = candidate_indices[valid_grok_indices[0]]
        grokking_epoch = epochs[first_grok_idx]
        return memorization_epoch, grokking_epoch
    else:
        # Memorized but no grokking event was found
        return memorization_epoch, -1

def find_grokking_point(epochs, train_acc, val_acc,
                        overfit_threshold=0.9, plateau_threshold=0.75,
                        jump_threshold=0.05, window_size=50,
                        min_delay_epochs=500, sustain_window_multiplier=3):
    """
    Analyzes a training trajectory for a true "delayed generalization" event.
    Requires a sustained period of overfitting before a sustained jump to a
    new, high-accuracy state.
    """
    if len(val_acc) < window_size * (sustain_window_multiplier + 1):
        return -1, -1

    val_acc_smooth = pd.Series(val_acc).rolling(window=window_size, min_periods=1, center=True).mean().values
    
    # 1. Find the point of sustained memorization
    sustained_high_train = (pd.Series(train_acc).rolling(window=window_size).mean() > overfit_threshold)
    true_indices = np.where(sustained_high_train)[0]
    if len(true_indices) == 0:
        return -1, -1 # Never memorized
        
    first_high_train_idx = true_indices[0]
    memorization_epoch = epochs[first_high_train_idx]

    # 2. Search for a grokking jump AFTER the memorization point
    # We stop searching early enough to have a window to check for sustained accuracy
    search_end_idx = len(epochs) - (window_size * sustain_window_multiplier)
    for i in range(first_high_train_idx + 1, search_end_idx):
        is_in_plateau = val_acc_smooth[i] < plateau_threshold
        has_sufficient_delay = (epochs[i] - memorization_epoch) > min_delay_epochs

        if is_in_plateau and has_sufficient_delay:
            jump_size = val_acc_smooth[i + window_size] - val_acc_smooth[i]
            
            if jump_size > jump_threshold:
                # --- NEW: Check for SUSTAINED high accuracy after the jump ---
                post_jump_window_start = i + window_size
                post_jump_window_end = i + window_size * sustain_window_multiplier
                avg_post_jump_acc = np.mean(val_acc_smooth[post_jump_window_start:post_jump_window_end])
                
                # The new state must be sustainably above the old plateau
                if avg_post_jump_acc > plateau_threshold:
                    grokking_epoch = epochs[i]
                    return memorization_epoch, grokking_epoch
                
    return memorization_epoch, -1 # Memorized but never grokked

def read_tensorboard_log(log_dir: str) -> pd.DataFrame:
    # ... (function is the same as before)
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
    # ... (function is the same as before)
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
    # ... (parser args are the same as before)
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


