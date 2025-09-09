import argparse
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# This script can now analyze logs from either wandb or tensorboard
SUPPORTED_LOGGERS = ["wandb", "tensorboard"]

def find_grokking_point(epochs, train_acc, val_acc,
                        overfit_threshold=0.9, plateau_threshold=0.7,
                        jump_threshold=0.05, window_size=50):
    """
    Analyzes a training trajectory to find a grokking event.
    """
    if len(val_acc) < window_size * 2:
        return -1 # Not enough data to analyze

    val_acc_smooth = pd.Series(val_acc).rolling(window=window_size, min_periods=1, center=True).mean().values

    in_plateau = False
    for i in range(window_size, len(epochs) - window_size):
        if train_acc[i] > overfit_threshold and val_acc_smooth[i] < plateau_threshold:
            in_plateau = True
        if in_plateau:
            jump_size = val_acc_smooth[i + window_size] - val_acc_smooth[i]
            if jump_size > jump_threshold:
                return epochs[i]
    return -1

def read_tensorboard_log(log_dir: str) -> pd.DataFrame:
    """
    Reads scalar data from a TensorBoard log directory into a pandas DataFrame.
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        raise ImportError("TensorBoard is required to analyze local logs. Please install with 'pip install tensorboard'.")

    ea = event_accumulator.EventAccumulator(log_dir,
        size_guidance={event_accumulator.SCALARS: 0} # Load all scalar data
    )
    ea.Reload()

    required_tags = {'train_accuracy', 'validation_accuracy'}
    if not required_tags.issubset(ea.Tags()['scalars']):
        return None # Run is missing required data

    train_events = ea.Scalars('train_accuracy')
    val_events = ea.Scalars('validation_accuracy')

    # Convert to DataFrames and merge
    df_train = pd.DataFrame([(e.step, e.value) for e in train_events], columns=['epoch', 'train_accuracy'])
    df_val = pd.DataFrame([(e.step, e.value) for e in val_events], columns=['epoch', 'validation_accuracy'])
    
    # Merge, forward-fill to align steps, and drop any initial NaNs
    history_df = pd.merge(df_train, df_val, on='epoch', how='outer').sort_values('epoch')
    history_df = history_df.fillna(method='ffill').dropna()
    
    return history_df

def main():
    parser = argparse.ArgumentParser(description="Analyze training runs for grokking events.")
    parser.add_argument("path", type=str, help="Path to the wandb project ('entity/project') or local TensorBoard log directory ('logs/').")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=SUPPORTED_LOGGERS, help="The logging backend to analyze.")
    
    args = parser.parse_args()

    analysis_results = []

    if args.logger == "wandb":
        try:
            import wandb
            api = wandb.Api()
            runs = api.runs(args.path)
            print(f"Found {len(runs)} runs in wandb project '{args.path}'. Analyzing...")
            
            for run in tqdm(runs, desc="Analyzing wandb runs"):
                required_keys = {"epoch", "train_accuracy", "validation_accuracy"}
                if not required_keys.issubset(run.history_keys['keys'].keys()):
                    continue
                
                history = run.history(keys=list(required_keys), pandas=True).dropna()
                grokking_epoch = find_grokking_point(history["epoch"].values, history["train_accuracy"].values, history["validation_accuracy"].values)
                
                if grokking_epoch != -1:
                    # Update wandb run with tag and metadata
                    new_tag = "groks"
                    if new_tag not in run.tags:
                        run.tags.append(new_tag)
                        run.summary["grokking_event"] = f"Grokking event detected around epoch: {grokking_epoch}"
                        run.update()
                        tqdm.write(f"  - Tagged run '{run.name}' as grokking at epoch {grokking_epoch}")

        except Exception as e:
            print(f"An error occurred with wandb: {e}")

    elif args.logger == "tensorboard":
        if not os.path.isdir(args.path):
            print(f"Error: Log directory not found at '{args.path}'")
            return
        
        run_dirs = sorted([d for d in os.listdir(args.path) if os.path.isdir(os.path.join(args.path, d))])
        print(f"Found {len(run_dirs)} potential runs in '{args.path}'. Analyzing...")

        for run_name in tqdm(run_dirs, desc="Analyzing TensorBoard logs"):
            log_dir = os.path.join(args.path, run_name)
            history = read_tensorboard_log(log_dir)

            if history is None or history.empty or len(history) < 100:
                continue

            grokking_epoch = find_grokking_point(history["epoch"].values, history["train_accuracy"].values, history["validation_accuracy"].values)
            analysis_results.append({
                'run_name': run_name,
                'grokking_detected': grokking_epoch != -1,
                'grokking_epoch': grokking_epoch
            })
        
        if analysis_results:
            summary_df = pd.DataFrame(analysis_results)
            summary_file = os.path.join(args.path, "grokking_analysis_summary.csv")
            summary_df.to_csv(summary_file, index=False)
            print("\n--- Analysis Summary ---")
            print(summary_df[summary_df['grokking_detected'] == True].to_markdown(index=False))
            print(f"\nFull summary saved to: {summary_file}")

if __name__ == '__main__':
    main()


