import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

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

def main():
    """
    Selects representative examples from a meta-analysis and plots their
    learning curves for publication.
    """
    parser = argparse.ArgumentParser(
        description="Generate a figure of representative learning curves from a grokking campaign.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("meta_analysis_file", type=str, help="Path to the grokking_meta_analysis.csv file.")
    parser.add_argument("analysis_summary_file", type=str, help="Path to the grokking_analysis_summary.csv file (to find run names).")
    parser.add_argument("logs_dir", type=str, help="Path to the parent directory containing all TensorBoard log folders.")
    parser.add_argument("--output_file", type=str, default="learning_curves_figure.pdf", help="Name for the output PDF figure.")
    parser.add_argument("--num_examples", type=int, default=4, help="Number of representative examples to plot.")
    parser.add_argument("--min_epochs", type=int, default=50000, help="Minimum number of epochs to show on the x-axis.")

    args = parser.parse_args()

    # --- 1. Load Analysis Results ---
    try:
        meta_df = pd.read_csv(args.meta_analysis_file)
        summary_df = pd.read_csv(args.analysis_summary_file)
    except FileNotFoundError as e:
        print(f"Error: Could not find an input file. {e}")
        return

    if 'run_name' in summary_df.columns and 'pdb_id' not in summary_df.columns:
        summary_df['pdb_id'] = summary_df['run_name'].apply(lambda x: x.split('-seed-')[0])

    # --- 2. Select Representative Runs ---
    representatives = {}
    grokking_runs_df = summary_df[summary_df['grokking_detected'] == True].copy()
    
    if grokking_runs_df.empty:
        print("No successful grokking runs found in the summary file. Cannot generate plot.")
        return

    if not grokking_runs_df['grokking_delay'].dropna().empty:
        representatives['Longest Delay'] = grokking_runs_df.loc[grokking_runs_df['grokking_delay'].idxmax()]
        representatives['Shortest Delay'] = grokking_runs_df.loc[grokking_runs_df['grokking_delay'].idxmin()]

    if not meta_df['grokking_frequency'].dropna().empty:
        highest_freq_pdb = meta_df.loc[meta_df['grokking_frequency'].idxmax()]['pdb_id']
        possible_runs = grokking_runs_df[grokking_runs_df['pdb_id'] == highest_freq_pdb]
        if not possible_runs.empty:
            if highest_freq_pdb not in [v['pdb_id'] for v in representatives.values()]:
                 representatives['Highest Frequency'] = possible_runs.iloc[0]

    sorted_by_delay = grokking_runs_df.sort_values('grokking_delay', ascending=False)
    for _, row in sorted_by_delay.iterrows():
        if len(representatives) >= args.num_examples: break
        if row['pdb_id'] not in [v['pdb_id'] for v in representatives.values()]:
            representatives[f"Mid Delay ({int(row['grokking_delay'])})"] = row
            
    print("Selected representative runs for plotting:")
    for key, val in representatives.items():
        print(f"  - {key}: Run Name {val['run_name']}")

    # --- 3. Create a Multi-Panel Plot ---
    num_plots = len(representatives)
    if num_plots == 0:
        print("No representative targets could be selected.")
        return
        
    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 4 * num_plots), sharex=True, sharey=True)
    if num_plots == 1: axes = [axes]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig.suptitle("Representative Learning Dynamics", fontsize=16, y=1.02)

    max_epoch_overall = 0

    for ax, (title, run_info) in zip(axes, representatives.items()):
        run_name = run_info['run_name']
        pdb_id = run_info['pdb_id']
        mem_epoch = run_info['memorization_epoch']
        grok_epoch = run_info['grokking_epoch']
        log_path = os.path.join(args.logs_dir, run_name)
        
        history = read_tensorboard_log(log_path)
        if history is None:
            ax.text(0.5, 0.5, f"Could not load log for\n{run_name}", ha='center', va='center')
            continue
        
        run_max_epoch = history['epoch'].max()
        max_epoch_overall = max(max_epoch_overall, run_max_epoch)

        sns.lineplot(x='epoch', y='train_accuracy', data=history, ax=ax, label='Train Accuracy', color='C0')
        sns.lineplot(x='epoch', y='validation_accuracy', data=history, ax=ax, label='Validation Accuracy', color='C1')
        
        # --- NEW: Visualize Memorization and Grokking Points ---
        if mem_epoch != -1:
            ax.axvline(x=mem_epoch, color='purple', linestyle=':', linewidth=2, alpha=0.8)

        if grok_epoch != -1:
            ax.axvline(x=grok_epoch, color='green', linestyle='--', linewidth=2, alpha=0.8)

        ax.set_title(f"{title}: {pdb_id}", fontsize=12)
        ax.set_ylabel("Accuracy")

        # Create a cleaner legend
        handles, labels = ax.get_legend_handles_labels()
        # Add custom handles for our event lines
        handles.extend([plt.Line2D([0], [0], color='purple', linestyle=':', lw=2),
                        plt.Line2D([0], [0], color='green', linestyle='--', lw=2)])
        labels.extend(['Memorization', 'Grokking'])
        ax.legend(handles=handles, labels=labels)
        
        ax.set_ylim(0.4, 1.05)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # --- 4. Set Consistent X-axis Limit ---
    xlim_upper = max(max_epoch_overall, args.min_epochs)
    axes[-1].set_xlim(left=-0.02 * xlim_upper, right=xlim_upper * 1.05)
    axes[-1].set_xlabel("Epoch", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # --- 5. Save the Figure ---
    try:
        plt.savefig(args.output_file, format='pdf', bbox_inches='tight')
        print(f"\nSuccess! Figure saved to: {args.output_file}")
    except Exception as e:
        print(f"\nError saving the plot: {e}")

if __name__ == "__main__":
    main()


