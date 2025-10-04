import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.container import BarContainer

def parse_results(file_path):
    """Parses the results.txt file to extract experiment data, including p-values and hyperparameters."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    new_data = """
    ==== cvs_2/cv_chembl_absolute_vectors_mean.npz_a_12_out_1e-3 ====
    stats --mean 0.6221 --std 0.1339 --k 11
    --- AUC vs. Random Test (H0: Mean AUC = 0.5) ---
    Mean Value: 0.6221
    Standard Deviation: 0.1339
    Number of pairs/folds (k): 11
    --- Results ---
    t-statistic: 3.0243
    Degrees of Freedom: 10
    One-tailed p-value: 0.0064
    ✅ Result is statistically significant (p < 0.05).
    Performance is likely better than random.
    """
    content += new_data
    
    blocks = content.strip().split('==== ')
    data = []
    header_re = re.compile(r"cvs_2/cv_chembl_(.+?)\.npz_a_12_out_(.+)")
    stats_re = re.compile(r"stats --mean ([\d.]+) --std ([\d.]+)")
    p_value_re = re.compile(r"One-tailed p-value: ([\d.e-]+)")

    for block in blocks:
        if not block: continue
        lines = block.strip().split('\n')
        header = lines[0].replace(' ====', '')
        header_match = header_re.search(header)
        if not header_match: continue
        feature_set, hyperparameter = header_match.groups()
        if "Error:" in lines[1]: continue
        
        stats_match = stats_re.search(lines[1])
        p_value_match = p_value_re.search(block)
        
        if stats_match:
            data.append({
                'feature_set': feature_set,
                'mean_auc': float(stats_match.group(1)),
                'std_dev': float(stats_match.group(2)),
                'p_value': float(p_value_match.group(1)) if p_value_match else 1.0,
                'hyperparameter': hyperparameter
            })
    return pd.DataFrame(data).dropna()

def map_to_groups(df):
    """Robustly maps feature_sets to Model Type, Ablation, and Signal Type."""
    def get_model_type(name):
        if 'plm' in name: return 'PLM (Mean Pooling)'
        if '_mean' in name: return 'AG (Mean Pooling)'
        return 'AG (Default)'
    df['Model Type'] = df['feature_set'].apply(get_model_type)

    def get_ablation(name):
        if 'wt' in name and 'delta' in name: return 'wt+delta'
        if 'wt_only' in name or 'wtonly' in name: return 'wt only'
        if 'absolute' in name: return 'mutant only'
        if 'randomized' in name: return 'mutant-randomized'
        return 'delta'
    df['Ablation'] = df['feature_set'].apply(get_ablation)
    
    def get_signal_type(name):
        if 'splice' in name:
            return 'Splice Only'
        return 'Full Signal'
    df['Signal Type'] = df['feature_set'].apply(get_signal_type)
    
    return df

def get_significance_marker(p):
    """Returns a significance marker string based on a p-value."""
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'ns' 

def plot_faceted_results(df):
    """Generates a faceted plot with an integrated, frameless legend."""
    best_idx = df.groupby(['Signal Type', 'Model Type', 'Ablation'])['mean_auc'].idxmax()
    summary_df = df.loc[best_idx].sort_values(by=['Signal Type', 'Ablation', 'Model Type'])
    summary_df = summary_df.reset_index(drop=True)

    print("--- Grouped Performance Summary ---")
    print(summary_df[['Signal Type', 'Model Type', 'Ablation', 'mean_auc', 'std_dev', 'p_value', 'hyperparameter']].to_string(index=False))

    signal_types = summary_df['Signal Type'].unique()
    
    fig, axes = plt.subplots(nrows=len(signal_types), ncols=1, figsize=(16, 9), sharex=True)
    if len(signal_types) == 1: axes = [axes]

    for i, signal_type in enumerate(signal_types):
        ax = axes[i]
        plot_data = summary_df[summary_df['Signal Type'] == signal_type]

        sns.barplot(
            x='Ablation', y='mean_auc', hue='Model Type', data=plot_data,
            palette={'AG (Default)': '#3b75af', 'AG (Mean Pooling)': '#66a3d9', 'PLM (Mean Pooling)': '#c85151'},
            ax=ax
        )
        
        ax.set_title(f'Analysis Group: {signal_type}', fontsize=16)
        ax.set_ylabel('Best Mean ROC AUC', fontsize=12)
        ax.set_ylim(0.5, 0.9)
        ax.get_legend().remove()

        bar_containers = [c for c in ax.containers if isinstance(c, BarContainer)]
        hue_order = plot_data['Model Type'].unique()
        
        for container_index, container in enumerate(bar_containers):
            model_type_name = hue_order[container_index]
            model_type_df = plot_data[plot_data['Model Type'] == model_type_name].reset_index(drop=True)
            
            for bar_index, bar in enumerate(container.patches):
                if bar_index >= len(model_type_df): continue
                data_row = model_type_df.iloc[bar_index]
                bar_top_y = bar.get_height() + data_row['std_dev']
                
                ax.errorbar(x=bar.get_x() + bar.get_width() / 2, y=bar.get_height(), yerr=data_row['std_dev'], fmt='none', ecolor='black', capsize=3)
                ax.text(bar.get_x() + bar.get_width() / 2., bar_top_y + 0.005, f"{data_row['mean_auc']:.4f}", ha='center', va='bottom', fontsize=9)
                marker = get_significance_marker(data_row['p_value'])
                ax.text(bar.get_x() + bar.get_width() / 2., bar_top_y + 0.02, marker, ha='center', va='bottom', fontsize=12, weight='bold')

    axes[-1].set_xlabel('Feature-Level Ablation', fontsize=12)
    
    # --- MODIFICATION: Create an integrated legend on the top subplot ---
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, 
                   loc='upper right', 
                   frameon=False, # Remove the frame
                   ncol=len(summary_df['Model Type'].unique()))
    
    fig.suptitle('Model Performance Comparison by Signal Type', fontsize=20, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('faceted_model_performance_final.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    raw_df = parse_results('results.txt')
    grouped_df = map_to_groups(raw_df)
    plot_faceted_results(grouped_df)
