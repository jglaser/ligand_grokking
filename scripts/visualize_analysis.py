import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    """
    Creates a publication-quality scatter plot to visualize the relationship
    between pocket properties and grokking behavior, separating grokking and
    non-grokking targets for clarity.
    """
    parser = argparse.ArgumentParser(
        description="Generate a scatter plot from the grokking meta-analysis results.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_file", type=str, help="Path to the grokking_meta_analysis.csv file.")
    parser.add_argument("--x_axis", type=str, default="hydrophobic_sasa_nm2",
                        help="Column to use for the X-axis.")
    parser.add_argument("--y_axis", type=str, default="polarity_score",
                        help="Column to use for the Y-axis.")
    parser.add_argument("--color_by", type=str, default="grokking_frequency",
                        help="Column to use for the color scale (e.g., 'grokking_frequency' or 'average_grokking_delay').")
    parser.add_argument("--output_file", type=str, default="grokking_visualization.pdf",
                        help="Name for the output PDF file.")

    args = parser.parse_args()

    # --- 1. Load Data ---
    try:
        df = pd.read_csv(args.input_file)
    except FileNotFoundError:
        print(f"Error: Meta-analysis file not found at '{args.input_file}'")
        return

    # --- 2. Validate Columns ---
    required_cols = {args.x_axis, args.y_axis, args.color_by, 'pdb_id', 'grokking_frequency'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"Error: One or more specified columns are missing from the input file: {list(missing)}")
        return
        
    # --- 3. Handle Missing Data and Partition ---
    df.dropna(subset=[args.x_axis, args.y_axis], inplace=True)
    
    # Partition the data into grokking and non-grokking sets
    grokking_df = df[df['grokking_frequency'] > 0].copy()
    non_grokking_df = df[df['grokking_frequency'] == 0].copy()

    # --- 4. Create the Plot using Seaborn/Matplotlib ---
    print(f"Generating plot: '{args.x_axis}' vs '{args.y_axis}', colored by '{args.color_by}'...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    # --- Conditional Plotting Logic ---
    if args.color_by == 'average_grokking_delay':
        # Only plot the targets that actually grokked when coloring by delay
        print(f"Plotting {len(grokking_df)} targets that grokked (coloring by delay).")
        grokking_df.dropna(subset=[args.color_by], inplace=True)
        
        if not grokking_df.empty:
            sns.scatterplot(
                data=grokking_df,
                x=args.x_axis,
                y=args.y_axis,
                hue=args.color_by,
                palette="viridis_r", # High delay -> Dark color
                size=args.color_by,
                # --- THE FIX ---
                # Ensure high delay corresponds to large size
                sizes=(30, 200),
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
                ax=ax
            )
    else:
        # Plotting frequency or other metrics: show non-grokking as a background
        print(f"Plotting {len(grokking_df)} grokking targets and {len(non_grokking_df)} non-grokking targets (as background).")
        
        # Plot the background points first
        sns.scatterplot(
            data=non_grokking_df,
            x=args.x_axis,
            y=args.y_axis,
            color='none',
            edgecolor='k',
            linewidth=1,
            s=20, # Small size for background points
            alpha=0.5,
            ax=ax,
            legend=False # No legend for background points
        )
        
        # Plot the grokking points on top
        if not grokking_df.empty:
            grokking_df[args.color_by].fillna(0, inplace=True)
            sns.scatterplot(
                data=grokking_df,
                x=args.x_axis,
                y=args.y_axis,
                hue=args.color_by,
                palette="viridis", # High frequency -> Bright color
                size=args.color_by, # High frequency -> Big size
                sizes=(30, 200),
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
                ax=ax
            )
    
    # --- 5. Customize and Save ---
    ax.set_title("Grokking Behavior vs. Pocket Physicochemical Properties", fontsize=16)
    
    x_label = args.x_axis.replace('_', ' ').title()
    if "sasa" in args.x_axis: x_label += " (nm²)"
        
    y_label = args.y_axis.replace('_', ' ').title()
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    
    legend = ax.get_legend()
    if legend:
        legend.set_title(args.color_by.replace('_', ' ').title())

    try:
        plt.savefig(args.output_file, format='pdf', bbox_inches='tight')
        print(f"\nSuccess! Plot saved to: {args.output_file}")
    except Exception as e:
        print(f"\nError saving the plot: {e}")


if __name__ == "__main__":
    main()


