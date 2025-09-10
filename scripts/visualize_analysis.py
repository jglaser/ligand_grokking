import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    """
    Creates a publication-quality scatter plot to visualize the relationship
    between pocket properties and grokking behavior.
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
    required_cols = {args.x_axis, args.y_axis, args.color_by, 'pdb_id'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"Error: One or more specified columns are missing from the input file: {list(missing)}")
        return
        
    # --- 3. Handle Missing Data Gracefully ---
    # First, drop any rows that are missing the core coordinates for the plot.
    df.dropna(subset=[args.x_axis, args.y_axis], inplace=True)
    
    # Then, specifically handle NaNs in the coloring column. This is crucial for
    # including targets that never grokked (where average_grokking_delay is NaN).
    if args.color_by in df.columns:
        df[args.color_by].fillna(0, inplace=True)
    
    print(f"Plotting {len(df)} data points after cleaning.")

    # --- 4. Create the Plot using Seaborn/Matplotlib ---
    print(f"Generating plot: '{args.x_axis}' vs '{args.y_axis}', colored by '{args.color_by}'...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))

    scatter_plot = sns.scatterplot(
        data=df,
        x=args.x_axis,
        y=args.y_axis,
        hue=args.color_by,
        palette="viridis",
        size=args.color_by, # Optionally size points by the same metric
        sizes=(20, 200),
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5
    )
    
    # --- 5. Customize and Save ---
    plt.title("Grokking Behavior vs. Pocket Physicochemical Properties", fontsize=16)
    
    # Create more descriptive labels
    x_label = args.x_axis.replace('_', ' ').title()
    if "sasa" in args.x_axis: x_label += " (nm²)"
        
    y_label = args.y_axis.replace('_', ' ').title()
    
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    # Update legend title
    legend = scatter_plot.get_legend()
    legend.set_title(args.color_by.replace('_', ' ').title())

    try:
        plt.savefig(args.output_file, format='pdf', bbox_inches='tight')
        print(f"\nSuccess! Plot saved to: {args.output_file}")
    except Exception as e:
        print(f"\nError saving the plot: {e}")


if __name__ == "__main__":
    main()


