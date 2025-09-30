import argparse
import numpy as np
from scipy.stats import t

def calculate_auc_p_value(mean_auc, std_auc, k):
    """
    Calculates the one-tailed p-value for a given set of ROC AUC stats
    from a k-fold cross-validation to test if performance is better than random (0.5).
    """
    # The null hypothesis is that the true mean AUC is 0.5
    mu0 = 0.5
    
    # 1. Calculate the t-statistic
    sem = std_auc / np.sqrt(k)
    # Avoid division by zero if std_dev is 0
    if sem == 0:
        # If std is 0 and mean is > 0.5, t-statistic is effectively infinite, p-value is 0
        # If mean is <= 0.5, it's not significant.
        t_stat = float('inf') if mean_auc > mu0 else 0
    else:
        t_stat = (mean_auc - mu0) / sem

    # 2. Calculate the degrees of freedom
    df = k - 1

    # 3. Calculate the one-tailed p-value using the survival function (1 - cdf)
    p_value = t.sf(t_stat, df)
    
    return t_stat, df, p_value

def main():
    """Main function to parse arguments and print results."""
    parser = argparse.ArgumentParser(
        description="Calculate the p-value for ROC AUC score from k-fold cross-validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Define command-line arguments
    parser.add_argument("--mean", type=float, required=True, help="Mean ROC AUC score.")
    parser.add_argument("--std", type=float, required=True, help="Standard deviation of ROC AUC scores.")
    parser.add_argument("--k", type=int, required=True, help="Number of folds (k).")
    
    args = parser.parse_args()
    
    # Unpack arguments
    mean_auc = args.mean
    std_auc = args.std
    k = args.k
    
    # Perform calculation
    t_stat, df, p_value = calculate_auc_p_value(mean_auc, std_auc, k)
    
    # --- Print the results ---
    print("\n--- Input Values ---")
    print(f"Mean AUC: {mean_auc}")
    print(f"Standard Deviation: {std_auc}")
    print(f"Number of folds (k): {k}\n")
    
    print("--- Results ---")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"Degrees of Freedom: {df}")
    print(f"One-tailed p-value: {p_value:.4f}\n")

    # --- Interpretation ---
    alpha = 0.05
    if p_value < alpha:
        print(f"✅ Result is statistically significant (p < {alpha}).")
        print("   We can reject the null hypothesis; performance is likely better than random.")
    else:
        print(f"❌ Result is not statistically significant (p >= {alpha}).")
        print("   We cannot reject the null hypothesis; performance may be due to chance.")

if __name__ == "__main__":
    main()
