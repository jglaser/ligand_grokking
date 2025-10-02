import argparse
import os
import polars as pl
import pandas as pd
from tqdm.auto import tqdm
import chembl_downloader

def find_mutation_from_sequences_robust(wt_seq: str, mt_seq: str) -> str | None:
    """
    Compares two protein sequences and returns a standardized mutation string.
    Handles multiple point mutations and single amino acid deletions.
    """
    if not isinstance(wt_seq, str) or not isinstance(mt_seq, str):
        return None

    # Handle Deletions (WT is longer than MT)
    if len(wt_seq) > len(mt_seq):
        if len(wt_seq) == len(mt_seq) + 1:
            for i in range(len(wt_seq)):
                if i == len(mt_seq) or wt_seq[i] != mt_seq[i]:
                    return f"{wt_seq[i]}{i + 1}del"
        return None

    # Handle Substitutions (lengths are equal)
    if len(wt_seq) == len(mt_seq):
        diffs = [(i, wt, mt) for i, (wt, mt) in enumerate(zip(wt_seq, mt_seq)) if wt != mt]
        if diffs:
            mutations = [f"{wt_aa}{pos + 1}{mt_aa}" for pos, wt_aa, mt_aa in diffs]
            return ",".join(sorted(mutations))

    return None

def main(args):
    """
    Builds a drug resistance dataset from ChEMBL by programmatically deriving
    mutations from sequence comparison using the correct schema joins.
    """
    print("--- 1. Loading Raw ChEMBL Tables into Polars ---")
    os.environ['PYSTOW_HOME'] = args.pystow_dir
    
    queries = {
        "target_relations": "SELECT tid, related_tid FROM target_relations WHERE relationship = 'SUBSET OF'",
        "activities": "SELECT assay_id, molregno, standard_units, standard_value FROM activities WHERE standard_units = 'nM' AND standard_value IS NOT NULL",
        "assays": "SELECT assay_id, tid, variant_id FROM assays",
        "compound_structures": "SELECT molregno, standard_inchi_key, canonical_smiles FROM compound_structures",
        "target_components": "SELECT tid, component_id FROM target_components",
        "component_sequences": "SELECT component_id, sequence AS wt_sequence, accession AS uniprot_id FROM component_sequences",
        "variant_sequences": "SELECT variant_id, sequence AS mt_sequence FROM variant_sequences"
    }
    
    dfs = {}
    for name, sql in queries.items():
        print(f"Loading table: {name}...")
        dfs[name] = pl.from_pandas(chembl_downloader.query(sql))

    print("\n--- 2. Deriving Mutations from Sequence Comparison ---")

    mutant_wt_pairs = dfs["target_relations"].rename({"tid": "mutant_tid", "related_tid": "wt_tid"})

    dfs["assays"] = dfs["assays"].with_columns(pl.col("variant_id").cast(pl.Int64))
    dfs["variant_sequences"] = dfs["variant_sequences"].with_columns(pl.col("variant_id").cast(pl.Int64))
    
    wt_data = mutant_wt_pairs.join(dfs["target_components"], left_on="wt_tid", right_on="tid") \
                               .join(dfs["component_sequences"], on="component_id") \
                               .select(["mutant_tid", "wt_tid", "wt_sequence", "uniprot_id"])

    mt_data = mutant_wt_pairs.join(dfs["assays"], left_on="mutant_tid", right_on="tid") \
                               .filter(pl.col("variant_id").is_not_null()) \
                               .join(dfs["variant_sequences"], on="variant_id") \
                               .select(["mutant_tid", "mt_sequence"]).unique()

    sequences = wt_data.join(mt_data, on="mutant_tid")
    
    sequences_pd = sequences.to_pandas()
    # --- Use the new robust function ---
    sequences_pd["mutation"] = sequences_pd.apply(
        lambda row: find_mutation_from_sequences_robust(row["wt_sequence"], row["mt_sequence"]),
        axis=1
    )
    sequences_pd.dropna(subset=['mutation'], inplace=True)
    derived_mutations = pl.from_pandas(sequences_pd[["mutant_tid", "wt_tid", "uniprot_id", "mutation"]])
    
    print(f"Successfully derived {len(derived_mutations)} unique mutations from sequence comparison.")

    # --- 3. Join with Activity Data ---
    activities_with_assays = dfs["activities"].join(dfs["assays"], on="assay_id")

    wt_activities = derived_mutations.join(activities_with_assays, left_on="wt_tid", right_on="tid") \
                                     .join(dfs["compound_structures"], on="molregno") \
                                     .rename({"standard_value": "wt_affinity_nm", "standard_inchi_key": "inchi_key"}) \
                                     .select(["mutant_tid", "inchi_key", "wt_affinity_nm"])

    mt_activities = derived_mutations.join(activities_with_assays, left_on="mutant_tid", right_on="tid") \
                                     .join(dfs["compound_structures"], on="molregno") \
                                     .rename({"standard_value": "mutant_affinity_nm", "standard_inchi_key": "inchi_key"}) \
                                     .select(["mutant_tid", "uniprot_id", "mutation", "canonical_smiles", "inchi_key", "mutant_affinity_nm"])

    print("Joining WT and MT DataFrames...")
    merged_df = mt_activities.join(wt_activities, on=["mutant_tid", "inchi_key"])
    
    print(f"Found {len(merged_df)} paired mutant-wild-type measurements.")

    if merged_df.is_empty():
        print("\n❌ Found no paired measurements.")
        return

    # --- 4. Process the Joined Data ---
    df = merged_df.to_pandas()
    df['fold_change'] = pd.to_numeric(df['mutant_affinity_nm']) / pd.to_numeric(df['wt_affinity_nm'])
    df['confers_resistance'] = (df['fold_change'] > args.resistance_threshold).astype(int)

    final_cols = ['uniprot_id', 'mutation', 'canonical_smiles', 'confers_resistance']
    final_df = df.groupby(['uniprot_id', 'mutation', 'canonical_smiles']).agg(
        confers_resistance=('confers_resistance', lambda x: x.mode()[0] if not x.empty else 0)
    ).reset_index().rename(columns={"canonical_smiles": "Ligand SMILES"})

    print(f"\n--- 5. Final Dataset ---")
    print(f"Generated a final dataset with {len(final_df)} unique entries.")
    print(f"Class distribution:\n{final_df['confers_resistance'].value_counts(normalize=True)}")

    final_df.to_parquet(args.output_path, index=False)
    print(f"\n✅ Done. Final dataset saved to: {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build a drug resistance dataset from ChEMBL by deriving mutations from sequence comparison.")
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the final Parquet dataset.')
    parser.add_argument('--resistance_threshold', type=float, default=2.0, help='Fold-change in affinity to define resistance.')
    parser.add_argument('--pystow_dir', type=str, default='.', help='Directory to store the downloaded ChEMBL database.')
    args = parser.parse_args()
    main(args)
