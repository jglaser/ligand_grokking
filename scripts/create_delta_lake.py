import polars as pl
import argparse

def main():
    """ Convert ligand info (BindingDB) to Delta Lake format for fast lookup """

    parser = argparse.ArgumentParser(
        description="Convert .tsv to delta-lake.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_file", type=str, default="BindingDB_All.tsv", help="Path to the bindingdb file.")
    parser.add_argument("lake_output_dir", type=str, default="delta_lake",
                        help="Delta-Lake Output directory")

    args = parser.parse_args()

    inclusion = (
        "PDB ID(s) for Ligand-Target Complex",
        "Target Name",
        "Ligand SMILES",
        "BindingDB Ligand Name",
        "UniProt (SwissProt) Primary ID of Target Chain 1",
    )
    assay_cols = (
        "Ki (nM)",
        "IC50 (nM)",
        "Kd (nM)",
        "EC50 (nM)",
        "kon (M-1-s-1)",
        "koff (s-1)",
    )
    (
        pl.scan_csv(
            args.input_file,
            separator="\t",
            has_header=True,
            quote_char=None,
            ignore_errors=True
        )
        .select(
            inclusion + 
            assay_cols
        )
        .with_columns([
            pl.col(assay_cols)
              .str.strip_chars(" <>.")
              .str.replace_all(",", "")
              .cast(pl.Float64, strict=False)
        ])
        .collect()
        .write_delta(
            args.lake_output_dir,
            mode="overwrite",
            delta_write_options = {"schema_mode": "overwrite"}
        )
    )

if __name__ == "__main__":
    main()
