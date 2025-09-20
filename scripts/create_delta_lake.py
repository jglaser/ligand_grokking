import argparse
import polars as pl
from rdkit import Chem
from rdkit import RDLogger
from loguru import logger
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Suppress RDKit warnings, as we handle errors
RDLogger.DisableLog("rdApp.*")


def canonicalize_smiles(smiles: str) -> str | None:
    """
    Generates canonical SMILES for a given SMILES string.
    Returns None if the input is invalid.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        return None
    except Exception:
        return None


def parallel_canonicalize(smiles_list: list[str]) -> list[str | None]:
    """
    Applies the canonicalize_smiles function in parallel using multiprocessing.
    """
    num_processes = cpu_count()
    logger.info(f"Starting parallel canonicalization with {num_processes} processes...")
    with Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(canonicalize_smiles, smiles_list, chunksize=1000),
                total=len(smiles_list),
                desc="Canonicalizing SMILES",
            )
        )
    return results


def main():
    """Convert ligand info from BindingDB TSV to Delta Lake format,
    including canonical SMILES and protein sequences."""

    parser = argparse.ArgumentParser(
        description="Convert .tsv to delta-lake, adding canonical_smiles and protein_sequence.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input_file", type=str, help="Path to the bindingdb file.")
    parser.add_argument("lake_output_dir", type=str, help="Delta-Lake Output directory")
    args = parser.parse_args()

    # --- Add "BindingDB Target Chain Sequence 1" to the columns to include ---
    inclusion = (
        "Ligand SMILES",
        "UniProt (SwissProt) Primary ID of Target Chain 1",
        "BindingDB Target Chain Sequence 1", # <-- ADDED
        "Target Name",
        "BindingDB Ligand Name",
    )
    assay_cols = (
        "Ki (nM)", "IC50 (nM)", "Kd (nM)", "EC50 (nM)", "kon (M-1-s-1)", "koff (s-1)"
    )

    logger.info(f"Reading {args.input_file} with Polars...")
    # Read the necessary columns directly to save memory
    df = pl.read_csv(
        args.input_file,
        separator="\t",
        has_header=True,
        quote_char=None,
        ignore_errors=True,
        columns=list(inclusion + assay_cols),
    )

    # --- Rename columns for clarity ---
    df = df.rename({
        "UniProt (SwissProt) Primary ID of Target Chain 1": "uniprot_id",
        "BindingDB Target Chain Sequence 1": "protein_sequence" # <-- ADDED
    })

    # --- Clean and cast assay columns ---
    df = df.with_columns(
        [
            pl.col(assay_cols)
            .str.strip_chars(" <>.")
            .str.replace_all(",", "")
            .cast(pl.Float64, strict=False)
        ]
    )

    # --- Parallel Canonicalization Step ---
    smiles_to_process = df["Ligand SMILES"].to_list()
    canonical_smiles_list = parallel_canonicalize(smiles_to_process)
    df = df.with_columns(
        pl.Series("canonical_smiles", canonical_smiles_list)
    )

    # --- Filter out invalid SMILES and sequences ---
    n_before = len(df)
    df = df.filter(
        pl.col("canonical_smiles").is_not_null() &
        pl.col("protein_sequence").is_not_null()
    )
    n_after = len(df)
    logger.info(f"Removed {n_before - n_after} rows with invalid SMILES or missing protein sequences.")

    logger.info("Writing to Delta Lake...")
    df.write_delta(
        args.lake_output_dir,
        mode="overwrite",
        delta_write_options={"schema_mode": "overwrite"},
    )

    logger.info(f"✅ Successfully created Delta Lake at {args.lake_output_dir}")
    logger.info(f"Final dataset has {len(df)} rows.")


if __name__ == "__main__":
    main()
