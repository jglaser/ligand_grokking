import argparse
import polars as pl
from rdkit import Chem
from rdkit import RDLogger
from loguru import logger
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import shutil # <-- ADDED for directory cleanup
from pathlib import Path # <-- ADDED for path handling

# Suppress RDKit warnings, as we handle errors
RDLogger.DisableLog("rdApp.*")


def canonicalize_smiles(smiles: str) -> str:
    """
    Generates canonical SMILES for a given SMILES string.
    Returns an empty string if the input is invalid.
    """
    if not smiles or not isinstance(smiles, str):
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else ""
    except Exception:
        return ""


def get_scaffold(smiles: str) -> str:
    """
    Calculates the Murcko scaffold for a SMILES string.
    Returns an empty string if there is no scaffold or the input is invalid.
    """
    if not smiles or not isinstance(smiles, str):
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            # MurckoScaffold can return an empty molecule for non-ring systems
            return Chem.MolToSmiles(scaffold) if scaffold.GetNumAtoms() > 0 else ""
        return ""
    except Exception:
        return ""


def parallel_apply(func, data_list: list, desc: str) -> list:
    """Applies a function in parallel using a multiprocessing Pool."""
    num_processes = cpu_count()
    logger.info(f"Starting parallel processing for '{desc}' with {num_processes} processes...")
    with Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(func, data_list, chunksize=1000),
                total=len(data_list),
                desc=desc,
            )
        )
    return results


def main():
    """Convert ligand info from BindingDB TSV to Delta Lake format."""
    parser = argparse.ArgumentParser(
        description="Convert .tsv to delta-lake with parallel preprocessing.",
    )
    parser.add_argument("input_file", type=str, help="Path to the bindingdb file.")
    parser.add_argument("lake_output_dir", type=Path, help="Delta-Lake Output directory") # <-- Changed to Path
    args = parser.parse_args()

    # --- FIX: Clean up existing Delta Lake directory ---
    if args.lake_output_dir.exists():
        logger.warning(f"Output directory {args.lake_output_dir} already exists. Removing it.")
        shutil.rmtree(args.lake_output_dir)

    inclusion = (
        "Ligand SMILES", "UniProt (SwissProt) Primary ID of Target Chain 1",
        "BindingDB Target Chain Sequence 1", "Target Name", "BindingDB Ligand Name",
    )
    assay_cols = (
        "Ki (nM)", "IC50 (nM)", "Kd (nM)", "EC50 (nM)", "kon (M-1-s-1)", "koff (s-1)"
    )

    logger.info(f"Reading {args.input_file} with Polars...")
    df = pl.read_csv(
        args.input_file, separator="\t", has_header=True, quote_char=None,
        ignore_errors=True, columns=list(inclusion + assay_cols),
    )
    df = df.rename({
        "UniProt (SwissProt) Primary ID of Target Chain 1": "uniprot_id",
        "BindingDB Target Chain Sequence 1": "protein_sequence"
    })
    df = df.with_columns(
        [pl.col(assay_cols).str.strip_chars(" <>.").str.replace_all(",", "").cast(pl.Float64, strict=False)]
    )

    # --- Parallel Canonicalization & Scaffolding ---
    smiles_to_process = df["Ligand SMILES"].to_list()
    canonical_smiles = parallel_apply(canonicalize_smiles, smiles_to_process, "Canonicalizing SMILES")
    df = df.with_columns(pl.Series("canonical_smiles", canonical_smiles))

    scaffolds = parallel_apply(get_scaffold, df["canonical_smiles"].to_list(), "Calculating Scaffolds")
    df = df.with_columns(pl.Series("scaffold", scaffolds))

    # --- Filtering ---
    n_before = len(df)
    df = df.filter(
        (pl.col("canonical_smiles") != "") &
        pl.col("protein_sequence").is_not_null()
    )
    logger.info(f"Removed {n_before - len(df)} rows with invalid SMILES or missing protein sequence.")

    if df.is_empty():
        logger.error("The final dataframe is empty after processing. Aborting.")
        return

    logger.info("Writing to Delta Lake...")
    df.write_delta(str(args.lake_output_dir), mode="overwrite") # <-- write_delta expects a string path
    logger.info(f"✅ Successfully created Delta Lake at {args.lake_output_dir}")


if __name__ == "__main__":
    main()
