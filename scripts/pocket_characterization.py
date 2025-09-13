import argparse
import pandas as pd
import numpy as np
import mdtraj as md
from Bio.PDB import PDBList, PDBParser
import os
import warnings
from tqdm.auto import tqdm
import concurrent.futures

# Suppress PDBConstructionWarning from BioPython
warnings.filterwarnings("ignore", category=UserWarning)

# --- Amino Acid Physicochemical Data ---

# (Constants like RESIDUE_VOLUMES, HBOND_DONORS, etc., remain unchanged)
# Average residue volumes in Angstrom^3 (from Zamyatnin, 1972)
RESIDUE_VOLUMES = {
    'ALA': 88.6, 'ARG': 173.4, 'ASN': 114.1, 'ASP': 111.1,
    'CYS': 108.5, 'GLN': 143.8, 'GLU': 138.4, 'GLY': 60.1,
    'HIS': 153.2, 'ILE': 166.7, 'LEU': 166.7, 'LYS': 168.6,
    'MET': 162.9, 'PHE': 189.9, 'PRO': 112.7, 'SER': 89.0,
    'THR': 116.1, 'TRP': 227.8, 'TYR': 193.6, 'VAL': 140.0
}

# H-bond donors/acceptors per residue side chain
HBOND_DONORS = {'ARG': 5, 'LYS': 3, 'TRP': 1, 'ASN': 2, 'GLN': 2, 'HIS': 2, 'SER': 1, 'THR': 1, 'TYR': 1}
HBOND_ACCEPTORS = {'ASP': 2, 'GLU': 2, 'ASN': 2, 'GLN': 2, 'HIS': 2, 'SER': 1, 'THR': 1, 'TYR': 1}

# Charge at neutral pH
RESIDUE_CHARGES = {'ARG': 1, 'LYS': 1, 'HIS': 0.5, 'ASP': -1, 'GLU': -1} # HIS is ~50% protonated

# Hydrophobicity
HYDROPHOBIC_RESIDUES = {'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO'}


def find_ligand(topology):
    """
    Automatically identifies the primary ligand in a PDB topology.
    Heuristic: The largest non-protein, non-water, non-ion molecule.
    """
    non_protein_residues = [res for res in topology.residues if not res.is_protein and not res.is_water]

    if not non_protein_residues:
        return None, None

    # Find the non-protein residue with the most heavy atoms
    ligand_residue = max(non_protein_residues, key=lambda res: sum(1 for atom in res.atoms if atom.element.symbol != 'H'))

    ligand_indices = [atom.index for atom in ligand_residue.atoms]
    return ligand_residue, ligand_indices

def characterize_pocket(pdb_id: str, uniprot_id: str, cache_dir: str = "pdb_cache"):
    """
    Uses a local PDB cache to load a structure and characterizes the binding pocket
    around the automatically detected ligand.

    Args:
        pdb_id: The 4-character PDB ID.
        uniprot_id: The UniProt ID for the target protein.
        cache_dir: The directory to use for storing and retrieving PDB files.

    Returns:
        A dictionary of calculated pocket properties, or None if an error occurs.
    """
    pdb_path = os.path.join(cache_dir, f"pdb{pdb_id.lower()}.ent")
    pdb_path_alt = os.path.join(cache_dir, f"{pdb_id.lower()}.pdb") # Alternative name

    try:
        # --- 1. Use cached PDB or download if not present ---
        if os.path.exists(pdb_path):
            pass # Use pdb_path
        elif os.path.exists(pdb_path_alt):
            pdb_path = pdb_path_alt # Use alternative path
        else:
            pdbl = PDBList()
            pdb_path = pdbl.retrieve_pdb_file(pdb_id, pdir=cache_dir, file_format='pdb', overwrite=True)

        if not os.path.exists(pdb_path):
            return None

        # --- 2. Extract Protein Name from Header ---
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, pdb_path)
        protein_name = structure.header.get('name', 'Unknown').strip().lower()

        # --- 3. Load with MDTraj and find ligand ---
        traj = md.load_pdb(pdb_path)
        protein_selection = traj.topology.select('protein')
        ligand_res, ligand_selection = find_ligand(traj.topology)

        if ligand_selection is None:
            return None

        # --- 4. Identify pocket residues (Robustly) ---
        neighbor_indices = md.compute_neighbors(traj, 0.5, ligand_selection, haystack_indices=protein_selection)

        if not neighbor_indices or len(neighbor_indices[0]) == 0:
            pocket_residues_indices = []
        else:
            pocket_atom_indices = np.unique(neighbor_indices[0])
            pocket_residues_indices = sorted(list(set(traj.topology.atom(i).residue.index for i in pocket_atom_indices)))

        pocket_res_objects = [traj.topology.residue(i) for i in pocket_residues_indices]
        pocket_atom_indices = [atom.index for res in pocket_res_objects for atom in res.atoms]

        # --- 5. Calculate properties ---
        if not pocket_res_objects:
            # Return name even if pocket is empty
            return {"protein_name": protein_name, "uniprot_id": uniprot_id, "num_residues": 0, "volume_A3": "0.0",
                    "hydrophobic_sasa_nm2": "0.00", "hbond_donors": 0, "hbond_acceptors": 0,
                    "net_charge": "0.0", "polarity_score": "0.00"}

        volume = sum(RESIDUE_VOLUMES.get(res.name, 0) for res in pocket_res_objects)

        sasa = md.shrake_rupley(traj.atom_slice(pocket_atom_indices), mode='residue')[0]
        hydrophobic_sasa = sum(sasa[i] for i, res in enumerate(pocket_res_objects) if res.name in HYDROPHOBIC_RESIDUES)

        donors = sum(HBOND_DONORS.get(res.name, 0) for res in pocket_res_objects)
        acceptors = sum(HBOND_ACCEPTORS.get(res.name, 0) for res in pocket_res_objects)

        charge = sum(RESIDUE_CHARGES.get(res.name, 0) for res in pocket_res_objects)
        n_polar = sum(1 for res in pocket_res_objects if res.name not in HYDROPHOBIC_RESIDUES and res.name != 'GLY')
        polarity_score = n_polar / len(pocket_res_objects) if pocket_res_objects else 0

        return {
            "protein_name": protein_name,
            "uniprot_id": uniprot_id,
            "num_residues": len(pocket_res_objects),
            "volume_A3": f"{volume:.1f}",
            "hydrophobic_sasa_nm2": f"{hydrophobic_sasa.sum():.2f}",
            "hbond_donors": donors,
            "hbond_acceptors": acceptors,
            "net_charge": f"{charge:.1f}",
            "polarity_score": f"{polarity_score:.2f}"
        }

    except Exception as e:
        return None

def main():
    parser = argparse.ArgumentParser(description="Characterize binding pockets in parallel from a PDBbind index file.")
    parser.add_argument("index_file", type=str, help="Path to the PDBbind index file (e.g., 'INDEX_refined_data.2020').")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers. Defaults to number of CPU cores.")

    args = parser.parse_args()

    if not os.path.isfile(args.index_file):
        print(f"Error: Index file not found at '{args.index_file}'")
        return

    # --- Create PDB cache directory ---
    cache_dir = "pdb_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"Created PDB cache directory at: ./{cache_dir}")

    # Read PDB and UniProt codes from the index file
    targets = []
    with open(args.index_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                targets.append((parts[0], parts[2]))

    results = []
    num_workers = args.num_workers or os.cpu_count()
    print(f"Found {len(targets)} targets. Characterizing pockets using up to {num_workers} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Pass the cache_dir to each worker
        future_to_target = {executor.submit(characterize_pocket, pdb_id, uniprot_id, cache_dir): (pdb_id, uniprot_id) for pdb_id, uniprot_id in targets}

        for future in tqdm(concurrent.futures.as_completed(future_to_target), total=len(targets), desc="Processing PDBs"):
            pdb_id, uniprot_id = future_to_target[future]
            try:
                properties = future.result()
                if properties:
                    properties["pdb_id"] = pdb_id
                    results.append(properties)
            except Exception as exc:
                # This catches potential exceptions from the worker threads
                pass # Errors are handled inside characterize_pocket

    # --- Create and save results table ---
    if results:
        df = pd.DataFrame(results)
        # Reorder columns to put name first
        df = df[["pdb_id", "uniprot_id", "protein_name", "num_residues", "volume_A3", "hydrophobic_sasa_nm2",
                 "hbond_donors", "hbond_acceptors", "net_charge", "polarity_score"]]

        output_file = "target_pocket_metadata.csv"
        df.to_csv(output_file, index=False)

        print("\n--- Pocket Characterization Summary ---")
        # Print a sample of the results
        print(df.head().to_markdown(index=False))
        print(f"\nFull results for {len(df)} targets saved to: {output_file}")
    else:
        print("\nNo PDB files were successfully processed.")

if __name__ == "__main__":
    main()
