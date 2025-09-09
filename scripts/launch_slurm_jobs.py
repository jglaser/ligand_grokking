import argparse
import os
import stat
import hashlib
import time

def generate_slurm_script(job_name: str, account: str, time: str, nodes: int, datasets_dir: str,
                        epochs: int, encoder_dim: int, n_inducing_points: int, task_file: str,
                        wandb_project: str, learning_rate: float):
    """
    Generates the content of the SLURM submission script for a large MPI job.
    """
    num_cores = nodes * 32
    # Convert to absolute path to be safe on compute nodes
    abs_datasets_dir = os.path.abspath(datasets_dir)
    slurm_script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH -p batch
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time={time}
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.err

# --- Environment Setup ---

# this is valid for my andes environment, but you'll likely have to modify it
source /lustre/orion/world-shared/stf006/glaser/miniconda3/etc/profile.d/conda.sh
conda activate jax

export TRANSFORMERS_OFFLINE=1
export HF_HOME=/lustre/orion/stf006/scratch/$USER
export WANDB_CACHE_DIR=/tmp/wandb_cache
export HTTP_PROXY=http://proxy.ccs.ornl.gov:3128
export HTTPS_PROXY=http://proxy.ccs.ornl.gov:3128

echo "Job starting on $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Allocated {nodes} nodes for a total of {num_cores} cores."

# --- Launch MPI Runner ---
# srun will start the mpi_runner.py script on every allocated core.
# We now pass all hyperparameters and the unique task list file to the MPI runner.
srun python mpi_runner.py \\
    --datasets_dir {abs_datasets_dir} \\
    --task_list {task_file} \\
    --epochs {epochs} \\
    --encoder_dim {encoder_dim} \\
    --n_inducing_points {n_inducing_points} \\
    --wandb_project {wandb_project} \\
    --learning_rate {learning_rate}

echo "Job finished with exit code $?"
"""
    return slurm_script_content

def main():
    """Main function to generate SLURM and task files for the campaign."""
    parser = argparse.ArgumentParser(description="Generate SLURM scripts for a large, multi-GPU MPI campaign.")
    parser.add_argument("datasets_dir", type=str, help="Path to the directory containing the prepared dataset subdirectories.")
    parser.add_argument("--seeds_per_pdb", type=int, default=5, help="Number of different RNG seeds to run for each PDB ID.")
    parser.add_argument("--job_name", type=str, default="grokking_chem_mpi", help="Name for the SLURM job.")
    parser.add_argument("--account", type=str, required=True, help="Your SLURM account/allocation name.")
    parser.add_argument("--time", type=str, default="02:00:00", help="Walltime for the job (HH:MM:SS).")
    parser.add_argument("--nodes", type=int, default=10, help="Number of nodes to request.")

    # --- Exposed Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=200000, help="Number of training epochs.")
    parser.add_argument("--encoder_dim", type=int, default=16, help="Dimensionality of the learned embedding.")
    parser.add_argument("--n_inducing_points", type=int, default=100, help="Number of inducing points for the SVGP.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--wandb_project", type=str, default="grok_pdbbind", help="Name of the Weights & Biases project.")


    args = parser.parse_args()

    # --- 1. Find PDB IDs ---
    try:
        pdb_ids = sorted([d for d in os.listdir(args.datasets_dir) if os.path.isdir(os.path.join(args.datasets_dir, d))])
        if not pdb_ids:
            print(f"Error: No dataset subdirectories found in '{args.datasets_dir}'")
            return
    except FileNotFoundError:
        print(f"Error: Datasets directory not found at '{args.datasets_dir}'")
        return

    # --- 2. Create Task List with a Unique Filename ---
    tasks = []
    for pdb_id in pdb_ids:
        for seed in range(args.seeds_per_pdb):
            tasks.append(f"{pdb_id},{seed}")
    
    # Generate a unique hash for the task list
    campaign_hash = hashlib.sha1(str(tasks).encode() + str(time.time()).encode()).hexdigest()[:8]
    task_file = f"task_list_{campaign_hash}.txt"
    
    with open(task_file, 'w') as f:
        f.write("\n".join(tasks))
    
    print(f"Generated task list for {len(pdb_ids)} PDB IDs with {args.seeds_per_pdb} seeds each.")
    print(f"Total tasks to run: {len(tasks)}")
    print(f"Task list saved to: {task_file}")

    # --- 3. Generate SLURM Script ---
    slurm_script_content = generate_slurm_script(
        args.job_name, args.account, args.time, args.nodes, args.datasets_dir,
        args.epochs, args.encoder_dim, args.n_inducing_points, task_file,
        args.wandb_project, args.learning_rate
    )
    slurm_file = "submit.slurm"
    with open(slurm_file, 'w') as f:
        f.write(slurm_script_content)

    os.chmod(slurm_file, stat.S_IRWXU)

    print(f"Generated SLURM submission script: {slurm_file}")
    print("\n--- Next Steps ---")
    print(f"1. IMPORTANT: Edit '{slurm_file}' to set up your correct environment.")
    print(f"2. Submit your job to the queue with: sbatch {slurm_file}")

if __name__ == "__main__":
    main()


