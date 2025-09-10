import argparse
import os
import stat
import hashlib
import time
import itertools
import math

def generate_slurm_script(job_name: str, account: str, time: str, nodes_per_job: int, partition: str,
                        num_chunks: int, datasets_dir: str, campaign_hash: str):
    """
    Generates the content of a single SLURM script that launches an array of MPI jobs.
    """
    abs_datasets_dir = os.path.abspath(datasets_dir)
    slurm_script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes_per_job}
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time={time}
#SBATCH --array=1-{num_chunks}
#SBATCH -o slurm/%x-%A_%a.out
#SBATCH -e slurm/%x-%A_%a.err

# --- Environment Setup (MODIFY THIS SECTION) ---
# ... (your environment setup here, e.g., conda activate)

echo "Job Array Task $SLURM_ARRAY_TASK_ID starting on $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"

# --- Select the correct task list for this chunk ---
# This explicitly constructs the filename using the unique campaign hash
TASK_FILE="task_list_chunk_${{SLURM_ARRAY_TASK_ID}}_{campaign_hash}.txt"
echo "This MPI job will process tasks from: $TASK_FILE"

# --- Launch the MPI job for this chunk ---
srun python mpi_runner.py \\
    --datasets_dir {abs_datasets_dir} \\
    --task_list "$TASK_FILE"

echo "Job Array Task $SLURM_ARRAY_TASK_ID finished with exit code $?"
"""
    return slurm_script_content

def main():
    """Main function to generate a single SLURM script that creates a job array of MPI jobs."""
    parser = argparse.ArgumentParser(description="Generate a single SLURM job array script for a large-scale MPI campaign.")
    parser.add_argument("datasets_dir", type=str, help="Path to the directory containing prepared dataset subdirectories.")
    parser.add_argument("--job_name", type=str, default="grok_sweep", help="Base name for the SLURM jobs.")
    parser.add_argument("--account", type=str, required=True, help="Your SLURM account/allocation name.")
    parser.add_argument("--time", type=str, default="02:00:00", help="Walltime for each MPI job in the array (HH:MM:SS).")
    parser.add_argument("--nodes_per_job", type=int, default=10, help="Number of nodes to allocate for EACH MPI job in the array.")
    parser.add_argument("--partition", type=str, default="batch", help="SLURM partition/queue.")
    parser.add_argument("--tasks_per_job", type=int, default=5000, help="Maximum number of training runs to process in a single MPI job.")

    # --- Hyperparameter Grid Definition ---
    # These are defined here to create the master task list
    seeds = range(5)
    learning_rates = [1e-4, 1e-5]
    encoder_dims = [8, 16]
    
    args = parser.parse_args()

    # --- 1. Find PDB IDs & Create Full Task List ---
    try:
        pdb_ids = sorted([d for d in os.listdir(args.datasets_dir) if os.path.isdir(os.path.join(args.datasets_dir, d))])
    except FileNotFoundError:
        print(f"Error: Datasets directory not found at '{args.datasets_dir}'")
        return

    tasks = [f"{p},{s},{lr},{ed}" for p, s, lr, ed in itertools.product(pdb_ids, seeds, learning_rates, encoder_dims)]
    print(f"Generated a total of {len(tasks)} unique training tasks.")

    # --- 2. Generate a SINGLE hash for the entire campaign ---
    campaign_hash = hashlib.sha1(str(tasks).encode() + str(time.time()).encode()).hexdigest()[:8]
    print(f"Generated unique campaign hash: {campaign_hash}")

    # --- 3. Chunk Tasks into Smaller Lists ---
    num_chunks = math.ceil(len(tasks) / args.tasks_per_job)
    task_chunks = [tasks[i:i + args.tasks_per_job] for i in range(0, len(tasks), args.tasks_per_job)]
    print(f"Splitting into {num_chunks} job array tasks (chunks), with max {args.tasks_per_job} training runs per chunk.")

    # --- 4. Generate a Task File for Each Chunk using the SAME hash ---
    for i, chunk in enumerate(task_chunks):
        chunk_num = i + 1
        # Use the single campaign hash for all task files
        task_file = f"task_list_chunk_{chunk_num}_{campaign_hash}.txt"
        with open(task_file, 'w') as f:
            f.write("\n".join(chunk))

    # --- 5. Generate a Single SLURM Submission Script for the Array ---
    slurm_script_content = generate_slurm_script(
        args.job_name, args.account, args.time, args.nodes_per_job, args.partition,
        num_chunks, args.datasets_dir, campaign_hash
    )
    
    slurm_file = "submit_array.slurm"
    with open(slurm_file, 'w') as f:
        f.write(slurm_script_content)
    
    os.chmod(slurm_file, stat.S_IRWXU)
    print(f"\nGenerated single SLURM submission script: {slurm_file}")
    
    print("\n--- Next Steps ---")
    print(f"1. IMPORTANT: Edit '{slurm_file}' to set up your correct environment.")
    print(f"2. Submit your job array to the queue with: sbatch {slurm_file}")
    print(f"   This will launch {num_chunks} independent MPI jobs.")

if __name__ == "__main__":
    main()


