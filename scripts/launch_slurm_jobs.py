import argparse
import os
import stat
import hashlib
import time

def generate_slurm_script(job_name, account, time, nodes, datasets_dir, epochs,
                        encoder_dim, n_inducing_points, task_file, wandb_project,
                        learning_rate, partition, logger, log_dir):
    num_gpus = nodes * 8
    abs_datasets_dir = os.path.abspath(datasets_dir)
    slurm_script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time={time}
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.err

# --- Environment Setup (MODIFY THIS SECTION) ---
# ... (your environment setup here)

echo "Job starting on $(hostname)"

# --- Launch MPI Runner ---
srun --cpu-bind=none python mpi_runner.py \\
    --datasets_dir {abs_datasets_dir} \\
    --task_list {task_file} \\
    --epochs {epochs} \\
    --encoder_dim {encoder_dim} \\
    --n_inducing_points {n_inducing_points} \\
    --wandb_project {wandb_project} \\
    --learning_rate {learning_rate} \\
    --logger {logger} \\
    --log_dir {log_dir}

echo "Job finished with exit code $?"
"""
    return slurm_script_content

def main():
    parser = argparse.ArgumentParser(description="Generate SLURM scripts for a large, multi-GPU MPI campaign.")
    # ... (existing args)
    parser.add_argument("datasets_dir", type=str)
    parser.add_argument("--seeds_per_pdb", type=int, default=5)
    parser.add_argument("--job_name", type=str, default="grokking_chem_mpi")
    parser.add_argument("--account", type=str, required=True)
    parser.add_argument("--time", type=str, default="02:00:00")
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--partition", type=str, default="debug")
    parser.add_argument("--epochs", type=int, default=200000)
    parser.add_argument("--encoder_dim", type=int, default=16)
    parser.add_argument("--n_inducing_points", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--wandb_project", type=str, default="grok_pdbbind")
    # --- New Logging Arguments ---
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"])
    parser.add_argument("--log_dir", type=str, default="logs", help="Base directory for local TensorBoard logs.")

    args = parser.parse_args()

    # --- Create task list with unique filename ---
    pdb_ids = sorted([d for d in os.listdir(args.datasets_dir) if os.path.isdir(os.path.join(args.datasets_dir, d))])
    tasks = [f"{pdb_id},{seed}" for pdb_id in pdb_ids for seed in range(args.seeds_per_pdb)]
    campaign_hash = hashlib.sha1(str(tasks).encode() + str(time.time()).encode()).hexdigest()[:8]
    task_file = f"task_list_{campaign_hash}.txt"
    with open(task_file, 'w') as f:
        f.write("\n".join(tasks))
    print(f"Generated task list '{task_file}' with {len(tasks)} total jobs.")
    
    # --- Generate SLURM Script ---
    slurm_script_content = generate_slurm_script(
        args.job_name, args.account, args.time, args.nodes, args.datasets_dir,
        args.epochs, args.encoder_dim, args.n_inducing_points, task_file,
        args.wandb_project, args.learning_rate, args.partition,
        args.logger, args.log_dir
    )
    slurm_file = "submit.slurm"
    with open(slurm_file, 'w') as f:
        f.write(slurm_script_content)
    os.chmod(slurm_file, stat.S_IRWXU)
    print(f"Generated SLURM submission script: {slurm_file}")
    print("\nNext Step: Edit 'submit.slurm' for your environment, then run 'sbatch submit.slurm'")

if __name__ == "__main__":
    main()


