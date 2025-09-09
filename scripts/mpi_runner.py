import os
import subprocess
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
import argparse
from itertools import repeat

def run_training_task(task_tuple):
    """
    This is the worker function that runs a single training job.
    It's executed by a worker process in the MPI pool.
    """
    task_line, datasets_dir, epochs, encoder_dim, n_inducing_points, wandb_project, learning_rate = task_tuple
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # --- Determine Local GPU ---
    local_rank = int(os.environ.get('SLURM_LOCALID', rank % 8))

    # --- Unpack Task and Run ---
    try:
        pdb_id, seed = task_line.split(',')
        
        print(f"Rank {rank} (Local task {local_rank}) starting task: PDB={pdb_id}, Seed={seed}", flush=True)

        train_file = os.path.join(datasets_dir, pdb_id, "train.csv")
        test_file = os.path.join(datasets_dir, pdb_id, "test.csv")
        run_name = f"{pdb_id}-seed-{seed}"

        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print(f"Rank {rank} skipping task {pdb_id}: Dataset files not found in '{os.path.join(datasets_dir, pdb_id)}'.", flush=True)
            return f"Skipped: {pdb_id}"

        command = [
            "python", "../svgp/train_classifier.py",
            train_file,
            test_file,
            "--random_seed", str(seed),
            "--wandb_run_name", run_name,
            "--wandb_project", wandb_project,
            "--learning_rate", str(learning_rate),
            # Pass the exposed hyperparameters
            "--n_inducing_points", str(n_inducing_points),
            "--encoder_dim", str(encoder_dim),
            "--epochs", str(epochs)
        ]

        subprocess.run(command, check=True)

        print(f"Rank {rank} successfully completed task: PDB={pdb_id}, Seed={seed}", flush=True)
        return f"Success: {pdb_id}"

    except Exception as e:
        print(f"Rank {rank} FAILED task: {task_line}. Error: {e}", flush=True)
        return f"Failed: {task_line}"


def main():
    """
    Main function, run by all MPI processes.
    The root process (rank 0) acts as the manager, distributing tasks.
    Other processes act as workers.
    """
    parser = argparse.ArgumentParser(description="MPI runner for training jobs.")
    parser.add_argument("--datasets_dir", type=str, required=True, help="Path to the directory containing dataset subdirectories.")
    parser.add_argument("--task_list", type=str, default="task_list.txt", help="Path to the unique task list file for this campaign.")
    # --- Exposed Hyperparameters with Defaults ---
    parser.add_argument("--epochs", type=int, default=200000, help="Number of training epochs.")
    parser.add_argument("--encoder_dim", type=int, default=16, help="Dimensionality of the learned embedding.")
    parser.add_argument("--n_inducing_points", type=int, default=100, help="Number of inducing points for the SVGP.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--wandb_project", type=str, default="grok_pdbbind", help="Name of the Weights & Biases project.")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    with MPICommExecutor(comm, root=0) as executor:
        if executor is not None:  # True for the root process (rank 0)
            print(f"Manager (Rank 0) started with {comm.Get_size() - 1} workers.")
            
            try:
                with open(args.task_list, 'r') as f:
                    tasks = f.read().strip().split('\n')
            except FileNotFoundError:
                print(f"Error: Task list '{args.task_list}' not found. Please ensure it was generated correctly.")
                tasks = []

            if tasks:
                # Combine tasks with the constant arguments for the worker function
                task_tuples = zip(
                    tasks, 
                    repeat(args.datasets_dir),
                    repeat(args.epochs),
                    repeat(args.encoder_dim),
                    repeat(args.n_inducing_points),
                    repeat(args.wandb_project),
                    repeat(args.learning_rate)
                )
                results = executor.map(run_training_task, task_tuples)
                
                completed_count = 0
                for result in results:
                    completed_count += 1
                
                print(f"Manager (Rank 0): All {completed_count}/{len(tasks)} tasks have been processed.")

    if rank == 0:
        print("Manager (Rank 0): Job complete.")

if __name__ == "__main__":
    main()


