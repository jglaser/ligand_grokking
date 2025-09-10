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
    # Unpack the full tuple, which now includes non-swept campaign parameters
    task_line, datasets_dir, epochs, n_inducing_points, wandb_project, logger, log_dir = task_tuple
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    local_rank = int(os.environ.get('SLURM_LOCALID', rank % 8))
    
    try:
        # Parse the extended task line containing swept hyperparameters
        pdb_id, seed, learning_rate, encoder_dim = task_line.split(',')
        
        print(f"Rank {rank} (Local rank {local_rank}) starting task: PDB={pdb_id}, Seed={seed}, LR={learning_rate}, EncDim={encoder_dim}", flush=True)
        
        train_file = os.path.join(datasets_dir, pdb_id, "train.csv")
        test_file = os.path.join(datasets_dir, pdb_id, "test.csv")
        
        # Create a more descriptive run name including the hyperparameters
        run_name = f"{pdb_id}-seed{seed}-lr{learning_rate}-ed{encoder_dim}"

        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print(f"Rank {rank} skipping task {pdb_id}: Dataset files not found.", flush=True)
            return f"Skipped: {pdb_id}"

        command = [
            "python", "train_classifier.py", train_file, test_file,
            "--random_seed", str(seed),
            "--wandb_run_name", run_name,
            "--wandb_project", wandb_project,
            # Use hyperparameters from the task list
            "--learning_rate", str(learning_rate),
            "--encoder_dim", str(encoder_dim),
            # Use non-swept hyperparameters passed from the runner's main function
            "--n_inducing_points", str(n_inducing_points),
            "--epochs", str(epochs),
            "--logger", logger,
            "--log_dir", log_dir
        ]
        subprocess.run(command, check=True)
        print(f"Rank {rank} successfully completed task: PDB={pdb_id}, Seed={seed}", flush=True)
        return f"Success: {pdb_id}"
    except Exception as e:
        print(f"Rank {rank} FAILED task: {task_line}. Error: {e}", flush=True)
        return f"Failed: {task_line}"

def main():
    parser = argparse.ArgumentParser(description="MPI runner for training jobs.")
    # Arguments that are constant for the entire campaign
    parser.add_argument("--datasets_dir", type=str, required=True)
    parser.add_argument("--task_list", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200000)
    parser.add_argument("--n_inducing_points", type=int, default=100)
    parser.add_argument("--wandb_project", type=str, default="grok_pdbbind")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"])
    parser.add_argument("--log_dir", type=str, default="logs", help="Base directory for local TensorBoard logs.")

    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    with MPICommExecutor(comm, root=0) as executor:
        if executor is not None:
            print(f"Manager (Rank 0) started with {comm.Get_size() - 1} workers.")
            try:
                with open(args.task_list, 'r') as f: tasks = f.read().strip().split('\n')
            except FileNotFoundError:
                print(f"Error: Task list '{args.task_list}' not found.")
                tasks = []
            
            if tasks:
                # Prepare tuples for the worker function, including the non-swept parameters
                task_tuples = zip(
                    tasks, 
                    repeat(args.datasets_dir),
                    repeat(args.epochs),
                    repeat(args.n_inducing_points),
                    repeat(args.wandb_project),
                    repeat(args.logger),
                    repeat(args.log_dir)
                )
                results = executor.map(run_training_task, task_tuples)
                completed_count = sum(1 for _ in results)
                print(f"Manager (Rank 0): All {completed_count}/{len(tasks)} tasks processed.")

    if rank == 0:
        print("Manager (Rank 0): Job complete.")

if __name__ == "__main__":
    main()


