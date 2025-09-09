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
    task_line, datasets_dir = task_tuple
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # --- Determine Local GPU ---
    local_rank = int(os.environ.get('SLURM_LOCALID', rank % 8))

    # --- Unpack Task and Run ---
    try:
        pdb_id, seed = task_line.split(',')
        
        print(f"Rank {rank} (Local GPU {local_rank}) starting task: PDB={pdb_id}, Seed={seed}", flush=True)

        # Use the passed datasets_dir to construct paths
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
            "--random_seed", seed,
            "--wandb_run_name", run_name,
            "--wandb_project", "grok_pdbbind",
            "--learning_rate", "1e-5",
            "--n_inducing_points", "100",
            "--encoder_dim", "16",
            "--epochs", "200000"
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
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    with MPICommExecutor(comm, root=0) as executor:
        if executor is not None:  # True for the root process (rank 0)
            print(f"Manager (Rank 0) started with {comm.Get_size() - 1} workers.")
            
            try:
                with open('task_list.txt', 'r') as f:
                    tasks = f.read().strip().split('\n')
            except FileNotFoundError:
                print("Error: task_list.txt not found. Please generate it first.")
                tasks = []

            if tasks:
                # Combine tasks with the datasets_dir for the worker function
                task_tuples = zip(tasks, repeat(args.datasets_dir))
                results = executor.map(run_training_task, task_tuples)
                
                completed_count = 0
                for result in results:
                    completed_count += 1
                
                print(f"Manager (Rank 0): All {completed_count}/{len(tasks)} tasks have been processed.")

    if rank == 0:
        print("Manager (Rank 0): Job complete.")

if __name__ == "__main__":
    main()


