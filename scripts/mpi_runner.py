import os
import subprocess
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
import argparse
from itertools import repeat

def run_training_task(task_tuple):
    # ... (function content is the same, but now receives logger args)
    task_line, datasets_dir, epochs, encoder_dim, n_inducing_points, wandb_project, learning_rate, logger, log_dir = task_tuple
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    try:
        pdb_id, seed = task_line.split(',')
        print(f"Rank {rank} starting task: PDB={pdb_id}, Seed={seed}", flush=True)
        train_file = os.path.join(datasets_dir, pdb_id, "train.csv")
        test_file = os.path.join(datasets_dir, pdb_id, "test.csv")
        run_name = f"{pdb_id}-seed-{seed}"

        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print(f"Rank {rank} skipping task {pdb_id}: Dataset files not found.", flush=True)
            return f"Skipped: {pdb_id}"

        command = [
            "python", "../svgp/train_classifier.py", train_file, test_file,
            "--random_seed", str(seed), "--wandb_run_name", run_name,
            "--wandb_project", wandb_project, "--learning_rate", str(learning_rate),
            "--n_inducing_points", str(n_inducing_points), "--encoder_dim", str(encoder_dim),
            "--epochs", str(epochs), "--logger", logger, "--log_dir", log_dir
        ]
        subprocess.run(command, check=True)
        print(f"Rank {rank} successfully completed task: PDB={pdb_id}, Seed={seed}", flush=True)
        return f"Success: {pdb_id}"
    except Exception as e:
        print(f"Rank {rank} FAILED task: {task_line}. Error: {e}", flush=True)
        return f"Failed: {task_line}"

def main():
    parser = argparse.ArgumentParser(description="MPI runner for training jobs.")
    # ... (existing args)
    parser.add_argument("--datasets_dir", type=str, required=True)
    parser.add_argument("--task_list", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200000)
    parser.add_argument("--encoder_dim", type=int, default=16)
    parser.add_argument("--n_inducing_points", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--wandb_project", type=str, default="grok_pdbbind")
    # --- New Logging Arguments ---
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
                task_tuples = zip(
                    tasks, repeat(args.datasets_dir), repeat(args.epochs),
                    repeat(args.encoder_dim), repeat(args.n_inducing_points),
                    repeat(args.wandb_project), repeat(args.learning_rate),
                    repeat(args.logger), repeat(args.log_dir)
                )
                results = executor.map(run_training_task, task_tuples)
                completed_count = sum(1 for _ in results)
                print(f"Manager (Rank 0): All {completed_count}/{len(tasks)} tasks processed.")

    if rank == 0:
        print("Manager (Rank 0): Job complete.")

if __name__ == "__main__":
    main()


