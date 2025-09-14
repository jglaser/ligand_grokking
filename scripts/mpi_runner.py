import argparse
import itertools
import subprocess
import os
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from tqdm.auto import tqdm
import pandas as pd

def run_task(task: dict):
    """
    Constructs and runs a training command based on a task dictionary.
    """
    target_id = task['target']
    lr = task['learning_rate']
    edim = task['encoder_dim']
    seed = task['random_seed']
    log_dir = task['log_dir']

    # Construct a descriptive run name and full log file path
    run_name = f"{target_id}-lr{lr}-edim{edim}-seed{seed}"
    log_file = os.path.join(log_dir, f"{run_name}.csv")

    # Construct the base command
    command = [
        "python", "../svgp/train_classifier.py",
        os.path.join(task['dataset_dir'], target_id, "train.csv"),
        os.path.join(task['dataset_dir'], target_id, "test.csv"),
        "--learning_rate", str(lr),
        "--encoder_dim", str(edim),
        "--random_seed", str(seed),
        "--log_file", log_file
    ]
    
    # Add optional arguments from the task
    if 'n_inducing_points' in task:
        command.extend(["--n_inducing_points", str(task['n_inducing_points'])])
    if 'temperature' in task:
        command.extend(["--temperature", str(task['temperature'])])
    if 'epochs' in task:
        command.extend(["--epochs", str(task['epochs'])])
    if 'batch_size' in task:
        command.extend(["--batch_size", str(task['batch_size'])])

    # This function now returns the result instead of printing,
    # as stdout can get messy with MPI.
    try:
        # We use DEVNULL to hide the verbose output of the training script
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"SUCCESS: {run_name}"
    except subprocess.CalledProcessError as e:
        return f"FAILURE: {run_name} exited with code {e.returncode}"

def main():
    parser = argparse.ArgumentParser(description="Run a hyperparameter sweep using MPI.")
    parser.add_argument("--datasets_dir", type=str, default="datasets", help="Parent directory containing the dataset subdirectories.")
    # --- THE CHANGE: Task list is now a CSV with hyperparameters ---
    parser.add_argument("--task_list", type=str, required=True, help="Path to a CSV file defining tasks (target, learning_rate, encoder_dim, random_seed).")
    parser.add_argument("--log_dir", type=str, default="logs_mpi", help="Parent directory to save all output CSV logs.")
    
    args = parser.parse_args()

    # Use MPICommExecutor for task distribution
    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is None:
            # Worker processes will wait here for tasks.
            return

        # The rest of the code is executed only on the root process (rank 0)
        
        # 1. Generate tasks from the CSV file
        tasks = []
        try:
            task_df = pd.read_csv(args.task_list)
            # Convert dataframe rows to a list of dictionaries
            tasks = task_df.to_dict('records')
            # Add static arguments to each task
            for task in tasks:
                task['dataset_dir'] = args.datasets_dir
                task['log_dir'] = args.log_dir

        except FileNotFoundError:
            print(f"Error: Task list file not found at '{args.task_list}'")
            tasks = []
        except Exception as e:
            print(f"Error reading or processing task file: {e}")
            tasks = []

        os.makedirs(args.log_dir, exist_ok=True)
        
        # 2. Map tasks to workers and collect results with a progress bar
        print(f"Root process distributing {len(tasks)} tasks to workers...")
        results = []
        # Use tqdm to create a progress bar for the map operation
        for result in tqdm(executor.map(run_task, tasks), total=len(tasks)):
            results.append(result)
        
        # 3. Process results
        print("\n--- MPI Run Summary ---")
        success_count = 0
        failure_count = 0
        
        for res in results:
            # Print failures for easier debugging
            if res.startswith("FAILURE"):
                print(res)
            if res.startswith("SUCCESS"):
                success_count += 1
            else:
                failure_count += 1
        print("-----------------------")
        print(f"Total tasks: {len(tasks)}")
        print(f"Successful runs: {success_count}")
        print(f"Failed runs: {failure_count}")
        print("-----------------------")


if __name__ == "__main__":
    main()


