Of course, here is the updated README.md with the Delta Lake creation step included.

# Data Preparation and Analysis Pipeline

This document outlines the steps to process protein-ligand data, from initial characterization to final analysis and visualization. The process is organized into a series of scripts that should be run sequentially.

-----

## 1\. Pocket Characterization

The first step is to characterize the binding pockets of the proteins. This is done using the `pocket_characterization.py` script. This script takes a PDBbind index file as input, downloads the corresponding PDB files, and calculates a set of physicochemical properties for the binding pocket of each protein.

**Example:**

```bash
python pocket_characterization.py INDEX_refined_data.2020
```

This will create a `target_pocket_metadata.csv` file containing the pocket features for each PDB ID.

-----

## 2\. Cluster and Select Representatives

Next, we cluster the binding pockets based on their physicochemical properties and select a representative PDB ID from each cluster. This is done to create a diverse and representative dataset for training. The `cluster_and_select.py` script takes the `target_pocket_metadata.csv` file as input and outputs a text file with the selected PDB IDs.

**Example:**

```bash
python cluster_and_select.py --metadata_file ../data/target_pocket_metadata.csv --n_clusters 200 --output_file ../data/representative_pdb_ids.txt
```

-----

## 3\. Curriculum Build/Dataset Creation

### 3.1 Create Delta Lake from BindingDB

Before building the individual datasets, we first process the raw BindingDB data into a Delta Lake. This provides a more efficient and queryable format for the next step. The `create_delta_lake.py` script handles this conversion.

**Example:**

```bash
python create_delta_lake.py BindingDB_All.tsv --output_path bindingdb_delta
```

This will create a `bindingdb_delta` directory containing the Delta Lake.

### 3.2 Dataset Creation

Now we build the datasets for each of the selected representative PDB IDs. The `run_curriculum_builder.py` script orchestrates this process by calling the `scaffold_splitter.py` script. This process involves filtering the Delta Lake of ligand information, splitting the data into training and testing sets based on molecular scaffolds, and saving them into individual directories for each target.

**Example:**

```bash
python run_curriculum_builder.py bindingdb_delta ../data/representative_pdb_ids.txt --output_dir datasets
```

This will create a `datasets` directory containing subdirectories for each PDB ID, with `train.csv` and `test.csv` files inside each.

-----

## 4\. Slurm Script Generation

After creating the datasets, the next step is to generate the SLURM script to launch the training jobs on a cluster. The `launch_slurm_jobs.py` script creates a job array that will run multiple MPI jobs in parallel.

**Example:**

```bash
python launch_slurm_jobs.py datasets --account=your_account_name
```

This will generate a `submit_array.slurm` file. Remember to edit this file to set up your correct environment before submitting the job.

-----

## 5\. Launch Slurm Job

Submit the generated SLURM script to the queue.

**Example:**

```bash
sbatch submit_array.slurm
```

This will launch the training jobs. Each job will process a chunk of the training tasks defined in the `launch_slurm_jobs.py` script, running the training via the `mpi_runner.py` and `train_classifier.py` scripts.

-----

## 6\. Postprocessing

### Analyze Run

After the training jobs are complete, analyze the TensorBoard logs for each run to detect grokking events. The `analyze_run.py` script processes a directory of logs and creates a summary CSV file.

**Example:**

```bash
python analyze_run.py logs/ --output_file grokking_analysis_summary.csv
```

### Meta-Analysis

Perform a meta-analysis to correlate the grokking behavior with the pocket's physicochemical properties. The `meta_analyzer.py` script takes the grokking summary and the pocket metadata as input and produces a combined analysis file.

**Example:**

```bash
python meta_analyzer.py grokking_analysis_summary.csv ../data/target_pocket_metadata.csv
```

-----

## 7\. Create Plots

Finally, visualize the results by creating plots of the learning curves and the meta-analysis.

**Learning Curves:**
The `visualize_curves.py` script generates a figure of representative learning curves from the analysis.

**Example:**

```bash
python visualize_curves.py grokking_meta_analysis.csv grokking_analysis_summary.csv logs/
```

**Analysis Visualization:**
The `visualize_analysis.py` script creates a scatter plot to visualize the relationship between pocket properties and grokking behavior.

**Example:**

```bash
python visualize_analysis.py grokking_meta_analysis.csv
```
