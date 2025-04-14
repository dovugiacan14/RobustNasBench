import os 
import json
import pickle 
import numpy as np
from pathlib import Path
from collections import defaultdict

def collect_acc_statistics(base_result_dir: str, best_arch_filename: str):
    """
    Collects testing accuracy statistics across multiple experiment folders.

    Each experiment folder is expected to contain a pickled file with validation
    results. This function loads these files, extracts the testing accuracy per
    generation, and aggregates them for statistical analysis.

    Args:
        base_result_dir (str): Path to the directory containing experiment subfolders.
        best_arch_filename (str): Name of the pickle file in each subfolder that 
                                  stores accuracy results.

    Returns:
        dict: A dictionary where each key is 'gen{index}' and the value is a list of 
              testing accuracies for that generation across all experiments.
    """
    # convert to Path object
    base_result_dir = Path(base_result_dir)
    if not base_result_dir.exists():
        raise FileNotFoundError(f"Base result directory not found: {base_result_dir}")
    
    stats = defaultdict(list)
    # iterate over each subdirectory
    for subdir_name in os.listdir(base_result_dir):
        subdir_path = base_result_dir / subdir_name
        file_path = subdir_path / best_arch_filename.strip()

        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        # extract accuracy values
        for num_gen, gen_val in enumerate(data[1]):
            key_name = f"gen{num_gen + 0}"
            stats[key_name].append(gen_val[0]["testing_accuracy"])
        
    return dict(stats)


def collect_total_eval_attacking(base_dir: str, best_arch_filename: str): 
    for sub_dir_name in os.listdir(base_dir):
        sub_dir = os.path.join(base_dir, sub_dir_name)
        stats = collect_acc_statistics(sub_dir , best_arch_filename)
    
    results = {}
    for gen, stat in stats.items(): 
        stat = np.array(stat)
        results[gen] = {
            "mean": np.mean(stat), 
            "std": np.std(stat)
        }
    
    return results["gen149"]["mean"], results["gen149"]["std"]

def get_optimal_statistics(cfg_path):
    with open(cfg_path, "r") as file:
        cfg = json.load(file)

    # Define the keys to extract from each architecture
    stat_keys = [
        "val_acc",
        "val_fgsm_3.0_acc",
        "val_fgsm_8.0_acc",
        "val_pgd_3.0_acc",
        "val_pgd_8.0_acc",
        "autoattack", 
        "synflow"
    ]

    # Use defaultdict to automatically initialize empty lists
    stats_dict = defaultdict(list)

    for statistic in cfg.values():
        for key in stat_keys:
            value = (
                statistic[key]["threeseed"]
                if key != "autoattack"
                else statistic[key]
            )
            stats_dict[key].append(value)

    # Get max value for each key
    optimal_stats = {key: max(values) for key, values in stats_dict.items()}
    return optimal_stats

if __name__ == "__main__": 
    base_path = "results/"
    for path in os.listdir(base_path): 
        print(f"==============={path}=================")
        base_dir = os.path.join(base_path, path)
        best_arch_filename = "best_architecture_each_gen.p"
        acc, std = collect_total_eval_attacking(base_dir, best_arch_filename)
        print(f"Acc= {acc * 100}")
        print(f"Std= {std}")

    # cfg_path = "config/cifar10.json"
    # optimal_statistic = get_optimal_statistics(cfg_path)
    # print(0)