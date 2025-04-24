import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
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

    val_acc = defaultdict(list)
    test_acc = defaultdict(list)
    rob_val_acc = defaultdict(list)
    fgsm3 = defaultdict(list)
    fgsm8 = defaultdict(list)
    pgd3 = defaultdict(list)
    pgd8 = defaultdict(list)
    autoattack = defaultdict(list)
    # iterate over each subdirectory
    for subdir_name in os.listdir(base_result_dir):
        subdir_path = base_result_dir / subdir_name
        file_path = subdir_path / best_arch_filename.strip()

        if not file_path.exists():
            logging.info(f"File not found: {file_path}")
            continue

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        # extract accuracy values
        for num_gen, gen_val in enumerate(data[1]):
            key_name = f"gen{num_gen + 0}"
            val_acc[key_name].append(gen_val[0]["val_acc"])
            test_acc[key_name].append(gen_val[0]["test_acc"])
            rob_val_acc[key_name].append(gen_val[0]["robust_acc"]["rob_val_acc"])
            fgsm3[key_name].append(gen_val[0]["robust_acc"]["val_fgsm_3"])
            fgsm8[key_name].append(gen_val[0]["robust_acc"]["val_fgsm_8"])
            pgd3[key_name].append(gen_val[0]["robust_acc"]["val_pgd_3"])
            pgd8[key_name].append(gen_val[0]["robust_acc"]["val_pgd_8"])
            autoattack[key_name].append(gen_val[0]["robust_acc"]["autoattack"])

    return {
        "val_acc": dict(val_acc),
        "test_acc": dict(test_acc),
        "rob_val_acc": dict(rob_val_acc),
        "fgsm3": dict(fgsm3),
        "fgsm8": dict(fgsm8),
        "pgd3": dict(pgd3),
        "pgd8": dict(pgd8),
        "auto_attack": dict(autoattack),
    }


def collect_total_eval_attacking(base_dir: str, best_arch_filename: str):
    all_metrics = {}

    for sub_dir_name in os.listdir(base_dir):
        sub_dir = os.path.join(base_dir, sub_dir_name)
        stats = collect_acc_statistics(sub_dir, best_arch_filename)

    for type_metric, stat in stats.items():
        last_gen_eval = stat["gen149"]
        last_gen_eval = np.array(last_gen_eval)
        mean = np.mean(last_gen_eval) * 100.00
        std = np.std(last_gen_eval)
        formatted = f"{mean:.2f} ± {std:.4f}"
        all_metrics[type_metric] = formatted

    return all_metrics


def export_all_eval_results(base_path: str, output_excel_path: str = "output.xlsx"):
    best_arch_filename = "best_architecture_each_gen.p"

    with pd.ExcelWriter(output_excel_path, engine="openpyxl", mode="w") as writer:
        for path in os.listdir(base_path):
            print(f"******************{path}*********************")
            base_dir = os.path.join(base_path, path)
            all_metrics = collect_total_eval_attacking(base_dir, best_arch_filename)

            df_result = pd.DataFrame([all_metrics])
            df_result.to_excel(writer, sheet_name=path, index=False)


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
    ]

    # Use defaultdict to automatically initialize empty lists
    stats_dict = defaultdict(list)

    for statistic in cfg.values():
        for key in stat_keys:
            value = (
                statistic[key]["threeseed"] if key != "autoattack" else statistic[key]
            )
            stats_dict[key].append(value)

    # Get max value for each key
    optimal_stats = {key: max(values) for key, values in stats_dict.items()}
    return optimal_stats


def get_optimal_acc_stats(p_path):
    with open(p_path, "rb") as p:
        data = pickle.load(p)

    stats = data["200"]
    test_acc_lst = []
    for _, val in stats.items():
        test_acc = val["test_acc"][-1]
        test_acc_lst.append(test_acc)
    return np.max(test_acc_lst)


if __name__ == "__main__":
    base_path = "results/"
    cfg_path = "config/imagenet.json"
    nasbench_path = "data/NASBench201/[ImageNet16-120]_data.p"

    export_all_eval_results(base_path)
    # optimal_stats = get_optimal_statistics(cfg_path)
    # optimal_stats = get_optimal_acc_stats(nasbench_path)
