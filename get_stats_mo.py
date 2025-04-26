import os
import json
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from constant import AVAILABLE_OPERATIONS, OP_NAMES_NB201, EDGE_LIST

robustness_stats_path = "config/cifar10.json"
with open(robustness_stats_path, "r") as f:
    robustness_db = json.load(f)

data_path = "data/NASBench201/[CIFAR-10]_data.p" 
with open(data_path, "rb") as f:
    nasbench201_data = pickle.load(f)


def decode_architecture(encoded_architecture):
    # Handle string input
    if isinstance(encoded_architecture, str):
        # Remove parentheses and split by comma
        encoded_architecture = tuple(
            map(int, encoded_architecture.strip("()").split(","))
        )

    # Now process as tuple
    ops = [OP_NAMES_NB201[idx] for idx in encoded_architecture]
    return "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(*ops)

def convert_str_to_ops(str_encoding):
    """
    Converts NB201 string representation to op_indices
    """
    nodes = str_encoding.split("+")

    def get_op(x):
        return x.split("~")[0]

    node_ops = [list(map(get_op, n.strip()[1:-1].split("|"))) for n in nodes]

    enc = []
    for u, v in EDGE_LIST:
        enc.append(OP_NAMES_NB201.index(node_ops[v - 2][u - 1]))

    return str(tuple(map(int, enc)))


def get_robustness_stats(data: dict):
    last_gen_archs = data[-1][-1]
    non_dominated_archs = last_gen_archs[0]
    non_dominated_archs = [decode_architecture(arch) for arch in non_dominated_archs]
    haskey_archs = last_gen_archs[1]

    # get test accuracy 
    result = defaultdict(list)
    for arch in haskey_archs:
        result["clean_acc"].append(nasbench201_data['200'][arch]["test_acc"][-1])

    # get robustness stats 
    for arch in non_dominated_archs:
        result["rob_val_acc"].append(robustness_db[arch]["val_acc"]["threeseed"])
        result["val_fgsm_3"].append(
            robustness_db[arch]["val_fgsm_3.0_acc"]["threeseed"]
        )
        result["val_fgsm_8"].append(
            robustness_db[arch]["val_fgsm_8.0_acc"]["threeseed"]
        )
        result["val_pgd_3"].append(robustness_db[arch]["val_pgd_3.0_acc"]["threeseed"])
        result["val_pgd_8"].append(robustness_db[arch]["val_pgd_8.0_acc"]["threeseed"])
        result["autoattack"].append(robustness_db[arch]["autoattack"])

    for key in result:
        result[key] = np.max(result[key])

    return result

def export_result_to_excel(data, output_path):
    with pd.ExcelWriter(output_path) as writer:
        for metric, data in data.items():
            df = pd.DataFrame(data).T.reset_index()
            df.columns = ['Objective'] + list(df.columns[1:])  
            df.to_excel(writer, sheet_name=metric, index=False)


def collect_total_stats(base_dir: str, filename: str, output_path: str):
    all_metrics_dict = {}
    for metric in os.listdir(base_dir):
        all_metrics = {}
        sub_dir = os.path.join(base_dir, metric)
        if not os.path.isdir(sub_dir):
            continue  
        for objectives in os.listdir(sub_dir):
            obj_dir = os.path.join(sub_dir, objectives)
            sub_folder = os.listdir(obj_dir)[0]
            sub_folder_path = os.path.join(obj_dir, sub_folder)
            summary_stats = defaultdict(list)
            for n_run in os.listdir(sub_folder_path):
                if n_run == "logging.txt":
                    continue
                n_run_path = os.path.join(sub_folder_path, n_run)
                file_path = os.path.join(n_run_path, filename)
                if not os.path.exists(file_path):
                    print(f"File not found: {file_path}")
                    continue

                with open(file_path, "rb") as f:
                    data = pickle.load(f)

                robustness_stats = get_robustness_stats(data)
                for key in robustness_stats:
                    summary_stats[key].append(robustness_stats[key])

            all_metrics[objectives] = {
                key: f"{np.mean(summary_stats[key])*100:.2f} ± {np.std(summary_stats[key]):.4f}"
                for key in summary_stats
            }
        all_metrics_dict[metric] = all_metrics
    export_result_to_excel(all_metrics_dict, output_path)

if __name__ == "__main__":
    base_path = "results/"
    filename = "#Evals_and_Elitist_Archive_search.p"
    output_path = "results/cifar10_stats.xlsx"
    result = collect_total_stats(base_path, filename, output_path)
    pass
