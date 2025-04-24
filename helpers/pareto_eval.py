import os
import json
import pickle
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from collections import defaultdict


load_dotenv()
base_dir = Path(os.environ["PARETO_FRONT_DIR"])
os.makedirs(base_dir, exist_ok=True)


OP_NAMES_NB201 = [
    "skip_connect",
    "none",
    "nor_conv_3x3",
    "nor_conv_1x1",
    "avg_pool_3x3",
]


def normalize_data(data_list: list):
    min_par = min(data_list)
    max_par = max(data_list)
    result = []
    for par in data_list:
        new_par = round((par - min_par) / (max_par - min_par), 6)
        result.append(new_par)
    return result

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


def get_min_max_complexity_metric(zero_suite_data):
    flops_list = []
    params_list = []
    for arch, zero_suite_stats in zero_suite_data.items():
        flops = zero_suite_stats["flops"]["score"]
        params = zero_suite_stats["params"]["score"]
        flops_list.append(flops)
        params_list.append(params)
    return min(flops_list), max(flops_list), min(params_list), max(params_list)


# Very slow for many datapoints.  Fastest for many costs, most readable
def is_pareto_efficient_dumb(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i] > c, axis=1)) and np.all(
            np.any(costs[i + 1 :] > c, axis=1)
        )
    return is_efficient


# Fairly fast for many datapoints, less fast for many costs, somewhat readable
def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                costs[is_efficient] < c, axis=1
            )  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


# Faster than is_pareto_efficient_simple, but less readable.
def is_pareto_efficient(costs, return_mask=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def plot_points(points, metric_name):
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]

    plt.figure(figsize=(12, 8))

    # Plot the points and connect them with a line
    plt.scatter(x_vals, y_vals, color="blue", s=50, label="Pareto Front Points")
    plt.plot(x_vals, y_vals, "r--", linewidth=2, label="Pareto Front")

    # Add labels based on the metric
    if "flops" in metric_name:
        plt.xlabel("FLOPs", fontsize=12)
    elif "params" in metric_name:
        plt.xlabel("Parameters", fontsize=12)

    # Extract the metric name for the y-axis
    metric = metric_name.split("_")[-1]
    plt.ylabel(f'1 - {metric.replace("_", " ").title()}', fontsize=12)

    plt.title(f'Pareto Front: {metric_name.replace("_", " ").title()}', fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=10)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"pareto_{metric_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    zero_suite_nasbench201_path = "config/zc_nasbench201.json"
    robustness_path = "config/imagenet.json"

    dataset = robustness_path.split("/")[-1].split(".")[0]
    if dataset == "imagenet":
        dataset = "ImageNet16-120"
    sub_dir = os.path.join(base_dir, dataset)
    os.makedirs(sub_dir, exist_ok=True)

    with open(zero_suite_nasbench201_path, "r") as f:
        zero_suite_nasbench201_conf = json.load(f)
    with open(robustness_path, "r") as f:
        robustness_conf = json.load(f)

    zero_suite_data = zero_suite_nasbench201_conf[dataset]
    min_flops, max_flops, min_params, max_params = get_min_max_complexity_metric(zero_suite_data)
    summary_stats = defaultdict(list)

    for arch, zero_suite_stats in zero_suite_data.items():
        str_arch = decode_architecture(arch)
        flops = round((zero_suite_stats["flops"]["score"] - min_flops) / (max_flops - min_flops), 6)
        params = round((zero_suite_stats["params"]["score"] - min_params) / (max_params - min_params), 6)
       
        val_acc_clean = robustness_conf[str_arch]["val_acc"]["threeseed"]
        val_fgsm3 = robustness_conf[str_arch]["val_fgsm_3.0_acc"]["threeseed"]
        val_fgsm8 = robustness_conf[str_arch]["val_fgsm_8.0_acc"]["threeseed"]
        val_pgd3 = robustness_conf[str_arch]["val_pgd_3.0_acc"]["threeseed"]
        val_pgd8 = robustness_conf[str_arch]["val_pgd_8.0_acc"]["threeseed"]
        auto_attack = robustness_conf[str_arch]["autoattack"]

        summary_stats["flops_rob_val_acc"].append((flops, 1 - val_acc_clean))
        summary_stats["flops_val_fgsm_3"].append((flops, 1 - val_fgsm3))
        summary_stats["flops_val_fgsm_8"].append((flops, 1 - val_fgsm8))
        summary_stats["flops_val_pgd_3"].append((flops, 1 - val_pgd3))
        summary_stats["flops_val_pgd_8"].append((flops, 1 - val_pgd8))
        summary_stats["flops_autoattack"].append((flops, 1 - auto_attack))

        summary_stats["params_rob_val_acc"].append((params, 1 - val_acc_clean))
        summary_stats["params_val_fgsm_3"].append((params, 1 - val_fgsm3))
        summary_stats["params_val_fgsm_8"].append((params, 1 - val_fgsm8))
        summary_stats["params_val_pgd_3"].append((params, 1 - val_pgd3))
        summary_stats["params_val_pgd_8"].append((params, 1 - val_pgd8))
        summary_stats["params_autoattack"].append((params, 1 - auto_attack))

    for key, val in summary_stats.items():
        points = np.array(val)
        pareto_mask = is_pareto_efficient(points, return_mask=True)
        pareto_front = points[pareto_mask]
        pareto_front = list(set(map(tuple, pareto_front)))
        pareto_front.sort(key=lambda x: x[0])
        plot_points(pareto_front, key)
        save_path = os.path.join(sub_dir, f"{key}.p")
        with open(save_path, "wb") as f:
            pickle.dump(pareto_front, f)
            print(f"Saved {key} to {save_path}")
        print(f"Pareto front for {key}: {pareto_front}")
