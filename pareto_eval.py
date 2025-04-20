import os
import json
import pickle
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from collections import defaultdict


load_dotenv()

zero_cost_metrics = [
    "epe_nas",
    "fisher",
    "grad_norm",
    "grasp",
    "l2_norm",
    "jacov",
    "nwot",
    "plain",
    "snip",
    "synflow",
    "zen",
]

datasets = ["cifar10", "cifar100", "ImageNet16-120"]

base_dir = Path(os.environ["PARETO_FRONT_DIR"])
os.makedirs(base_dir, exist_ok= True)


def normalize_data(data_list: list):
    min_par = min(data_list)
    max_par = max(data_list)
    result = []
    for par in data_list:
        new_par = round((par - min_par) / (max_par - min_par), 6)
        result.append(new_par)
    return result


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


def plot_points(points, zc_metric):

    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]

    plt.figure(figsize=(10, 6))
    plt.scatter(x_vals, y_vals, color="blue", label="Candidates")
    plt.plot(
        x_vals, y_vals, color="red", linestyle="--", linewidth=2, label="Connected Line"
    )

    plt.xlabel("FLOPs")
    plt.ylabel(zc_metric)
    plt.title(f"FLOPs vs {zc_metric}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file = "config/zc_nasbench201.json"
    with open(file, "r") as f:
        conf = json.load(f)

    for dt in datasets:
        summary_stats = defaultdict(list)
        subdir = os.path.join(base_dir, dt)
        os.makedirs(subdir, exist_ok=True)
        for key, val in conf[dt].items():
            for zero_cost_metric in zero_cost_metrics:
                metric = val[zero_cost_metric]["score"]
                summary_stats[zero_cost_metric].append(metric)
            summary_stats["flops"].append(val["flops"]["score"])
            summary_stats["params"].append(val["params"]["score"])

        summary_stats = dict(summary_stats)
        flops = summary_stats["flops"]
        for zero_cost_metric in zero_cost_metrics:
            target = summary_stats[zero_cost_metric]
            neg_target = [par * (-1.0) for par in target]
            points = np.array(list(zip(flops, neg_target)))
            pareto_mask = is_pareto_efficient(points, return_mask=True)
            pareto_front = points[pareto_mask]
            pareto_front = list(set(map(tuple, pareto_front)))
            pareto_front.sort(key=lambda x: x[0])
            # plot_points(pareto_front, zero_cost_metric)
            save_path = os.path.join(subdir, f"flops_{zero_cost_metric}.p")
            with open(save_path, "wb") as f:
                pickle.dump(pareto_front, f)
