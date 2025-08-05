import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from constant import AVAILABLE_OPERATIONS, OP_NAMES_NB201, EDGE_LIST

robustness_stats_path = "config/cifar10.json"
with open(robustness_stats_path, "r") as f:
    robustness_db = json.load(f)

data_path = "data/NASBench201/[CIFAR-10]_data.p"
with open(data_path, "rb") as f:
    nasbench201_data = pickle.load(f)


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
        nondominated_point_mask = np.any(costs > costs[next_point_index], axis=1)
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


def select_closest_to_ideal(x_values, y_values, pareto_mask):
    ideal = np.array([x_values.max(), y_values.max()])
    pareto_points = np.stack((x_values[pareto_mask], y_values[pareto_mask]), axis=1)
    distances = np.linalg.norm(pareto_points - ideal, axis=1)
    selected_pareto_idx = np.argmin(distances)

    # Convert back to original index
    original_indices = np.nonzero(pareto_mask)[0]
    selected_idx = original_indices[selected_pareto_idx]
    selected_point = (x_values[selected_idx], y_values[selected_idx])

    return selected_idx, selected_point


def visualize_pareto_front(
    x_values,
    y_values,
    mask,
    save_path=None,
    x_label="Clean Test Accuracy",
    y_label="FGSM (ε=8) Accuracy",
    title="Pareto Front Visualization",
):
    """
    Visualize Pareto front on a 2D scatter plot and optionally save the figure.
    """
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    mask = np.array(mask)

    x_pareto = x_values[mask]
    y_pareto = y_values[mask]

    x_rest = x_values[~mask]
    y_rest = y_values[~mask]

    pareto_sorted_indices = np.argsort(x_pareto)
    x_pareto_sorted = x_pareto[pareto_sorted_indices]
    y_pareto_sorted = y_pareto[pareto_sorted_indices]

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(9, 7))

    # Nền sáng xám
    ax.set_facecolor("#f7f2f2")
    fig.patch.set_facecolor("#f7f2f2")

    # Scatter plots
    ax.scatter(x_rest, y_rest, c="green", label="Non-Pareto", alpha=0.6, s=60)
    ax.scatter(
        x_pareto_sorted,
        y_pareto_sorted,
        c="red",
        label="Pareto Front",
        edgecolors="black",
        s=90,
        zorder=3,
    )
    ax.plot(
        x_pareto_sorted,
        y_pareto_sorted,
        linestyle="--",
        color="red",
        alpha=0.7,
        linewidth=2,
    )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=15)

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)

    legend = ax.legend(loc="lower right", fontsize=12, frameon=True)
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1.2)
    legend.get_frame().set_facecolor("#e0e0e0")

    ax.grid(True, linestyle=":", linewidth=1, color="gray")
    fig.tight_layout()

    if save_path:
        save_path_name = os.path.join(save_path, "robustness_clean_pareto.png")
        plt.savefig(save_path_name, dpi=300)
        print(f"Plot saved to: {save_path_name}")
    
    plt.close()

    # plt.show()


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


def get_robustness_stats(data: dict, target_folder):
    last_gen_archs = data[-1][-1]
    non_dominated_archs = last_gen_archs[0]
    non_dominated_archs = [decode_architecture(arch) for arch in non_dominated_archs]
    haskey_archs = last_gen_archs[1]

    # get test accuracy
    result = defaultdict(list)
    for arch in haskey_archs:
        result["clean_acc"].append(nasbench201_data["200"][arch]["test_acc"][-1])

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

    clean_acc_lst = np.array(result["clean_acc"])
    fgsm_8_lst = np.array(result["val_fgsm_8"])
    costs = np.stack([clean_acc_lst, fgsm_8_lst], axis=1)
    mask = is_pareto_efficient(costs)
    visualize_pareto_front(clean_acc_lst, fgsm_8_lst, mask, target_folder)
    
    selected_idx, selected_point = select_closest_to_ideal(clean_acc_lst, fgsm_8_lst, mask)
    for key in result:
        result[key] = result[key][selected_idx]
    return result


# Synflow, L2, SNip, fisher, Jacov/NWOT
def export_result_to_excel(data, output_path):
    with pd.ExcelWriter(output_path) as writer:
        for metric, data in data.items():
            df = pd.DataFrame(data).T.reset_index()
            df.columns = ["Objective"] + list(df.columns[1:])
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

                robustness_stats = get_robustness_stats(data, n_run_path)
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
    output_path = "cifar10_stats.xlsx"
    result = collect_total_stats(base_path, filename, output_path)
