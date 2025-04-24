import os
import json 
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import defaultdict
from constant import attack_method, search_metrics


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default="results/val_acc/SO-NAS201-1_GA_synflow_1304145745",
    )
    parser.add_argument(
        "--metric", type=int, default=9, help="type of attack", choices=range(0, 6)
    )
    parser.add_argument(
        "--option", type=int, default=1, help="type of attack", choices=range(0, 6)
    )
    return parser.parse_args()

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
        "autoattack"
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
            stats[key_name].append(gen_val[0]["search_metric"])
        
    return dict(stats)

def collect_total_eval_attacking(base_dir: str, best_arch_filename: str): 
    results = defaultdict(list)
    for attack in os.listdir(base_dir):
        attack_dir = os.path.join(base_dir, attack)
        for folder in os.listdir(attack_dir):
            folder_path = os.path.join(attack_dir, folder)
            results[attack] = collect_acc_statistics(folder_path, best_arch_filename)
    return results


def visualize_nGen_accuracy(stats: dict, metric_name: str, save_dir: str= "figures"): 
    # format input data 
    gens = []
    means = [] 
    stds = []
    for gen_key in sorted(stats.keys(), key=lambda x: int(x.replace('gen', ''))):
        gens.append(int(gen_key.replace('gen', '')))
        means.append(stats[gen_key]['mean'])
        stds.append(stats[gen_key]['std'])
    
    means = np.array(means)
    stds = np.array(stds)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(gens, means, color="darkorange")
    plt.fill_between(gens, means - stds, means + stds, color="navajowhite", alpha=0.2)

    # Set plot titles and labels
    plt.title(f"Search by {metric_name}")
    plt.xlabel('Generation')
    plt.ylabel('Test Accuracy')

    # Set x-axis ticks with intervals (e.g., every 30 generations)
    xtick_step = 30
    max_gen = max(gens)
    plt.xticks(np.arange(0, max_gen + 1, xtick_step))

    # Add grid, legend, and layout adjustments
    # plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    # Create directory if it doesn't exist and save the figure
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, f"{metric_name}_accuracy.png")
    plt.savefig(fig_path, dpi=300)

    # Display the plot
    plt.show()


def visualize_synthetic_data(data_dict, save_dir: str= "figures/"): 
    """visualize multiple test accuracy results with standard deviation as fill area."""
    # create plot 
    plt.figure(figsize=(10, 6))

    # define colors for different lines 
    colors = {
        'val_fgsm_3.0_acc': {'fill': 'peachpuff', 'line': 'coral', 'point': 'darkred'},
        'val_fgsm_8.0_acc': {'fill': 'navajowhite', 'line': 'darkorange', 'point': 'orangered'},
        'val_pgd_3.0_acc': {'fill': 'lightblue', 'line': 'dodgerblue', 'point': 'navy'},
        'val_pgd_8.0_acc': {'fill': 'lightcyan', 'line': 'teal', 'point': 'darkcyan'},
        'autoattack': {'fill': 'lightgreen', 'line': 'forestgreen', 'point': 'darkgreen'},
        'val_acc': {'fill': 'lavender', 'line': 'purple', 'point': 'darkmagenta'}
    }

    # iterate through each key in the data dictionary 
    for key in data_dict.keys(): 
        acc_values = np.array(data_dict[key]['max_test'])
        std_values = np.array(data_dict[key]['std_test'])
        x = list(range(len(acc_values)))

        # fill between 
        plt.fill_between(
            x,
            acc_values - std_values,
            acc_values + std_values,
            color=colors[key]['fill'],
            alpha=0.5,
        )

        # line plot for accuracy 
        plt.plot(
            x,
            acc_values, 
            color=colors[key]['line'], 
            linewidth= 2, 
            label= f"{key}"
        )

        plt.scatter(
            x, 
            acc_values, 
            color= colors[key]['point'], 
            s= 40, 
            marker= "s", 
        )
    
    # axis and label title 
    plt.xlabel("Run Index") 
    plt.ylabel("Test Accuracy") 
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # save figure 
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"accuracy_comparison.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.clf()

def visualize_column_chart(attack_stats, optimal_stats, save_dir: str = "figures"):
    keys = list(attack_stats.keys())
    x = np.arange(len(keys))  
    width = 0.35             

    # initialize the figure
    plt.figure(figsize=(10, 6))

    # plot bars for attack statistics
    plt.bar(x - width/2, [attack_stats[k] for k in keys], width, 
            label='Attack Stats', color='skyblue')

    # plot bars for optimal statistics
    plt.bar(x + width/2, [optimal_stats[k] for k in keys], width, 
            label='Optimal Statistic', color='orange')

    # set labels and title
    plt.xticks(x, keys, rotation=45)
    plt.ylabel('Accuracy')
    plt.title('Comparison of Attack Stats vs Optimal Statistic')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    # improve layout
    plt.tight_layout()

    # show the plot
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"optimal.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.clf()

if __name__ == "__main__":
    best_arch_filename = "best_architecture_each_gen.p"
    args = parse_arguments()
    if args.option == 1: 
        statistics = collect_acc_statistics(
            args.base_dir, best_arch_filename
        )
        summray_stats = {}
        for gen, gen_exp in statistics.items():
            gen_exp_arr = np.array(gen_exp)
            summray_stats[gen] = {
                "mean": np.mean(gen_exp_arr),
                "std": np.std(gen_exp_arr)
            }
        
        visualize_nGen_accuracy(
            stats= summray_stats, 
            metric_name= search_metrics[args.metric], 
        )

    elif args.option == 2: 
        statistics = collect_total_eval_attacking("results/", best_arch_filename)
        visualize_synthetic_data(statistics)    
    
    elif args.option == 3: 
        cfg_path = "config/cifar10.json"
        optimal_statistic = get_optimal_statistics(cfg_path)
        statistics = collect_total_eval_attacking("results/", best_arch_filename)
        attack_stats = {}
        for metric, stat in statistics.items(): 
            attack_stats[metric] = np.max(stat["max_test"])
        visualize_column_chart(attack_stats, optimal_statistic)