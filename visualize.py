import os
import pickle
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from constant import attack_method


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default="results/val_fgsm_3.0_acc/SO-NAS201-1_GA_synflow_1104222020",
    )
    parser.add_argument(
        "--attack", type=int, default=0, help="type of attack", choices=range(0, 6)
    )
    return parser.parse_args()


def collect_acc_statistics(base_result_dir: str, best_arch_filename: str):
    """
    Collect mean and std of validation and test accuracy from experiment folders.

    Args:
        base_result_dir (str): Path to the base result directory.
        best_arch_filename (str): Filename containing architecture accuracy results.

    Returns:
        Tuple containing:
            - val_acc_mean_each_exp (List[float])
            - val_acc_std_each_exp (List[float])
            - test_acc_mean_each_exp (List[float])
            - test_acc_std_each_exp (List[float])
    """
    # convert to Path object
    base_result_dir = Path(base_result_dir)

    # initialize output lists
    val_acc_best_each_exp = []
    val_acc_mean_each_exp = []
    val_acc_std_each_exp = []

    test_acc_best_each_exp = []
    test_acc_mean_each_exp = []
    test_acc_std_each_exp = []

    if not base_result_dir.exists():
        raise FileNotFoundError(f"Base result directory not found: {base_result_dir}")

    # iterate over each subdirectory
    for subdir_name in os.listdir(base_result_dir):
        val_acc_each_exp = []
        test_acc_each_exp = []

        subdir_path = base_result_dir / subdir_name
        file_path = subdir_path / best_arch_filename.strip()

        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        # extract accuracy values
        nGens_each_run = data[1]
        for gen in nGens_each_run:
            val_acc = gen[0]["validation_accuracy"]
            test_acc = gen[0]["testing_accuracy"]
            val_acc_each_exp.append(val_acc)
            test_acc_each_exp.append(test_acc)

        # convert to numpy for stats
        val_acc_each_exp = np.array(val_acc_each_exp)
        test_acc_each_exp = np.array(test_acc_each_exp)

        val_acc_best_each_exp.append(np.max(val_acc_each_exp))
        val_acc_mean_each_exp.append(np.mean(val_acc_each_exp))
        val_acc_std_each_exp.append(np.std(val_acc_each_exp))

        test_acc_best_each_exp.append(np.max(test_acc_each_exp))
        test_acc_mean_each_exp.append(np.mean(test_acc_each_exp))
        test_acc_std_each_exp.append(np.std(test_acc_each_exp))

    return {
        "max_val": val_acc_best_each_exp,
        "mean_val": val_acc_mean_each_exp, 
        "std_val": val_acc_std_each_exp,
        "max_test": test_acc_best_each_exp, 
        "mean_test": test_acc_mean_each_exp, 
        "std_test": test_acc_std_each_exp  
    }


def visualize_test_accuracy_result(
    acc_values: list,
    acc_std: list,
    attack_name: str,
    save_dir: str = "figures"
):
    """Visualize test accuracy with standard deviation as fill area."""

    # Prepare x-axis as index of runs
    x = list(range(len(acc_values)))

    # Convert input lists to numpy arrays
    acc_array = np.array(acc_values)
    std_array = np.array(acc_std)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Fill between (accuracy ± std)
    plt.fill_between(
        x,
        acc_array - std_array,
        acc_array + std_array,
        color="navajowhite",
        alpha=0.5,
        label="Accuracy ± Std"
    )

    # Line plot for accuracy
    plt.plot(
        x,
        acc_array,
        color="darkorange",
        linewidth=2,
        label="Test Accuracy"
    )

    # Scatter plot with square markers
    plt.scatter(
        x,
        acc_array,
        color="orangered",
        s=40,
        marker="s",
        label="Points"
    )

    # Axis labels and title
    plt.xlabel("Run Index")
    plt.ylabel("Test Accuracy")
    plt.title(f"{attack_name}")

    # Add legend and grid
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save the figure
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{attack_name}.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.clf()


if __name__ == "__main__":
    os.makedirs("figures/", exist_ok= True)
    best_arch_filename = "best_architecture_each_gen.p"
    args = parse_arguments()
    statistics = collect_acc_statistics(
        args.base_dir, best_arch_filename
    )
    visualize_test_accuracy_result(
        acc_values= statistics["max_test"], 
        acc_std= statistics["std_test"], 
        attack_name= attack_method[args.attack], 
    )
