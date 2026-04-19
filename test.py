OP_NAMES_NB201 = [
    "skip_connect",
    "none",
    "nor_conv_3x3",
    "nor_conv_1x1",
    "avg_pool_3x3",
]

AVAILABLE_OPERATIONS = [
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3",
]


import os
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def decode_architecture_from_nasbench(encoded_architecture):
    """Decode architecture from NAS-Bench format using OP_NAMES_NB201"""
    # Handle string input
    if isinstance(encoded_architecture, str):
        # Remove parentheses and split by comma
        encoded_architecture = tuple(
            map(int, encoded_architecture.strip("()").split(","))
        )

    # Now process as tuple
    ops = [OP_NAMES_NB201[idx] for idx in encoded_architecture]
    return "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(*ops)


def decode_architecture_from_zc(encoded_architecture):
    """Decode architecture from zc_nasbench format using AVAILABLE_OPERATIONS"""
    # Handle string input
    if isinstance(encoded_architecture, str):
        # Remove parentheses and split by comma
        encoded_architecture = tuple(
            map(int, encoded_architecture.strip("()").split(","))
        )

    # Now process as tuple
    ops = [AVAILABLE_OPERATIONS[idx] for idx in encoded_architecture]
    return "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(*ops)


def get_data(encoded_architecture, robust_info_path, zc_metrics_path, data_path):
    # Convert architecture to different formats
    arch_str = decode_architecture_from_nasbench(encoded_architecture)  # Format: "|nor_conv_3x3~0|+|..."
    arch_tuple = str(encoded_architecture)  # Format: "(4, 2, 1, 2, 1, 1)"
    arch_simple = ''.join(map(str, encoded_architecture))  # Format: "421211"

    # 1. Get vac_acc_12 from pickle file
    with open(data_path, 'rb') as p:
        data = pickle.load(p)
    vac_acc_12 = data['12'][arch_simple]['val_acc'][11]  # Index 11 = epoch 12

    # 2. Get SynFlow from zc_nasbench201.json
    with open(zc_metrics_path, 'r') as f:
        zc_metrics = json.load(f)
    synflow = zc_metrics['cifar100'][arch_tuple]['synflow']['score']

    # 3. Get Clean accuracy from robust_info
    with open(robust_info_path, 'r') as f:
        robust_info = json.load(f)
    clean = robust_info[arch_str]['val_acc']['threeseed']

    return vac_acc_12, synflow, clean


def get_all_architectures_data(robust_info_path, zc_metrics_path, data_path):
    """Get data for all 15,625 architectures"""
    # Load all data files once
    with open(data_path, 'rb') as p:
        data = pickle.load(p)

    with open(zc_metrics_path, 'r') as f:
        zc_metrics = json.load(f)

    with open(robust_info_path, 'r') as f:
        robust_info = json.load(f)

    results = {
        'vac_acc_12': [],
        'synflow': [],
        'clean': [],
        'architectures': []
    }

    # Iterate through all architectures in zc metrics
    for arch_tuple_str in tqdm(zc_metrics['cifar100'].keys(), desc="Processing architectures"):
        try:
            # Parse architecture tuple
            arch_tuple = tuple(map(int, arch_tuple_str.strip("()").split(",")))
            arch_simple = ''.join(map(str, arch_tuple))
            arch_str = decode_architecture_from_nasbench(arch_tuple)

            # Get data
            vac_acc_12 = data['12'][arch_simple]['val_acc'][11]
            synflow = zc_metrics['cifar100'][arch_tuple_str]['synflow']['score']
            clean = robust_info[arch_str]['val_acc']['threeseed']

            results['vac_acc_12'].append(vac_acc_12 * 100)  # Convert to percentage
            results['synflow'].append(synflow)
            results['clean'].append(clean * 100)  # Convert to percentage
            results['architectures'].append(arch_tuple)

        except (KeyError, ValueError) as e:
            # Skip architectures not found in all datasets
            continue

    return results


def plot_nasbench_visualization(results, dataset_name='CIFAR-100'):
    """Create visualization similar to the reference image"""
    vac_acc_12 = np.array(results['vac_acc_12'])
    synflow = np.array(results['synflow'])
    clean = np.array(results['clean'])

    # Find indices of maximum values
    max_val_acc_idx = np.argmax(vac_acc_12)
    max_synflow_idx = np.argmax(synflow)

    # DEBUG: Check if they are the same
    print(f"\n=== DEBUG ===")
    print(f"max_val_acc_idx: {max_val_acc_idx}")
    print(f"max_synflow_idx: {max_synflow_idx}")
    print(f"Are they the same? {max_val_acc_idx == max_synflow_idx}")
    print(f"Max Val-Acc: {vac_acc_12[max_val_acc_idx]:.2f}%, Clean: {clean[max_val_acc_idx]:.2f}%")
    print(f"Max SynFlow: {synflow[max_synflow_idx]:.2f}, Clean: {clean[max_synflow_idx]:.2f}%")

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Validation accuracy at epoch 12 vs Clean accuracy
    ax1.scatter(vac_acc_12, clean, alpha=0.3, s=20, color='blue', label='Networks', zorder=1)
    ax1.scatter(vac_acc_12[max_val_acc_idx], clean[max_val_acc_idx],
               alpha=1.0, s=200, color='red', marker='*', edgecolors='black', linewidth=2, zorder=10,
               label=f'Network with highest Val-Acc-12 (Clean-Acc = {clean[max_val_acc_idx]:.2f}%)')
    ax1.set_xlabel('Validation accuracy at epoch 12-th (%)', fontsize=12)
    ax1.set_ylabel('Accuracy on Clean Data (Clean-Acc) (%)', fontsize=12)
    ax1.set_title('Validation Accuracy vs Clean Accuracy', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=10)

    # Set xlim based on actual data range with padding
    val_acc_min, val_acc_max = np.min(vac_acc_12), np.max(vac_acc_12)
    padding = (val_acc_max - val_acc_min) * 0.05  # 5% padding
    ax1.set_xlim(val_acc_min - padding, val_acc_max + padding)
    ax1.set_ylim(0, 80)

    # Right plot: SynFlow score vs Clean accuracy
    ax2.scatter(synflow, clean, alpha=0.3, s=20, color='blue', label='Networks', zorder=1)

    # Plot max synflow point with multiple visual enhancements
    ax2.scatter(synflow[max_synflow_idx], clean[max_synflow_idx],
               alpha=1.0, s=300, color='orange', marker='*', edgecolors='red', linewidth=3, zorder=10,
               label=f'Network with highest SynFlow (Clean-Acc = {clean[max_synflow_idx]:.2f}%)')

    # Add annotation text
    ax2.annotate(f'Max SynFlow\n({synflow[max_synflow_idx]:.1f}, {clean[max_synflow_idx]:.1f}%)',
                xy=(synflow[max_synflow_idx], clean[max_synflow_idx]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2))

    ax2.set_xlabel('Synaptic Flow score', fontsize=12)
    ax2.set_ylabel('Accuracy on Clean Data (Clean-Acc) (%)', fontsize=12)
    ax2.set_title('SynFlow vs Clean Accuracy', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=10)

    # Set xlim based on actual data range with padding
    synflow_min, synflow_max = np.min(synflow), np.max(synflow)
    padding = (synflow_max - synflow_min) * 0.05  # 5% padding
    ax2.set_xlim(synflow_min - padding, synflow_max + padding)
    ax2.set_ylim(0, 80)

    # Main title
    fig.suptitle(dataset_name, fontsize=16, y=1.02)

    # Caption
    caption = (f'Fig. 4. Visualization of all {len(results["architectures"])} networks in NAS-Bench-201 '
              f'with respect to (Left) validation accuracy at epoch 12th vs. accuracy on clean data '
              f'after adversarial training, and (Right) SynFlow vs. accuracy on clean data after '
              f'adversarial training.')
    fig.text(0.5, -0.05, caption, ha='center', fontsize=10, wrap=True)

    plt.tight_layout()
    return fig



if __name__ == "__main__":
    robust_info_path = 'config/cifar10.json'
    zc_metrics_path = 'config/zc_nasbench201.json'
    data_path = "data/NASBench201/[CIFAR-10]_data.p"

    # Get data for all architectures
    print("Collecting data for all architectures...")
    results = get_all_architectures_data(robust_info_path, zc_metrics_path, data_path)

    print(f"\nTotal architectures processed: {len(results['architectures'])}")
    print(f"Validation accuracy at epoch 12 - Min: {min(results['vac_acc_12']):.2f}%, Max: {max(results['vac_acc_12']):.2f}%")
    print(f"SynFlow score - Min: {min(results['synflow']):.2f}, Max: {max(results['synflow']):.2f}")
    print(f"Clean accuracy - Min: {min(results['clean']):.2f}%, Max: {max(results['clean']):.2f}%")

    # Create visualization
    print("\nCreating visualization...")
    fig = plot_nasbench_visualization(results, dataset_name='CIFAR-10')

    plt.show()
