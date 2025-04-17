import os
import torch
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt


def create_directory(base_path: str, sub_path: str):
    """
    Create a directory (including nested subdirectories if needed) and return the full path.

    Args:
        base_path (str): The base directory path.
        sub_path (str, optional): A subdirectory path to append under base_path.

    Returns:
        str: The full path to the created directory.
    """
    full_path = os.path.join(base_path, sub_path) if sub_path else base_path
    os.makedirs(full_path, exist_ok=True)
    return full_path


def check_valid(hash_key, **kwargs):
    """Check if the current solution already exists on the set of checklists."""
    return np.all([hash_key not in kwargs[L] for L in kwargs])


def get_hashkey(arch):
    """
    This function is used to get the hash key of architecture. The hash key is used to avoid the existence of duplication in the population.\n
    - *Output*: The hash key of architecture.
    """
    return "".join(map(str, arch))


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)


def find_the_better(x, y):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    sub_ = x - y
    x_better = np.all(sub_ <= 0)
    y_better = np.all(sub_ >= 0)
    if x_better == y_better:  # True - True
        return -1
    if y_better:  # False - True
        return 1
    return 0  # True - False


"""=========calculate IGD value==========="""


def calculate_Euclidean_distance(x1, x2):
    euclidean_distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return euclidean_distance


def calculate_IGD_value(pareto_front, non_dominated_front):
    non_dominated_front = np.unique(non_dominated_front, axis=0)
    d = 0
    for s in pareto_front:
        d += min([calculate_Euclidean_distance(s, s_) for s_ in non_dominated_front])
    return round(d / len(pareto_front), 6)


"""============== log result NSGAII ==============="""


def save_reference_point(reference_point, path_results, error="None"):
    pickle.dump(
        reference_point, open(f"{path_results}/reference_point({error}).p", "wb")
    )

def save_Non_dominated_Front_and_Elitist_Archive(
    non_dominated_front, n_evals, elitist_archive, n_gens, path_results
):
    """
    - This function is used to save the non-dominated front and Elitist Archive at the end of each generation.
    """
    pickle.dump(
        [non_dominated_front, n_evals],
        open(f"{path_results}/non_dominated_front/gen_{n_gens}.p", "wb"),
    )
    pickle.dump(
        elitist_archive, open(f"{path_results}/elitist_archive/gen_{n_gens}.p", "wb")
    )


def visualize_IGD_value_and_nEvals(
    nEvals_history, IGD_history, path_results, error="search"
):
    """
    - This function is used to visualize 'IGD_values' and 'nEvals' at the end of the search.
    """
    plt.xscale("log")
    plt.xlabel("#Evals")
    plt.ylabel("IGD value")
    plt.grid()
    plt.plot(nEvals_history, IGD_history)
    plt.savefig(f"{path_results}/#Evals-IGD({error})")
    plt.clf()


def visualize_Elitist_Archive_and_Pareto_Front(
    elitist_archive, pareto_front, objective_0, path_results, error="testing"
):
    non_dominated_front = np.array(elitist_archive)
    non_dominated_front = np.unique(non_dominated_front, axis=0)

    plt.scatter(
        pareto_front[:, 0],
        pareto_front[:, 1],
        facecolors="none",
        edgecolors="b",
        s=40,
        label=f"Pareto-optimal Front",
    )
    plt.scatter(
        non_dominated_front[:, 0],
        non_dominated_front[:, 1],
        c="red",
        s=15,
        label=f"Non-dominated Front",
    )

    plt.xlabel(objective_0 + "(normalize)")
    plt.ylabel("Error")

    plt.legend()
    plt.grid()
    plt.savefig(f"{path_results}/non_dominated_front({error})")
    plt.clf()


def visualize_Elitist_Archive(elitist_archive, objective_0, path_results):
    non_dominated_front = np.array(elitist_archive)
    non_dominated_front = np.unique(non_dominated_front, axis=0)

    plt.scatter(
        non_dominated_front[:, 0],
        non_dominated_front[:, 1],
        facecolors="none",
        edgecolors="b",
        s=40,
        label=f"Non-dominated Front",
    )

    plt.xlabel(objective_0 + "(normalize)")
    plt.ylabel("Error")

    plt.legend()
    plt.grid()
    plt.savefig(f"{path_results}/non_dominated_front")
    plt.clf()
