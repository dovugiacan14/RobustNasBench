import pickle
import numpy as np
import matplotlib.pyplot as plt
from .utils import calculate_IGD_value


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