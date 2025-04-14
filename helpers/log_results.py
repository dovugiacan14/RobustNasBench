import pickle
import numpy as np
import matplotlib.pyplot as plt
from .utils import calculate_IGD_value


def compute_std_per_gen(pop_history):
    std_history = []
    for pop in pop_history:
        # Assume 'F' contains fitness values
        if isinstance(pop, dict) and "F" in pop:
            fitnesses = pop["F"]
        else:
            # fallback in case pop is a list of numbers
            fitnesses = pop

        std = np.std(fitnesses)
        std_history.append(std)

    return std_history


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


# def do_each_gen(type_of_problem, robust_type, metric, **kwargs):
#     algorithm = kwargs["algorithm"]
#     if type_of_problem == "single-objective":
#         pop = {
#             "X": algorithm.pop.get("X"),
#             "hashKey": algorithm.pop.get("hashKey"),
#             "F": algorithm.pop.get("F"),
#         }
#         algorithm.pop_history.append(pop)

#         F = algorithm.pop.get("F")
#         best_arch_F = np.max(F)
#         algorithm.best_F_history.append(best_arch_F)

#         idx_best_arch = F == best_arch_F
#         best_arch_X_list = np.unique(algorithm.pop.get("X")[idx_best_arch], axis=0)
#         best_arch_list = []

#         for arch_X in best_arch_X_list:
#             if robust_type == "val_acc":
#                 arch_info = {
#                     "X": arch_X,
#                     "search_metric": algorithm.problem.get_zero_cost_metric(arch_X, metric),
#                     "testing_accuracy": algorithm.problem.get_accuracy(arch_X, final=True),
#                     "validation_accuracy": algorithm.problem.get_accuracy(arch_X)
#                 }
#             else: 
#                 arch_info = {
#                     "X": arch_X,
#                     "testing_accuracy": algorithm.problem.get_robustness_metric(
#                         arch_X, robust_type, final=True
#                     ),
#                     "validation_accuracy": algorithm.problem.get_robustness_metric(
#                         arch_X, robust_type
#                     ),
#                 }
#             best_arch_list.append(arch_info)
#         algorithm.best_arch_history.append(best_arch_list)
#         algorithm.nGens_history.append(algorithm.nGens + 1)

#     elif type_of_problem == "multi-objective":
#         non_dominated_front = np.array(algorithm.E_Archive_search.F)
#         non_dominated_front = np.unique(non_dominated_front, axis=0)

#         # Update reference point (use for calculating the Hypervolume value)
#         algorithm.reference_point_search[0] = max(
#             algorithm.reference_point_search[0], max(non_dominated_front[:, 0])
#         )
#         algorithm.reference_point_search[1] = max(
#             algorithm.reference_point_search[1], max(non_dominated_front[:, 1])
#         )

#         IGD_value_search = calculate_IGD_value(
#             pareto_front=algorithm.problem.pareto_front_validation,
#             non_dominated_front=non_dominated_front,
#         )

#         algorithm.nEvals_history_each_gen.append(algorithm.nEvals)
#         algorithm.IGD_history_each_gen.append(IGD_value_search)

#     else:
#         raise ValueError(f"Not supported {type_of_problem} problem")


# def finalize(type_of_problem, metric, robustness_type, **kwargs):
#     algorithm = kwargs["algorithm"]
#     save_dir = algorithm.path_results

#     if type_of_problem == "single-objective":
#         gens = algorithm.nGens_history
#         best_f = np.array(
#             [gen[0]["validation_accuracy"] for gen in algorithm.best_arch_history]
#         )  # ensure float for plotting

#         plt.figure(figsize=(10, 6))
#         plt.xlim([0, gens[-1] + 2])

#         # Plot line and scatter
#         plt.plot(gens, best_f, c="blue", label="Best F")
#         plt.scatter(gens, best_f, c="black", s=12, label="Best Architecture")

#         # Label, legend, title
#         plt.xlabel("#Gens")
#         plt.ylabel(robustness_type)
#         plt.title(metric)
#         # plt.legend(loc="best")

#         plt.xticks(np.arange(0, gens[-1] + 30, 30))

#         # save plot
#         plt.tight_layout()
#         plt.savefig(f"{save_dir}/best_architecture_each_gen.png")
#         plt.clf()

#         # save data
#         with open(f"{save_dir}/best_architecture_each_gen.p", "wb") as f:
#             pickle.dump([gens, algorithm.best_arch_history], f)

#         with open(f"{save_dir}/population_each_gen.p", "wb") as f:
#             pickle.dump([gens, algorithm.pop_history], f)

#     elif type_of_problem == "multi-objective":
#         pickle.dump(
#             [algorithm.nEvals_history, algorithm.IGD_history_search],
#             open(f"{algorithm.path_results}/#Evals_and_IGD_search.p", "wb"),
#         )
#         pickle.dump(
#             [algorithm.nEvals_history, algorithm.IGD_history_evaluate],
#             open(f"{algorithm.path_results}/#Evals_and_IGD_evaluate.p", "wb"),
#         )
#         pickle.dump(
#             [algorithm.nEvals_history, algorithm.E_Archive_history_search],
#             open(f"{algorithm.path_results}/#Evals_and_Elitist_Archive_search.p", "wb"),
#         )
#         pickle.dump(
#             [algorithm.nEvals_history, algorithm.E_Archive_history_evaluate],
#             open(
#                 f"{algorithm.path_results}/#Evals_and_Elitist_Archive_evaluate.p", "wb"
#             ),
#         )
#         pickle.dump(
#             [algorithm.nEvals_history_each_gen, algorithm.IGD_history_each_gen],
#             open(f"{algorithm.path_results}/#Evals_and_IGD_each_gen.p", "wb"),
#         )

#         save_reference_point(
#             reference_point=algorithm.reference_point_search,
#             path_results=algorithm.path_results,
#             error="search",
#         )
#         save_reference_point(
#             reference_point=algorithm.reference_point_evaluate,
#             path_results=algorithm.path_results,
#             error="evaluate",
#         )

#         visualize_Elitist_Archive_and_Pareto_Front(
#             elitist_archive=algorithm.E_Archive_search.F,
#             pareto_front=algorithm.problem.pareto_front_validation,
#             objective_0=algorithm.problem.objectives_lst[0],
#             path_results=algorithm.path_results,
#             error="search",
#         )

#         visualize_Elitist_Archive_and_Pareto_Front(
#             elitist_archive=algorithm.E_Archive_history_evaluate[-1][-1],
#             pareto_front=algorithm.problem.pareto_front_testing,
#             objective_0=algorithm.problem.objectives_lst[0],
#             path_results=algorithm.path_results,
#             error="evaluate",
#         )

#         visualize_IGD_value_and_nEvals(
#             IGD_history=algorithm.IGD_history_search,
#             nEvals_history=algorithm.nEvals_history,
#             path_results=algorithm.path_results,
#             error="search",
#         )

#         visualize_IGD_value_and_nEvals(
#             IGD_history=algorithm.IGD_history_evaluate,
#             nEvals_history=algorithm.nEvals_history,
#             path_results=algorithm.path_results,
#             error="evaluate",
#         )
#     else:
#         raise ValueError(f"Not supported {type_of_problem} problem")
