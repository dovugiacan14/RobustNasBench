import os
import time
import argparse
from sys import platform
from datetime import datetime
from helpers.utils import create_directory
from src.factory import get_problem, get_algorithm
from constant import population_size_dict, zero_cost_metrics

from operators.crossover import PointCrossover
from operators.mutation import BitStringMutation
from operators.sampling.random_sampling import RandomSampling
from algorithms.GeneticAlgorithm import GeneticAlgorithm
from operators.selection import TournamentSelection, RankAndCrowdingSurvival


if platform == "linux" or platform == "linux2":
    root_project = "/".join(os.path.abspath(__file__).split("/")[:-1])
elif platform == "win32" or platform == "win64":
    root_project = "\\".join(os.path.abspath(__file__).split("\\")[:-1])
else:
    raise ValueError()


def parse_argument():
    parser = argparse.ArgumentParser()

    # PROBLEM
    parser.add_argument(
        "--problem_name",
        "-problem",
        type=str,
        default="SO-NAS201-1",
        help="the problem name",
        choices=[
            "SO-NAS101",
            "SO-NAS201-1",
            "SO-NAS201-2",
            "SO-NAS201-3",
            "MO-NAS101",
            "MO-NAS201-1",
            "MO-NAS201-2",
            "MO-NAS201-3",
        ],
    )

    # EVOLUTIONARY ALGORITHM
    parser.add_argument(
        "--algorithm_name",
        type=str,
        default="GA",
        help="the algorithm name",
        choices=["GA", "NSGA-II"],
    )

    # ENVIRONMENT
    parser.add_argument(
        "--path_results",
        "-ps",
        type=str,
        default="results",
        help="path for saving results",
    )
    parser.add_argument(
        "--n_runs", type=int, default=2, help="number of experiment runs"
    )
    parser.add_argument(
        "--metric", type=int, default=9, help="zero-cost metric to search", choices=range(0, 11)
    )
    parser.add_argument(
        "--attack", type=int, default=5, help="type of attack", choices=range(0, 6)
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--debug", type=int, default=0, help="debug mode (0 or 1)")

    return parser.parse_args()


def main(args):
    # create folder to save result
    base_results_path = create_directory(args.path_results or root_project, zero_cost_metrics[args.metric])
    timestamp = datetime.now().strftime(f"{args.problem_name}_{args.algorithm_name}_%d%m%H%M%S")
    root_path = create_directory(base_results_path, timestamp)
    PATH_DATA = os.path.join(root_project, "data")
   
    # initialize problem
    problem = get_problem(
        problem_name=args.problem_name, 
        metric=args.metric, 
        robustness= args.attack, 
        path_data=PATH_DATA
    )
    problem._set_up()

    # initialize statistic
    pop_size = population_size_dict[args.problem_name]
    n_runs = args.n_runs
    init_seed = args.seed
    random_seeds_list = [init_seed + run * 100 for run in range(n_runs)]

    sampling = RandomSampling()
    crossover = PointCrossover("2X")
    mutation = BitStringMutation()

    # initialize algorithm and selection method
    # algorithm = get_algorithm(algorithm_name=args.algorithm_name)
    if args.algorithm_name == "GA":
        algorithm = GeneticAlgorithm()
        survival = TournamentSelection(k=4)
    elif args.algorithm_name == "NSGA-II":
        survival = RankAndCrowdingSurvival()
    else:
        raise ValueError()

    algorithm.set_hyperparameters(
        pop_size=pop_size,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        survival=survival,
        path_data_zero_cost_method=PATH_DATA,
        debug=bool(args.debug),
    )

    # logging and solve problem
    with open(f"{root_path}/logging.txt", "w") as f:
        f.write(f"******* PROBLEM *******\n")
        f.write(f"- Benchmark: NasBench201\n")
        f.write(f"- Dataset: {problem.dataset}\n")
        f.write(f"- Maximum number of evaluations: {problem.maxEvals}\n")
        f.write(f"- List of objectives: {problem.objectives_lst}\n\n")

        f.write(f"******* ALGORITHM *******\n")
        f.write(f"- Algorithm: {args.algorithm_name}\n")
        f.write(f"- Population size: {algorithm.pop_size}\n")
        f.write(f"- Crossover method: {algorithm.crossover.method}\n")
        f.write(f"- Mutation method: Bit-string\n")
        f.write(f"- Selection method: {algorithm.survival.name}\n\n")

        f.write(f"******* ENVIRONMENT *******\n")
        f.write(f"- Number of running experiments: {n_runs}\n")
        f.write(f"- Random seed each run: {random_seeds_list}\n")
        f.write(f"- Path for saving results: {root_path}\n")
        f.write(f"- Debug: {algorithm.debug}\n\n")

    executed_time_list = []
    for run_i in range(n_runs):
        print(f"---- Run {run_i + 1}/{n_runs} ----")
        random_seed = random_seeds_list[run_i]
        path_results = root_path + "/" + f"{run_i}"
        os.mkdir(path_results)
        s = time.time()

        algorithm.reset()
        algorithm.path_results = path_results
        algorithm.solve(problem, random_seed)
        executed_time = time.time() - s
        executed_time_list.append(executed_time)
        print("This run take", executed_time_list[-1], "seconds")


if __name__ == "__main__":
    args = parse_argument()
    main(args)
