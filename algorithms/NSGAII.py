"""
Inspired: https://github.com/msu-coinlab/pymoo
"""

import pickle
import numpy as np
from helpers.utils import set_seed
from src.population import Population
from src.individual import Individual
from helpers.elastic_archive import ElitistArchive
from helpers.utils import (
    get_hashkey,
    calculate_IGD_value,
    save_reference_point, 
    save_Non_dominated_Front_and_Elitist_Archive, 
    visualize_Elitist_Archive,
    visualize_IGD_value_and_nEvals, 
    visualize_Elitist_Archive_and_Pareto_Front  
)

INF = 9999999


class NSGAII:
    def __init__(self, objective, **kwargs):
        self.name = "NSGA-II"
        self.pop_size = None
        self.problem = None
        self.sampling = None
        self.crossover = None
        self.mutation = None
        self.survival = None
        self.E_Archive_search = None
        self.E_Archive_evaluate = None
        self.individual = Individual(rank=INF, crowding=-1)
        self.__get_objective(objective)

    def reset(self):
        self.pop = None

        self.seed = 0
        self.nGens = 0
        self.nEvals = 0

        self.path_results = None

        # SONAS problems
        self.nGens_history = []
        self.best_F_history = []
        self.pop_history = []
        self.best_arch_history = []

        # MONAS problems
        self.IGD_history_each_gen = []
        self.nEvals_history_each_gen = []

        self.reference_point_search = [-np.inf, -np.inf]
        self.reference_point_evaluate = [-np.inf, -np.inf]

        self.E_Archive_search = ElitistArchive()
        self.E_Archive_search.algorithm = self
        self.E_Archive_evaluate = ElitistArchive(log_each_change=False)

        self.nEvals_history = []
        self.E_Archive_history_search = []
        self.E_Archive_history_evaluate = []
        self.IGD_history_search = []
        self.IGD_history_evaluate = []
        self.individual = Individual(rank=INF, crowding=-1)

    def __get_objective(self, objective):
        first_objective = objective.split("/")[0]
        second_objective = objective.split("/")[1]
        self.obj1 = first_objective
        self.obj2 = second_objective 

    def set_hyperparameters(
        self,
        pop_size=None,
        sampling=None,
        crossover=None,
        mutation=None,
        survival=None,
        **kwargs,
    ):
        self.pop_size = pop_size
        self.sampling = sampling
        self.crossover = crossover
        self.mutation = mutation
        self.survival = survival
        self.debug = kwargs["debug"]

    def set_up(self, problem, seed):
        self.problem = problem
        self.seed = seed
        set_seed(self.seed)

        self.sampling.nSamples = self.pop_size

    def initialize(self):
        P = self.sampling.do(self.problem)
        for i in range(self.pop_size):
            F = self.evaluate(P[i].X)
            P[i].set("F", F)
            self.E_Archive_search.update(P[i])
        self.pop = P

    def evaluate(self, X):
        F = self.problem._evaluate(X, complex_metric= self.obj1)
        self.nEvals += 1
        return F

    def log_elitist_archive(self):  # For solving MONAS problems
        non_dominated_front = np.array(self.E_Archive_search.F)
        non_dominated_front = np.unique(non_dominated_front, axis=0)

        self.nEvals_history.append(self.nEvals)

        # IGD_value_search = calculate_IGD_value(
        #     pareto_front=self.problem.pareto_front_validation,
        #     non_dominated_front=non_dominated_front,
        # )

        # self.IGD_history_search.append(IGD_value_search)
        self.E_Archive_history_search.append(
            [
                self.E_Archive_search.X.copy(),
                self.E_Archive_search.hashKey.copy(),
                self.E_Archive_search.F.copy(),
            ]
        )

        size = len(self.E_Archive_search.X)
        tmp_pop = Population(size)
        for i in range(size):
            X = self.E_Archive_search.X[i]
            hashKey = get_hashkey(X)
            score_obj1 = self.problem._get_complexity_metric(X, self.obj1)
            robustness_metric = self.problem._get_robustness_metric(X)
            score_obj2 = 1 - robustness_metric[self.obj2]
            # F = [
            #     self.problem._get_complexity_metric(X, self.obj1),
            #     1 - self.problem._get_accuracy(X, final=True),
            # ]
            F = [score_obj1, score_obj2]
            tmp_pop[i].set("X", X)
            tmp_pop[i].set("hashKey", hashKey)
            tmp_pop[i].set("F", F)
            self.E_Archive_evaluate.update(tmp_pop[i])

        non_dominated_front = np.array(self.E_Archive_evaluate.F)
        non_dominated_front = np.unique(non_dominated_front, axis=0)

        self.reference_point_evaluate[0] = max(
            self.reference_point_evaluate[0], max(non_dominated_front[:, 0])
        )
        self.reference_point_evaluate[1] = max(
            self.reference_point_evaluate[1], max(non_dominated_front[:, 1])
        )

        IGD_value_evaluate = calculate_IGD_value(
            pareto_front=self.problem.pareto_front_testing,
            non_dominated_front=non_dominated_front,
        )
        self.IGD_history_evaluate.append(IGD_value_evaluate)
        self.E_Archive_history_evaluate.append(
            [
                self.E_Archive_evaluate.X.copy(),
                self.E_Archive_evaluate.hashKey.copy(),
                self.E_Archive_evaluate.F.copy(),
            ]
        )
        self.E_Archive_evaluate = ElitistArchive(log_each_change=False)
    
    def mating(self, P): 
        O = self.crossover.do(self.problem, P, algorithm=self)
        O = self.mutation.do(self.problem, P, O, algorithm=self)
        return O

    def next(self, pop): 
        offspings = self.mating(pop)
        pool = pop.merge(offspings)
        pop = self.survival.do(pool, self.pop_size)
        self.pop = pop 

    def do_each_gen(self):
        non_dominated_front = np.array(self.E_Archive_search.F)
        non_dominated_front = np.unique(non_dominated_front, axis=0)

        # update reference point (use for calculating the Hypervolume value)
        self.reference_point_search[0] = max(
            self.reference_point_search[0], max(non_dominated_front[:, 0])
        )
        self.reference_point_search[1] = max(
            self.reference_point_search[1], max(non_dominated_front[:, 1])
        )

        IGD_value_search = calculate_IGD_value(
            pareto_front=self.problem.pareto_front_testing,
            non_dominated_front=non_dominated_front,
        )

        self.nEvals_history_each_gen.append(self.nEvals)
        self.IGD_history_each_gen.append(IGD_value_search)

    def finalize(self, metric):
        # save in file 
        pickle.dump(
            [self.nEvals_history, self.IGD_history_search],
            open(f"{self.path_results}/#Evals_and_IGD_search.p", "wb"),
        )
        pickle.dump(
            [self.nEvals_history, self.IGD_history_evaluate],
            open(f"{self.path_results}/#Evals_and_IGD_evaluate.p", "wb"),
        )
        pickle.dump(
            [self.nEvals_history, self.E_Archive_history_search],
            open(f"{self.path_results}/#Evals_and_Elitist_Archive_search.p", "wb"),
        )
        pickle.dump(
            [self.nEvals_history, self.E_Archive_history_evaluate],
            open(f"{self.path_results}/#Evals_and_Elitist_Archive_evaluate.p", "wb"),
        )
        pickle.dump(
            [self.nEvals_history_each_gen, self.IGD_history_each_gen],
            open(f"{self.path_results}/#Evals_and_IGD_each_gen.p", "wb"),
        )

        # save points 
        save_reference_point(
            reference_point= self.reference_point_search, 
            path_results= self.path_results, 
            error= "search"
        )
        save_reference_point(
            reference_point= self.reference_point_evaluate, 
            path_results= self.path_results, 
            error= "evaluate"
        )

        # visualize 
        visualize_Elitist_Archive_and_Pareto_Front(
            elitist_archive= self.E_Archive_search.F, 
            pareto_front= self.problem.pareto_front_validation, 
            objective_0= self.problem.objectives_lst[0], 
            path_results= self.path_results, 
            error= "search"
        )
        visualize_Elitist_Archive_and_Pareto_Front(
            elitist_archive= self.E_Archive_history_evaluate[-1][-1], 
            pareto_front= self.problem.pareto_front_testing,
            objective_0= self.problem.objectives_lst[0],
            path_results= self.path_results, 
            error= "evaluate"
        )
        visualize_IGD_value_and_nEvals(
            IGD_history= self.IGD_history_search, 
            nEvals_history= self.nEvals_history, 
            path_results= self.path_results, 
            error= "search"
        )
        visualize_IGD_value_and_nEvals(
            IGD_history= self.IGD_history_search, 
            nEvals_history= self.nEvals_history, 
            path_results= self.path_results, 
            error= "evaluate"
        )

    def solve(self, problem, seed):
        self.set_up(problem, seed)
        self.initialize()
        self.do_each_gen()
        while self.nEvals < self.problem.maxEvals:
            self.nGens += 1
            self.next(self.pop)
            self.do_each_gen()
        self.finalize(metric=self.problem.zc_metric)
