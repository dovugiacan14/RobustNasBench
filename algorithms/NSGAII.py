"""
NSGA-II Implementation for Multi-Objective Neural Architecture Search
Inspired by: https://github.com/msu-coinlab/pymoo
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
    visualize_Elitist_Archive_and_Pareto_Front  
)

INF = 9999999

class NSGAII:
    """NSGA-II algorithm implementation for multi-objective optimization."""
    
    def __init__(self, objective, **kwargs):
        """Initialize NSGA-II algorithm.
        
        Args:
            objective (str): String in format "obj1/obj2" specifying the objectives
            **kwargs: Additional keyword arguments
        """
        self.name = "NSGA-II"
        self.pop_size = None
        self.problem = None
        self.sampling = None
        self.crossover = None
        self.mutation = None
        self.survival = None
        self.path_results = None
        self.path_data_zero_cost_method = None
        self.debug = False
        
        # Parse objectives
        first_obj, second_obj = objective.split("/")
        self.complexity_obj = first_obj    # First objective (e.g., FLOPs)
        self.robustness_obj = second_obj   # Second objective (e.g., robustness)

    def reset(self):
        """Reset the algorithm state."""
        self.pop = None
        self.seed = 0
        self.n_generations = 0
        self.n_evaluations = 0
        
        # Initialize archives
        self.E_Archive_search = ElitistArchive()
        self.E_Archive_search.algorithm = self
        self.E_Archive_evaluate = ElitistArchive(log_each_change=False)
        
        # Reference points
        self.search_ref_point = [-np.inf, -np.inf]
        self.eval_ref_point = [-np.inf, -np.inf]
        
        # History tracking
        self.generation_history = []
        self.best_fitness_history = []
        self.population_history = []
        self.best_architecture_history = []
        self.igd_history_per_gen = []
        self.evals_history_per_gen = []
        self.evals_history = []
        self.search_archive_history = []
        self.eval_archive_history = []
        self.igd_search_history = []
        self.igd_eval_history = []
        
        self.individual = Individual(rank=INF, crowding=-1)

    def set_hyperparameters(self, pop_size=None, sampling=None, crossover=None, 
                          mutation=None, survival=None, path_data_zero_cost_method=None,
                          debug=False, **kwargs):
        """Set algorithm hyperparameters.
        
        Args:
            pop_size (int): Population size
            sampling: Sampling operator
            crossover: Crossover operator
            mutation: Mutation operator
            survival: Survival selection operator
            path_data_zero_cost_method: Path to data for zero-cost method
            debug: Debug mode
            **kwargs: Additional parameters
        """
        self.pop_size = pop_size
        self.sampling = sampling
        self.crossover = crossover
        self.mutation = mutation
        self.survival = survival
        self.path_data_zero_cost_method = path_data_zero_cost_method
        self.debug = debug

    def set_up(self, problem, seed):
        self.problem = problem
        self.seed = seed
        set_seed(self.seed)

        self.sampling.nSamples = self.pop_size

    def initialize(self):
        """Initialize the population and evaluate initial solutions."""
        population = self.sampling.do(self.problem)
        for i in range(self.pop_size):
            fitness = self.evaluate(population[i].X)
            population[i].set("F", fitness)
            self.E_Archive_search.update(population[i])
        self.pop = population

    def evaluate(self, architecture):
        """Evaluate an architecture.
        
        Args:
            architecture: Architecture to evaluate
            
        Returns:
            Fitness values for the architecture
        """
        fitness = self.problem._evaluate(architecture, complex_metric=self.complexity_obj)
        self.n_evaluations += 1
        return fitness

    def log_elitist_archive(self):
        """Update both search and evaluation archives."""
        # Update search archive history
        search_front = np.array(self.E_Archive_search.F)
        search_front = np.unique(search_front, axis=0)
        self.evals_history.append(self.n_evaluations)
        self.search_archive_history.append([
            self.E_Archive_search.X.copy(),
            self.E_Archive_search.hashKey.copy(),
            self.E_Archive_search.F.copy(),
        ])

        # Update evaluation archive
        size = len(self.E_Archive_search.X)
        temp_pop = Population(size)
        for i in range(size):
            arch = self.E_Archive_search.X[i]
            hash_key = get_hashkey(arch)
            complexity_score = self.problem._get_complexity_metric(arch, self.complexity_obj)
            robustness_scores = self.problem._get_robustness_metric(arch)
            robustness_score = 1 - robustness_scores[self.robustness_obj]
            
            fitness = [complexity_score, robustness_score]
            temp_pop[i].set("X", arch)
            temp_pop[i].set("hashKey", hash_key)
            temp_pop[i].set("F", fitness)
            self.E_Archive_evaluate.update(temp_pop[i])

        # Update evaluation metrics
        eval_front = np.array(self.E_Archive_evaluate.F)
        eval_front = np.unique(eval_front, axis=0)
        
        self.eval_ref_point[0] = max(self.eval_ref_point[0], max(eval_front[:, 0]))
        self.eval_ref_point[1] = max(self.eval_ref_point[1], max(eval_front[:, 1]))

        igd_value = calculate_IGD_value(
            pareto_front=self.problem.pareto_front_testing,
            non_dominated_front=eval_front,
        )
        self.igd_eval_history.append(igd_value)
        self.eval_archive_history.append([
            self.E_Archive_evaluate.X.copy(),
            self.E_Archive_evaluate.hashKey.copy(),
            self.E_Archive_evaluate.F.copy(),
        ])

    def mating(self, population):
        """Perform crossover and mutation to create offspring.
        
        Args:
            population: Current population
            
        Returns:
            Offspring population
        """
        offspring = self.crossover.do(self.problem, population, algorithm=self)
        offspring = self.mutation.do(self.problem, population, offspring, algorithm=self)
        return offspring

    def next_generation(self, population):
        """Create next generation.
        
        Args:
            population: Current population
            
        Returns:
            New population
        """
        offspring = self.mating(population)
        combined_pop = population.merge(offspring)
        new_pop = self.survival.do(combined_pop, self.pop_size)
        self.pop = new_pop

    def update_generation_metrics(self):
        """Update metrics for the current generation."""
        # Get non-dominated architectures
        non_dominated_archs = self.E_Archive_search.X
        non_dominated_archs = np.unique(non_dominated_archs, axis=0)

        # Calculate non-dominated front
        non_dominated_front = []
        for arch in non_dominated_archs:
            complexity_score = self.problem._get_complexity_metric(arch, self.complexity_obj)
            robustness_scores = self.problem._get_robustness_metric(arch)
            robustness_score = 1 - robustness_scores[self.robustness_obj]
            non_dominated_front.append([complexity_score, robustness_score])
        
        non_dominated_front = np.array(non_dominated_front)
        non_dominated_front = np.unique(non_dominated_front, axis=0)

        # Update reference points
        self.search_ref_point[0] = max(self.search_ref_point[0], max(non_dominated_front[:, 0]))
        self.search_ref_point[1] = max(self.search_ref_point[1], max(non_dominated_front[:, 1]))

        # Calculate IGD
        igd_value = calculate_IGD_value(
            pareto_front=self.problem.pareto_front_testing,
            non_dominated_front=non_dominated_front
        )

        self.evals_history_per_gen.append(self.n_evaluations)
        self.igd_history_per_gen.append(igd_value)

    def finalize(self):
        """Finalize the optimization process and save results.
        
        Args:
            metric: Metric to use for final evaluation
        """
        # Save evaluation history
        pickle.dump(
            [self.evals_history, self.igd_eval_history],
            open(f"{self.path_results}/#Evals_and_IGD_evaluate.p", "wb"),
        )
        pickle.dump(
            [self.evals_history, self.search_archive_history],
            open(f"{self.path_results}/#Evals_and_Elitist_Archive_search.p", "wb"),
        )
        pickle.dump(
            [self.evals_history, self.eval_archive_history],
            open(f"{self.path_results}/#Evals_and_Elitist_Archive_evaluate.p", "wb"),
        )
        pickle.dump(
            [self.evals_history_per_gen, self.igd_history_per_gen],
            open(f"{self.path_results}/#Evals_and_IGD_each_gen.p", "wb"),
        )

        # Save reference points
        save_reference_point(
            reference_point=self.search_ref_point,
            path_results=self.path_results,
            error="search"
        )
        save_reference_point(
            reference_point=self.eval_ref_point,
            path_results=self.path_results,
            error="evaluate"
        )

        # Visualize results
        visualize_Elitist_Archive_and_Pareto_Front(
            elitist_archive=self.E_Archive_evaluate.F,
            pareto_front=self.problem.pareto_front_testing,
            objective_0=self.complexity_obj,
            path_results=self.path_results,
            error="evaluate"
        )

    def solve(self, problem, seed):
        """Solve the optimization problem.
        
        Args:
            problem: Problem to solve
            seed: Random seed
        """
        self.set_up(problem, seed)
        self.initialize()
        self.update_generation_metrics()
        
        while self.n_evaluations < self.problem.maxEvals:
            self.n_generations += 1
            self.next_generation(self.pop)
            self.update_generation_metrics()
            
        self.finalize()
