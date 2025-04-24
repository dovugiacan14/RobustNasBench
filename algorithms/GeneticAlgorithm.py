"""
Inspired: https://github.com/msu-coinlab/pymoo
"""
import pickle
import numpy as np 
import matplotlib.pyplot as plt 
from helpers.utils import set_seed
from helpers.elastic_archive import ElitistArchive


class GeneticAlgorithm:
    def __init__(self, **kwargs):
        self.pop = None
        self.problem = None
        self.pop_size = None 
        self.sampling = None 
        self.crossover = None
        self.mutation = None
        self.survival = None

        self.seed = 0
        self.nGens = 0
        self.nEvals = 0
        self.path_results = None
        self.debug = False

        # SO-NAS 
        self.nGens_history = []
        self.best_F_history = []
        self.pop_history = []
        self.best_arch_history = []

        self.reference_point_search = [-np.inf, -np.inf]
        self.reference_point_evaluate = [-np.inf, -np.inf]
        self.E_Archive_search = None
        self.E_Archive_evaluate = None

        self.nEvals_history = []
        self.E_Archive_history_search = []
        self.E_Archive_history_evaluate = []
        self.IGD_history_search = []
        self.IGD_history_evaluate = []
       
    def set_hyperparameters(
        self, 
        pop_size= None, 
        sampling= None, 
        crossover= None, 
        mutation= None, 
        survival= None, 
        **kwargs
    ): 
        self.pop_size= pop_size 
        self.sampling= sampling 
        self.crossover= crossover
        self.mutation= mutation
        self.survival= survival
        self.debug= kwargs["debug"]

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
        self.E_Archive_evaluate = ElitistArchive(log_each_change=False)

        self.nEvals_history = []
        self.E_Archive_history_search = []
        self.E_Archive_history_evaluate = []
        self.IGD_history_search = []
        self.IGD_history_evaluate = []
    
    def _initialize(self):
        P = self.sampling.do(self.problem)
        for i in range(self.pop_size):
            F = self.evaluate(P[i].X)
            P[i].set("F", F)
        self.pop = P
    
    def set_up(self, problem, seed): 
        self.problem = problem
        self.seed = seed 
        set_seed(self.seed)

        self.sampling.nSamples = self.pop_size
    
    def mating(self, P): 
        O = self.crossover.do(self.problem, P, algorithm=self)
        O = self.mutation.do(self.problem, P, O, algorithm=self)
        return O
    
    def next(self, pop): 
        offspings = self.mating(pop)
        pool = pop.merge(offspings)
        pop = self.survival.do(pool, self.pop_size)
        self.pop = pop 

    def evaluate(self, X): 
        """call function *problem.evaluate* to evaluate the fitness values of solutions."""
        F = self.problem._evaluate(X)
        self.nEvals +=1 
        return F

    def do_each_gen(self, metric): 
        pop = {
            "X": self.pop.get("X"), 
            "hashKey": self.pop.get("hashKey"),
            "F": self.pop.get("F")
        }
        self.pop_history.append(pop)

        F = self.pop.get("F")
        best_arch_F = np.max(F)
        self.best_F_history.append(best_arch_F)

        idx_best_arch = F == best_arch_F
        best_arch_X_list = np.unique(self.pop.get("X")[idx_best_arch], axis= 0)
        best_arch_list = []
        if metric == "val_acc_clean": 
            metric = "val_accuracy"
        for arch_X in best_arch_X_list: 
            arch_info = {
                "X": arch_X, 
                "search_metric": self.problem._get_zero_cost_metric(arch_X, metric), 
                "robust_acc": self.problem._get_robustness_metric(arch_X), 
                "test_acc": self.problem._get_accuracy(arch_X, final= True), 
                "val_acc": self.problem._get_accuracy(arch_X)
            }
            best_arch_list.append(arch_info)
        self.best_arch_history.append(best_arch_list)
        self.nGens_history.append(self.nGens + 1) 
    
    def finalize(self, metric): 
        try: 
            save_dir = self.path_results 

            # summary and save result 
            gens = self.nGens_history
            best_f = np.array([gen[0]["val_acc"] for gen in self.best_arch_history])

            # plot 
            plt.figure(figsize=(10, 6))
            plt.xlim([0, gens[-1] + 2])

            # plot line and scatter 
            plt.plot(gens, best_f, c="blue", label="Best F")
            plt.scatter(gens, best_f, c="black", s=12, label="Best Architecture")

            # label, legend, title 
            plt.xlabel("#Gens")
            plt.ylabel("attack_name")
            plt.title(metric)

            plt.xticks(np.arange(0, gens[-1] + 30, 30))
            
            # save plot 
            plt.tight_layout()
            plt.savefig(f"{save_dir}/best_architecture_each_gen.png")
            plt.clf()

            # save data 
            with open(f"{save_dir}/best_architecture_each_gen.p", "wb") as f:
                pickle.dump([gens, self.best_arch_history], f)

            with open(f"{save_dir}/population_each_gen.p", "wb") as f:
                pickle.dump([gens, self.pop_history], f)

        except Exception: 
            raise ValueError(f"Not supported attack_name problem")

    
    def solve(self, problem, seed): 
        self.set_up(problem, seed)
        self._initialize()
        self.do_each_gen(metric= self.problem.zc_metric)
        while self.nEvals < self.problem.maxEvals: 
            self.nGens += 1 
            self.next(self.pop)
            self.do_each_gen(metric= self.problem.zc_metric)

        self.finalize(metric= self.problem.zc_metric)
