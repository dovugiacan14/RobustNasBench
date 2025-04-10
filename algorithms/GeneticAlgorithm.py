"""
Inspired: https://github.com/msu-coinlab/pymoo
"""

from algorithms import Algorithm


class GeneticAlgorithm(Algorithm):
    def __init__(self, **kwargs):
        super().__init__(name="GA", **kwargs)

    def _initialize(self):
        P = self.sampling.do(self.problem)
        for i in range(self.pop_size):
            F = self.evaluate(P[i].X)
            P[i].set("F", F)
        self.pop = P
