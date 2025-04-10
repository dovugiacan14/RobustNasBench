from src.population import Population
from helpers.utils import get_hashkey


class RandomSampling:
    def __init__(self, nSamples=0):
        self.nSamples = nSamples

    def do(self, problem, **kwargs):
        problem_name = problem.name
        P = Population(self.nSamples)
        n = 0

        P_hashKey = []
        while n < self.nSamples:
            X = problem.sample_a_compact_architecture()
            if problem.isValid(X):
                hashKey = get_hashkey(X, problem_name)
                if hashKey not in P_hashKey:
                    P[n].set('X', X)
                    P[n].set('hashKey', hashKey)
                    n += 1
        return P