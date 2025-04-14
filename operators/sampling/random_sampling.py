from src.population import Population
from helpers.utils import get_hashkey


class RandomSampling:
    def __init__(self, nSamples=0):
        self.nSamples = nSamples

    def do(self, problem, **kwargs):
        P = Population(self.nSamples)
        n = 0

        P_hashKey = []
        while n < self.nSamples:
            X = problem._get_a_compact_architecture()
            hashKey = get_hashkey(X)
            if hashKey not in P_hashKey:
                P[n].set('X', X)
                P[n].set('hashKey', hashKey)
                n += 1
        return P