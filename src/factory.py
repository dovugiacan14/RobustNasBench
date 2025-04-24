from algorithms import NSGAII 
from algorithms import GeneticAlgorithm 
from problems.nasbench201 import NASBench201
from constant import problem_configuration, search_metrics 


def get_problem(problem_name, metric, **kwargs):
    try: 
        config = problem_configuration[problem_name]
        zc_metric = search_metrics[metric]
        return NASBench201(
            maxEvals=config['maxEvals'], 
            dataset=config['dataset'], 
            zc_metric= zc_metric, 
            type_of_problem=config['type_of_problem'], 
            **kwargs
        )
    except: 
        raise ValueError(f'Not supporting this problem - {problem_name}.')

def get_algorithm(algorithm_name, **kwargs):
    if algorithm_name == 'GA':
        return GeneticAlgorithm()
    elif algorithm_name == 'NSGA-II':
        return NSGAII()
    else:
        raise ValueError(f'Not supporting this algorithm - {algorithm_name}.')