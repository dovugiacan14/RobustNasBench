from algorithms import NSGAII 
from algorithms import GeneticAlgorithm 
from problems.nasbench101 import NASBench101 
from problems.nasbench201 import NASBench201
from constant import problem_configuration, zero_cost_metrics, attack_method 


def get_problem(problem_name, metric, robustness, **kwargs):
    config = problem_configuration[problem_name]
    fitness_metric = zero_cost_metrics[metric]
    robustness_type = attack_method[robustness]
    if 'NAS101' in problem_name:
        return NASBench101(maxEvals=config['maxEvals'], dataset=config['dataset'], type_of_problem=config['type_of_problem'], **kwargs)
    elif 'NAS201' in problem_name:
        return NASBench201(
            maxEvals=config['maxEvals'], 
            dataset=config['dataset'], 
            fitness_metric= fitness_metric, 
            robust_type= robustness_type,
            type_of_problem=config['type_of_problem'], 
            **kwargs
        )
    else:
        raise ValueError(f'Not supporting this problem - {problem_name}.')

def get_algorithm(algorithm_name, **kwargs):
    if algorithm_name == 'GA':
        return GeneticAlgorithm()
    elif algorithm_name == 'NSGA-II':
        return NSGAII()
    else:
        raise ValueError(f'Not supporting this algorithm - {algorithm_name}.')