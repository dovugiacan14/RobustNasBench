from algorithms import NSGAII 
from algorithms import GeneticAlgorithm 
from problems.nasbench201 import NASBench201
from constant import problem_configuration, zero_cost_metrics, attack_method 


def get_problem(problem_name, metric, robustness, **kwargs):
    try: 
        config = problem_configuration[problem_name]
        zc_metric = zero_cost_metrics[metric]
        robustness_type = attack_method[robustness]
        return NASBench201(
            maxEvals=config['maxEvals'], 
            dataset=config['dataset'], 
            zc_metric= zc_metric, 
            robust_type= robustness_type,
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