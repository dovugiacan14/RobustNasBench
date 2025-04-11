import os
import json
import pickle
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from problems.NAS_problem import Problem
from constant import AVAILABLE_OPERATIONS

load_dotenv()

zero_cost_file = Path(os.environ["ZERO_COST_NASBENCH201"])
if not zero_cost_file.exists():
    raise FileNotFoundError(f"Zero-cost config file not found: {zero_cost_file}")

nas_robbench_file = Path(os.environ["NAS_ROBBENCH"])
if not nas_robbench_file.exists():
    raise FileNotFoundError(
        f"Nas-Robust Benchmark config file not found: {nas_robbench_file}"
    )


def get_key_in_data(arch):
    """
    Get the key which is used to represent the architecture in "self.data".
    """
    return "".join(map(str, arch))


def normalize_data_name(dataset_name):
    if dataset_name in ["CIFAR-10", "CIFAR-100"]:
        dataset_name = dataset_name.lower().replace("-", "")
    return dataset_name


def decode_architecture(encoded_architecture: tuple):
    ops = [AVAILABLE_OPERATIONS[idx] for idx in encoded_architecture]
    return "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(*ops)


class NASBench201(Problem):
    def __init__(self, dataset, maxEvals, fitness_metric, robust_type, **kwargs):
        """
        # NAS-Benchmark-201 provides us with the information (e.g., the training loss, the testing accuracy,
        the validation accuracy, the number of FLOPs, etc) of all architectures in the search space. Therefore, if we
        want to evaluate any architectures in the search space, we just need to query its information in the data.\n
        -----------------------------------------------------------------

        - path_data -> the path contains NAS-Bench-201 data.
        - data -> NAS-Bench-201 data.
        - available_ops -> the available operators can choose in the search space.
        - maxLength -> the maximum length of compact architecture.
        """
        super().__init__(maxEvals, "NASBench201", dataset, **kwargs)

        self.type_of_problem = kwargs["type_of_problem"]
        if self.type_of_problem == "single-objective":
            self.objectives_lst = ["val_acc"]
        elif self.type_of_problem == "multi-objective":
            self.objectives_lst = ["FLOPs", "val_error"]
        else:
            raise ValueError()

        """ ------- Additional Hyper-parameters ------- """
        self.available_ops = [0, 1, 2, 3, 4]
        self.maxLength = 6

        self.path_data = kwargs["path_data"] + "/NASBench201"
        self.fitness_metric = fitness_metric
        self.robust_type = robust_type
        self.data = None
        self.zero_cost_data = None
        self.robustness_data = None
        self.best_arch = None

    def _get_accuracy(self, arch, final=False):
        """
        - Get the accuracy of architecture. E.g., the testing accuracy, the validation accuracy.
        """
        key = get_key_in_data(arch)
        if final:
            acc = self.data["200"][key]["test_acc"]
        else:
            acc = self.data["200"][key]["val_acc"]
        return acc

    def _get_zero_cost_metric(self, arch, metric, final=False):
        try:
            dataset = normalize_data_name(self.dataset)
            zero_cost_eval_dict = self.zero_cost_data[dataset]
            encode_arch = str(tuple(map(int, arch)))
            score = zero_cost_eval_dict[encode_arch][metric]["score"]
            return score
        except Exception as e:
            raise e

    def _get_robustness_metric(self, arch, robust_type, final=False):
        try: 
            encode_arch = tuple(map(int, arch))
            decode_arch = decode_architecture(encode_arch)
            robustness_eval_dict = self.robustness_data[decode_arch]
            if robust_type == "autoattack":
                score = robustness_eval_dict[robust_type]
            else: 
                score = robustness_eval_dict[robust_type]["threeseed"]
            return score
        except Exception as e: 
            raise e 

    def _get_complexity_metric(self, arch):
        """
        - In NAS-Bench-201 problem, the efficiency metric is nFLOPs.
        - The returned nFLOPs is normalized.
        """
        key = get_key_in_data(arch)
        nFLOPs = round(
            (self.data["200"][key]["FLOPs"] - self.min_max["FLOPs"]["min"])
            / (self.min_max["FLOPs"]["max"] - self.min_max["FLOPs"]["min"]),
            6,
        )
        return nFLOPs

    def _set_up(self):
        available_datasets = ["CIFAR-10", "CIFAR-100", "ImageNet16-120"]
        if self.dataset not in available_datasets:
            raise ValueError(
                f"Just only supported these datasets: CIFAR-10; CIFAR-100; ImageNet16-120."
                f"{self.dataset} dataset is not supported at this time."
            )

        f_data = open(f"{self.path_data}/[{self.dataset}]_data.p", "rb")
        self.data = pickle.load(f_data)
        f_data.close()

        # load zero-cost data
        with open(zero_cost_file, "r") as file:
            self.zero_cost_data = json.load(file)
        file.close()

        # load attack method data
        with open(nas_robbench_file, "r") as rb_file:
            self.robustness_data = json.load(rb_file)
        rb_file.close()

        if self.type_of_problem == "single-objective":
            self.best_arch = None

        elif self.type_of_problem == "multi-objective":
            f_min_max = open(f"{self.path_data}/[{self.dataset}]_min_max.p", "rb")
            self.min_max = pickle.load(f_min_max)
            f_min_max.close()

            f_pareto_front_testing = open(
                f"{self.path_data}/[{self.dataset}]_pareto_front(testing).p", "rb"
            )
            self.pareto_front_testing = pickle.load(f_pareto_front_testing)
            f_pareto_front_testing.close()

            f_pareto_front_validation = open(
                f"{self.path_data}/[{self.dataset}]_pareto_front(validation).p", "rb"
            )
            self.pareto_front_validation = pickle.load(f_pareto_front_validation)
            f_pareto_front_validation.close()

        print("--> Set Up - Done")

    def _get_a_compact_architecture(self):
        return np.random.choice(self.available_ops, self.maxLength)

    def _evaluate(self, arch):
        if self.type_of_problem == "single-objective":
            if self.fitness_metric == "val_acc":
                acc = self.get_accuracy(arch)
            else:
                acc = self.get_zero_cost_metric(arch, self.fitness_metric)
            return acc
        elif self.type_of_problem == "multi-objective":
            complex_metric = self.get_complexity_metric(arch)
            return [complex_metric, 1 - acc]

    def _isValid(self, arch):
        return True
