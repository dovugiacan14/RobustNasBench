import os
import json
import pickle
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from constant import AVAILABLE_OPERATIONS, EDGE_LIST, OP_NAMES_NB201

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

def convert_str_to_ops(str_encoding):
    """
    Converts NB201 string representation to op_indices
    """
    nodes = str_encoding.split("+")

    def get_op(x):
        return x.split("~")[0]
    node_ops = [list(map(get_op, n.strip()[1:-1].split("|"))) for n in nodes]

    enc = []
    for u, v in EDGE_LIST:
        enc.append(OP_NAMES_NB201.index(node_ops[v - 2][u - 1]))

    return str(tuple(map(int, enc)))


class NASBench201:
    def __init__(self, dataset, maxEvals, zc_metric, **kwargs):
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
        self.dataset = dataset 
        self.maxEvals = maxEvals
        self.zc_metric = zc_metric
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
            acc = self.data["200"][key]["test_acc"][-1]
        else:
            acc = self.data["200"][key]["val_acc"][-1]
        return acc

    def _get_zero_cost_metric(self, arch, metric, final=False):
        try:
            dataset = normalize_data_name(self.dataset)
            zero_cost_eval_dict = self.zero_cost_data[dataset]
            str_arch = decode_architecture(arch)
            encode_arch = convert_str_to_ops(str_arch)
            score = zero_cost_eval_dict[encode_arch][metric]["score"]
            return score
        except Exception as e:
            raise e

    def _get_robustness_metric(self, arch):
        try: 
            summary_score = {}
            encode_arch = tuple(map(int, arch))
            decode_arch = decode_architecture(encode_arch)
            robustness_eval_dict = self.robustness_data[decode_arch]
            summary_score["rob_val_acc"] = robustness_eval_dict["val_acc"]["threeseed"]
            summary_score["val_fgsm_3"] = robustness_eval_dict["val_fgsm_3.0_acc"]["threeseed"]
            summary_score["val_fgsm_8"] = robustness_eval_dict["val_fgsm_8.0_acc"]["threeseed"]
            summary_score["val_pgd_3"] = robustness_eval_dict["val_pgd_3.0_acc"]["threeseed"]
            summary_score["val_pgd_8"] = robustness_eval_dict["val_pgd_8.0_acc"]["threeseed"]
            summary_score["autoattack"] = robustness_eval_dict["autoattack"]
            return summary_score
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
            if self.zc_metric == "val_acc":
                acc = self._get_accuracy(arch)
            else:
                acc = self._get_zero_cost_metric(arch, self.zc_metric)
            return acc
        elif self.type_of_problem == "multi-objective":
            complex_metric = self._get_complexity_metric(arch)
            return [complex_metric, 1 - acc]
