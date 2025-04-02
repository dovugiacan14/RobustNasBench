import json 
from scipy.stats import spearmanr
from collections import defaultdict
from libnas.lib.utils.search_space import NASBench201SearchSpace

zero_cost_metrics = [
    "epe_nas", 
    "fisher", 
    "grad_norm", 
    "flops", 
    "grasp", 
    "jacov",
    "l2_norm",
    "nwot",
    "params", 
    "plain", 
    "snip", 
    "synflow",
    "zen", 
    "swap",
    "meco", 
    "meco_opt", 
    "zico", 
    "val_accuracy"
]

nasbench_search_space = NASBench201SearchSpace()

def get_eval_arch(target_file, isomorph_file, des_file): 
    with open(target_file, "r") as tf: 
        target = json.load(tf)

    with open(isomorph_file, "r") as file:
        isomorph = json.load(file) 
    
    isomorph_lst = list(isomorph.values())
    for arch_lst in isomorph_lst: 
        if len(arch_lst) == 1: 
            continue 

        available_arch = arch_lst[0]
        eval_metric = target[available_arch]
        for i in range(1, len(arch_lst)):  
            target[arch_lst[i]] = eval_metric

    with open(des_file, 'w', encoding= 'utf-8') as f: 
            json.dump(target, f, indent= 4, ensure_ascii= False)

    return target


def compute_correlation(zc_nasbench, robustnas, dataset): 
    if dataset not in zc_nasbench: 
        return 
    evaluate_details = zc_nasbench[dataset]

    fgsm_3_acc = []
    fgsm_8_acc = []
    pdg_3_acc = []
    pdg_8_acc = []
    auto_attack = []
    for arch in evaluate_details: 
        continue 


if __name__ == "__main__": 
    zc_nasbench_file = "dataset/zc_nasbench201.json"
    with open(zc_nasbench_file, "r") as zc: 
        zc_nasbench = json.load(zc)

    robustnas_file = "dataset/cifar10.json"
    with open(robustnas_file, "r") as rf: 
        robustnas = json.load(rf)

    correlation = compute_correlation(zc_nasbench, robustnas, dataset= "cifar10")

    