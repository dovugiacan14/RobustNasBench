import json 
import pandas as pd
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

    # S1: collect data 
    fgsm_3_acc = []
    fgsm_8_acc = []
    pdg_3_acc = []
    pdg_8_acc = []
    auto_attack = []

    zc_eval_dict = defaultdict(list)
    for arch, details in evaluate_details.items(): 
        decode_arch = nasbench_search_space.decode_architecture(eval(arch))
        robust_evals = robustnas[decode_arch]

        # get information from robustnas 
        val_fgsm_3 = robust_evals["val_fgsm_3.0_acc"]
        val_fgsm_8 = robust_evals["val_fgsm_8.0_acc"]
        val_pgd_3 = robust_evals["val_pgd_3.0_acc"]
        val_pgd_8 = robust_evals["val_pgd_8.0_acc"]

        fgsm_3_acc.append(val_fgsm_3["threeseed"])
        fgsm_8_acc.append(val_fgsm_8["threeseed"])
        pdg_3_acc.append(val_pgd_3["threeseed"])
        pdg_8_acc.append(val_pgd_8["threeseed"])
        auto_attack.append(robust_evals["autoattack"])

        # get zero-cost results 
        for metric in zero_cost_metrics:
            if metric == "val_accuracy": 
                zc_eval_dict[metric].append(details[metric])
            else: 
                zc_eval_dict[metric].append(details[metric]['score'])
  
    # S2: compute Spearman correlation
    correlation_df = pd.DataFrame(
        index= ["val_fgsm_3.0_acc", "val_fgsm_8.0_acc", "val_pgd_3.0_acc", "val_pgd_8.0_acc"],
        columns= zc_eval_dict.keys(),
        dtype= float
    )
    p_value_df = pd.DataFrame(
        index= ["val_fgsm_3.0_acc", "val_fgsm_8.0_acc", "val_pgd_3.0_acc", "val_pgd_8.0_acc"],
        columns= zc_eval_dict.keys(),
        dtype= float
    )
    for i, values in zip( ["val_fgsm_3.0_acc", "val_fgsm_8.0_acc", "val_pgd_3.0_acc", "val_pgd_8.0_acc"], [fgsm_3_acc, fgsm_8_acc, pdg_3_acc, pdg_8_acc]):
        for j in zc_eval_dict.keys(): 
            corr, p_val = spearmanr(values, zc_eval_dict[j])
            correlation_df.loc[i, j] = corr
            p_value_df.loc[i, j] = p_val

    # S3: save result
    with pd.ExcelWriter(f"{dataset}.xlsx") as writer: 
        correlation_df.to_excel(writer, sheet_name= "correlation")
        p_value_df.to_excel(writer, sheet_name= "p_value")
    
    print("Done.!")


if __name__ == "__main__": 
    zc_nasbench_file = "dataset/zc_nasbench201.json"
    with open(zc_nasbench_file, "r") as zc: 
        zc_nasbench = json.load(zc)

    robustnas_file = "dataset/cifar10.json"
    with open(robustnas_file, "r") as rf: 
        robustnas = json.load(rf)

    compute_correlation(zc_nasbench, robustnas, dataset= "cifar10")

    