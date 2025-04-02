import json 

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

def compute_correlation(): 
    pass 


if __name__ == "__main__": 
    zc_nasbench_file = "dataset/zc_nasbench_201.json"
    
    pass 
    