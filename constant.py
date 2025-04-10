problem_configuration = {
    "SO-NAS101": {
        "maxEvals": 5000,
        "dataset": "CIFAR-10",
        "type_of_problem": "single-objective",
    },
    "SO-NAS201-1": {
        "maxEvals": 3000,
        "dataset": "CIFAR-10",
        "type_of_problem": "single-objective",
    },
    "SO-NAS201-2": {
        "maxEvals": 1000,
        "dataset": "CIFAR-100",
        "type_of_problem": "single-objective",
    },
    "SO-NAS201-3": {
        "maxEvals": 1000,
        "dataset": "ImageNet16-120",
        "type_of_problem": "single-objective",
    },
    "MO-NAS101": {
        "maxEvals": 30000,
        "dataset": "CIFAR-10",
        "type_of_problem": "multi-objective",
    },
    "MO-NAS201-1": {
        "maxEvals": 3000,
        "dataset": "CIFAR-10",
        "type_of_problem": "multi-objective",
    },
    "MO-NAS201-2": {
        "maxEvals": 3000,
        "dataset": "CIFAR-100",
        "type_of_problem": "multi-objective",
    },
    "MO-NAS201-3": {
        "maxEvals": 3000,
        "dataset": "ImageNet16-120",
        "type_of_problem": "multi-objective",
    },
}

population_size_dict = {
    "SO-NAS101": 100,
    "SO-NAS201-1": 40,
    "SO-NAS201-2": 40,
    "SO-NAS201-3": 40,
    "MO-NAS101": 100,
    "MO-NAS201-1": 20,
    "MO-NAS201-2": 20,
    "MO-NAS201-3": 20,
}

zero_cost_metrics = {
    0:  "epe_nas", 
    1:  "fisher", 
    2:  "flops", 
    3:  "grad_norm", 
    4:  "grasp",
    5:  "l2_norm", 
    6:  "jacov", 
    7:  "nwot", 
    8:  "params", 
    9:  "plain", 
    10: "snip",
    11: "synflow",
    12: "zen"
}

attack_method = {
    0: "val_fgsm_3.0_acc",
    1: "val_fgsm_8.0_acc",
    2: "val_pgd_3.0_acc",
    3: "val_pgd_8.0_acc",
    4: "autoattack",
}

AVAILABLE_OPERATIONS = [
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3",
]
