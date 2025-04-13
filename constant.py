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
        "maxEvals": 3000,
        "dataset": "CIFAR-100",
        "type_of_problem": "single-objective",
    },
    "SO-NAS201-3": {
        "maxEvals": 3000,
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
    "SO-NAS201-1": 20,
    "SO-NAS201-2": 20,
    "SO-NAS201-3": 20,
    "MO-NAS101": 100,
    "MO-NAS201-1": 20,
    "MO-NAS201-2": 20,
    "MO-NAS201-3": 20,
}

zero_cost_metrics = {
    0:  "epe_nas", 
    1:  "fisher", 
    2:  "grad_norm", 
    3:  "grasp",
    4:  "l2_norm", 
    5:  "jacov", 
    6:  "nwot", 
    7:  "plain", 
    8:  "snip",
    9:  "synflow",
    10: "zen"
}

attack_method = {
    0: "val_fgsm_3.0_acc",
    1: "val_fgsm_8.0_acc",
    2: "val_pgd_3.0_acc",
    3: "val_pgd_8.0_acc",
    4: "autoattack",
    5: "val_acc"
}

# Encode - Decode Architecture 
AVAILABLE_OPERATIONS = [
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3",
]

OP_NAMES_NB201 = [
    "skip_connect",
    "none",
    "nor_conv_3x3",
    "nor_conv_1x1",
    "avg_pool_3x3",
]

EDGE_LIST = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
