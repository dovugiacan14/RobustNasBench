import pickle
import json
import numpy as np 

path  = "data/NASBench201/[CIFAR-100]_data.p"
with open(path, "rb") as file:
    data = pickle.load(file)

print(data)

file = "config/zc_nasbench201.json"
with open(file, "r") as f:
    conf = json.load(f)

print(conf)


# OP_NAMES_NB201 = [
#     "skip_connect",
#     "none",
#     "nor_conv_3x3",
#     "nor_conv_1x1",
#     "avg_pool_3x3",
# ]
# AVAILABLE_OPERATIONS = [
#     "none",
#     "skip_connect",
#     "nor_conv_1x1",
#     "nor_conv_3x3",
#     "avg_pool_3x3",
# ]

# EDGE_LIST = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))

# arch = np.array([4, 0, 3, 3, 3, 1])


# def decode_architecture(encoded_architecture: tuple):
#     ops = [AVAILABLE_OPERATIONS[idx] for idx in encoded_architecture]
#     return "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(*ops)


# def convert_str_to_op_indices(str_encoding):
#     """
#     Converts NB201 string representation to op_indices
#     """
#     nodes = str_encoding.split("+")

#     def get_op(x):
#         return x.split("~")[0]

#     node_ops = [list(map(get_op, n.strip()[1:-1].split("|"))) for n in nodes]

#     enc = []
#     for u, v in EDGE_LIST:
#         enc.append(OP_NAMES_NB201.index(node_ops[v - 2][u - 1]))

#     return tuple(enc)

# str_arch = decode_architecture(arch)
# enc = convert_str_to_op_indices(str_arch)
# print(0)


