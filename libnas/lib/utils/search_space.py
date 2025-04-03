import numpy as np
from abc import ABC, abstractmethod

AVAILABLE_OPERATIONS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]

def extract_operation(op_string):
    return op_string.split("~")[0]

class NeuralSearchSpace(ABC):
    def __init__(self, name):
        self.name = name

    def generate_architecture(self, genotype=False):
        architecture = self._random_sample()
        if genotype:
            architecture = self.encode_architecture(architecture)
        return architecture

    def validate_architecture(self, architecture_str):
        return True

    @abstractmethod
    def get_available_operations(self, node_idx):
        pass

    @abstractmethod
    def _random_sample(self):
        pass

    @abstractmethod
    def encode_architecture(self, architecture_str):
        pass

    @abstractmethod
    def decode_architecture(self, encoded_architecture):
        pass


class NASBench201SearchSpace(NeuralSearchSpace):
    def __init__(self):
        super().__init__("NAS-Bench-201")

    def _random_sample(self):
        sampled_ops = np.random.choice(AVAILABLE_OPERATIONS, 6)
        return "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(*sampled_ops)

    def encode_architecture(self, architecture_str):
        nodes = architecture_str.split("+")
        node_ops = [list(map(extract_operation, n.strip()[1:-1].split("|"))) for n in nodes]
        encoded_architecture = [AVAILABLE_OPERATIONS.index(op) for ops in node_ops for op in ops]
        return encoded_architecture

    def decode_architecture(self, encoded_architecture: tuple):
        ops = [AVAILABLE_OPERATIONS[idx] for idx in encoded_architecture]
        return "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(*ops)

    def get_available_operations(self, node_idx):
        return list(range(len(AVAILABLE_OPERATIONS)))


if __name__ == "__main__":
    architecture_str = "|nor_conv_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|none~0|nor_conv_3x3~1|avg_pool_3x3~2|"
    # network = (4, 0, 3, 1, 4, 3)
    search_space = NASBench201SearchSpace()
    encoded_architecture = search_space.encode_architecture(architecture_str)
    print(encoded_architecture)