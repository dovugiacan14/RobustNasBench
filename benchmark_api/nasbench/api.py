# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import random


from benchmark_api.nasbench.lib import config
from benchmark_api.nasbench.lib import model_spec as _model_spec
import numpy as np

# Bring ModelSpec to top-level for convenience. See lib/model_spec.py.
ModelSpec = _model_spec.ModelSpec


class OutOfDomainError(Exception):
    """Indicates that the requested graph is outside of the search domain."""


class NASBench(object):
    """User-facing API for accessing the NASBench dataset."""
    def __init__(self, dataset_file=None, seed=None):
        """Initialize dataset, this should only be done once per experiment.

        Args:
          dataset_file: path to .tfrecord file containing the dataset.
          seed: random seed used for sampling queried models. Two NASBench objects
            created with the same seed will return the same data points when queried
            with the same models in the same order. By default, the seed is randomly
            generated.
        """
        self.config = config.build_config()
        random.seed(seed)
       
        self.history = {}
        self.training_time_spent = 0.0
        self.total_epochs_spent = 0

    def query(self, model_spec, epochs=108, stop_halfway=False):
        """Fetch one of the evaluations for this model spec.

        Each call will sample one of the config['num_repeats'] evaluations of the
        model. This means that repeated queries of the same model (or isomorphic
        models) may return identical metrics.

        This function will increment the budget counters for benchmarking purposes.
        See self.training_time_spent, and self.total_epochs_spent.

        This function also allows querying the evaluation metrics at the halfway
        point of training using stop_halfway. Using this option will increment the
        budget counters only up to the halfway point.

        Args:
          model_spec: ModelSpec object.
          epochs: number of epochs trained. Must be one of the evaluated number of
            epochs, [4, 12, 36, 108] for the full dataset.
          stop_halfway: if True, returned dict will only contain the training time
            and accuracies at the halfway point of training (num_epochs/2).
            Otherwise, returns the time and accuracies at the end of training
            (num_epochs).

        Returns:
          dict containing the evaluated data for this object.

        Raises:
          OutOfDomainError: if model_spec or num_epochs is outside the search space.
        """
        if epochs not in self.valid_epochs:
          raise OutOfDomainError('invalid number of epochs, must be one of %s'
                                 % self.valid_epochs)

        fixed_stat, computed_stat = self.get_metrics_from_spec(model_spec)
        sampled_index = random.randint(0, self.config['num_repeats'] - 1)
        computed_stat = computed_stat[epochs][sampled_index]

        data = dict()
        data['module_adjacency'] = fixed_stat['module_adjacency']
        data['module_operations'] = fixed_stat['module_operations']
        data['trainable_parameters'] = fixed_stat['trainable_parameters']

        if stop_halfway:
            data['training_time'] = computed_stat['halfway_training_time']
            data['train_accuracy'] = computed_stat['halfway_train_accuracy']
            data['validation_accuracy'] = computed_stat['halfway_validation_accuracy']
            data['test_accuracy'] = computed_stat['halfway_test_accuracy']
        else:
            data['training_time'] = computed_stat['final_training_time']
            data['train_accuracy'] = computed_stat['final_train_accuracy']
            data['validation_accuracy'] = computed_stat['final_validation_accuracy']
            data['test_accuracy'] = computed_stat['final_test_accuracy']
            self.training_time_spent += data['training_time']
        if stop_halfway:
            self.total_epochs_spent += epochs // 2
        else:
            self.total_epochs_spent += epochs

        return data

    def is_valid(self, model_spec):
        """Checks the validity of the model_spec.

        For the purposes of benchmarking, this does not increment the budget
        counters.

        Args:
          model_spec: ModelSpec object.

        Returns:
          True if model is within space.
        """
        try:
            self._check_spec(model_spec)
        except OutOfDomainError:
            return False
        return True

    def get_budget_counters(self):
        """Returns the time and budget counters."""
        return self.training_time_spent, self.total_epochs_spent

    def reset_budget_counters(self):
        """Reset the time and epoch budget counters."""
        self.training_time_spent = 0.0
        self.total_epochs_spent = 0

    def evaluate(self, model_spec, model_dir):
        """Trains and evaluates a model spec from scratch (does not query dataset).

        This function runs the same procedure that was used to generate each
        evaluation in the dataset.  Because we are not querying the generated
        dataset of trained models, there are no limitations on number of vertices,
        edges, operations, or epochs. Note that the results will not exactly match
        the dataset due to randomness. By default, this uses TPUs for evaluation but
        CPU/GPU can be used by setting --use_tpu=false (GPU will require installing
        tensorflow-gpu).

        Args:
          model_spec: ModelSpec object.
          model_dir: directory to store the checkpoints, summaries, and logs.

        Returns:
          dict contained the evaluated data for this object, same structure as
          returned by query().
        """
        pass

    def hash_iterator(self):
        """Returns iterator over all unique model hashes."""
        return self.fixed_statistics.keys()

    def get_metrics_from_hash(self, module_hash):
        """Returns the metrics for all epochs and all repeats of a hash.

        This method is for dataset analysis and should not be used for benchmarking.
        As such, it does not increment any of the budget counters.

        Args:
          module_hash: MD5 hash, i.e., the values yielded by hash_iterator().

        Returns:
          fixed stats and computed stats of the model spec provided.
        """
        fixed_stat = copy.deepcopy(self.fixed_statistics[module_hash])
        computed_stat = copy.deepcopy(self.computed_statistics[module_hash])
        return fixed_stat, computed_stat

    def get_metrics_from_spec(self, model_spec):
        """Returns the metrics for all epochs and all repeats of a model.
        This method is for dataset analysis and should not be used for benchmarking.
        As such, it does not increment any of the budget counters.

        Args:
          model_spec: ModelSpec object.

        Returns:
          fixed stats and computed stats of the model spec provided.
        """
        self._check_spec(model_spec)
        module_hash = self._hash_spec(model_spec)
        return self.get_metrics_from_hash(module_hash)

    def _check_spec(self, model_spec):
        """Checks that the model spec is within the dataset."""
        if not model_spec.valid_spec:
            raise OutOfDomainError('invalid spec, provided graph is disconnected.')

        num_vertices = len(model_spec.ops)
        num_edges = np.sum(model_spec.matrix)

        if num_vertices > self.config['module_vertices']:
            raise OutOfDomainError('too many vertices')

        if num_edges > self.config['max_edges']:
            raise OutOfDomainError('too many edges')

        if model_spec.ops[0] != 'input':
            raise OutOfDomainError('first operation should be \'input\'')

        if model_spec.ops[-1] != 'output':
            raise OutOfDomainError('last operation should be \'output\'')

        for op in model_spec.ops[1:-1]:
            if op not in self.config['available_ops']:
                raise OutOfDomainError('unsupported op %s (available ops = %s)'
                                       % (op, self.config['available_ops']))

    def _hash_spec(self, model_spec):
        """Returns the MD5 hash for a provided model_spec."""
        return model_spec.hash_spec(self.config['available_ops'])


class _NumpyEncoder(json.JSONEncoder):
    """Converts numpy objects to JSON-serializable format."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Matrices converted to nested lists
            return obj.tolist()
        elif isinstance(obj, np.generic):
            # Scalars converted to closest Python type
            return np.asscalar(obj)
        return json.JSONEncoder.default(self, obj)
