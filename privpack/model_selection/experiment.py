from typing import List, Callable

from privpack.utils.metrics import Metric
from privpack.core.architectures import GenerativeAdversarialNetwork

from sklearn.model_selection import KFold
import torch

class Expectations:
    
    class Expectation:
        def __init__(self, metric: Metric, value: float, relation: Callable[[float, float], bool] = lambda x,y: x == y):
            self.metric = metric
            self.value = value
            self.relation = relation

    def __init__(self):
        self._current_index = 0
        self.expectations: List[self.Expectation] = []

    def add_expectation(self, metric: Metric, value: float, relation: Callable[[float, float], bool] = lambda x,y: x == y):
        expectation = self.Expectation(metric, value, relation)
        self.expectations.append(expectation)
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if (self._current_index < len(self.expectations)):
            res = self.expectations[self._current_index]
            self._current_index += 1
            return res

        self._current_index = 0
        raise StopIteration


class Experiment:

    def __init__(self, network: GenerativeAdversarialNetwork, expectations: Expectations):
        self.network = network
        self.expectations = expectations

    def _compute_expecations(self, data):
        released_data = self.network(data)
        results = {}

        for expectation in self.expectations:
            actual_out = expectation.metric(released_data, data)
            expected_out = expectation.value
            status = expectation.relation(actual_out, expected_out)

            results[expectation.metric.name] = {
                'satisfies': bool(status),
                'expected_out': expected_out,
                'actual_out': actual_out
            }

        return results

    def run(self, data, n_splits, epochs, batch_size, **kwargs):
        kf = KFold(n_splits=n_splits)

        runs_results = {}
        for (index, (train_indices, test_indices)) in enumerate(kf.split(data)):
            self.network.reset()

            train_data = data[train_indices]
            test_data = data[test_indices]
            
            self.network.train(train_data, test_data, epochs=epochs, batch_size=batch_size, **kwargs)

            train_expectations = self._compute_expecations(train_data)
            test_expectations = self._compute_expecations(test_data)

            total_expectations = {
                'train': train_expectations,
                'test': test_expectations
            }

            runs_results[index] = total_expectations

        return runs_results

            