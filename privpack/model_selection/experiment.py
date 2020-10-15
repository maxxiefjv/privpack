"""
This module defines how experiments are conducted. What is expected from your network
and how many train/validation splits should be trained and tested.
"""

from typing import List, Callable

from privpack.utils.metrics import Metric
from privpack.core.architectures import GenerativeAdversarialNetwork

from sklearn.model_selection import KFold
import torch
import json

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
        with torch.no_grad():
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

    def _average_metric_results(self, results):
        metric_names = [x.metric.name for x in self.expectations]

        train_average_metric_results = {}
        test_average_metric_results = {}

        for i in results.keys():
            train_resuls_i = results[i]['train']
            test_results_i = results[i]['test']

            for name in metric_names:
                train_metric_results = train_resuls_i[name]
                test_metric_results = test_results_i[name]

                train_average_metric_results.setdefault(name, {})
                train_average_metric_results[name]['satisfies'] = train_average_metric_results[name].setdefault('satisfies', True) and train_metric_results['satisfies']
                train_average_metric_results[name]['expected_out'] = train_metric_results['expected_out']

                train_average_metric_results[name].setdefault('actual_out', 0)
                train_average_metric_results[name]['actual_out'] += train_metric_results['actual_out'] / len(results.keys())

                test_average_metric_results.setdefault(name, {})
                test_average_metric_results[name]['satisfies'] = test_average_metric_results[name].setdefault('satisfies', True) and test_metric_results['satisfies']
                test_average_metric_results[name]['expected_out'] = test_metric_results['expected_out']

                test_average_metric_results[name].setdefault('actual_out', 0)
                test_average_metric_results[name]['actual_out'] += test_metric_results['actual_out'] / len(results.keys())

        return {
            'train_average': train_average_metric_results,
            'test_average': test_average_metric_results,
        }

    def run(self, data, n_splits, epochs, batch_size, verbose=False, **kwargs):
        kf = KFold(n_splits=n_splits)

        runs_results = {}
        for (index, (train_indices, test_indices)) in enumerate(kf.split(data)):
            self.network.reset()

            train_data = data[train_indices]
            test_data = data[test_indices]

            if (verbose):
                print(f"Training network with settings: {}".format(json.dumps({
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'n_splits': n_splits,
                    'gan_criterion': self.network.gan_criterion 
                }, indent=4)))

            self.network.train(train_data, test_data, epochs=epochs, batch_size=batch_size, verbose=verbose, **kwargs)

            train_expectations = self._compute_expecations(train_data)
            test_expectations = self._compute_expecations(test_data)

            total_expectations = {
                'train': train_expectations,
                'test': test_expectations
            }

            runs_results[index] = total_expectations

        runs_results['averages'] = self._average_metric_results(runs_results)
        return runs_results
