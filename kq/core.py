import pandas
import numpy as np


class MergableFeatures:
    def train_feature(self):
        raise NotImplementedError

    def test_feature(self):
        raise NotImplementedError

    def valid_feature(self):
        raise NotImplementedError

    @staticmethod
    def train_multiple(features):
        data = pandas.concat([feature.train_feature() for feature in features], 1)
        return data

    @staticmethod
    def test_multiple(features):
        data = pandas.concat([feature.test_feature() for feature in features], 1)
        return data

    @staticmethod
    def valid_multiple(features):
        data = pandas.concat([feature.valid_feature() for feature in features], 1)
        return data


weights = np.array([1.309028344, 0.472001959])