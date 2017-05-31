import pandas
import nose.tools
import numpy as np
import sklearn


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
dictweights = dict(enumerate(weights))


def weight_from(y):
    return weights[y]


def score_data(y_true, y_pred, weighted=True):
    global weights
    y_pred = np.asarray(y_pred)
    if len(y_pred.shape) == 2:
        if y_pred.shape[1] == 1:
            y_pred = y_pred[:, 0]
        elif y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
        else:
            assert y_pred.shape[1] <= 2, 'Unexpected shape: ' + str(y_pred.shape)

    nose.tools.assert_equal(y_true.shape, y_pred.shape)
    weights = weights[y_true]
    if weighted:
        loss = sklearn.metrics.log_loss(y_true, y_pred, sample_weight=weights)
    else:
        loss = sklearn.metrics.log_loss(y_true, y_pred)

    return loss


def fillna(vec, v=0, inplace=False):
    if not inplace:
        vec = vec.copy()
    isna = np.isnan(vec)
    vec[isna] = v
    return vec