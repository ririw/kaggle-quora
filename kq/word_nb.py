import os

import luigi
import numpy as np
from plumbum import colors
from sklearn import naive_bayes, metrics

from kq import core
from kq.count_matrix import CountFeature
from kq.dataset import Dataset


class NaiveBayesClassifier(luigi.Task):
    resources = {'cpu': 1}

    def requires(self):
        yield CountFeature()
        yield Dataset()

    def output(self):
        return luigi.LocalTarget('cache/nb/predictions.npy')

    def run(self):
        self.output().makedirs()
        cls = naive_bayes.BernoulliNB()
        cls.fit(CountFeature().load_dataset('train'),
                Dataset().load_named('train').is_duplicate)

        preds = cls.predict_proba(CountFeature().load_dataset('valid'))[:, 1]
        valid_isdup = Dataset().load_named('valid').is_duplicate
        weights = core.weights[valid_isdup.values]
        loss = metrics.log_loss(valid_isdup.values, preds, sample_weight=weights)
        print(colors.green | str(loss))

        merge_pred = cls.predict_proba(CountFeature().load_dataset('merge'))[:, 1]
        np.save('cache/nb/merge_predictions.npy', merge_pred)

        test = CountFeature().load_dataset('test')
        pred = cls.predict_proba(test)[:, 1]
        np.save('cache/nb/predictions_tmp.npy', pred)
        os.rename('cache/nb/predictions_tmp.npy', 'cache/nb/predictions.npy')

    def load(self):
        assert self.complete()
        return np.load('cache/nb/merge_predictions.npy')

    def load_test(self):
        assert self.complete()
        return np.load('cache/nb/predictions.npy')

