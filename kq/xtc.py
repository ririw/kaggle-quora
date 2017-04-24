import os

import luigi
import pandas
import numpy as np
import scipy.sparse as sp
from plumbum import colors

from kq import core, distances, dataset, shared_words, shared_entites, tfidf_matrix
from sklearn import ensemble, metrics


class XTCClassifier(luigi.Task):
    def requires(self):
        yield dataset.Dataset()
        yield shared_words.WordVectors()
        yield shared_entites.SharedEntities()
        yield distances.AllDistances()
        yield tfidf_matrix.TFIDFFeature()

    def output(self):
        return luigi.LocalTarget('cache/XTC/predictions.csv')

    def load_data(self, subset):
        ix = {'train': 0, 'merge': 1, 'valid': 2}[subset]
        wv = tfidf_matrix.TFIDFFeature.load_dataset(subset)
        se = np.nan_to_num(shared_entites.SharedEntities().load()[ix])
        ad = distances.AllDistances().load()[ix]
        y = dataset.Dataset().load()[ix].is_duplicate.values

        res = sp.hstack([wv, se, ad])

        return res, y

    def load_test_data(self):
        wv = tfidf_matrix.TFIDFFeature.load_dataset('test')
        se = np.nan_to_num(shared_entites.SharedEntities().load_test())
        ad = distances.AllDistances().load_test()
        res = sp.hstack([wv, se, ad])
        return res

    def run(self):
        self.output().makedirs()
        X, y = self.load_data('train')
        weights = dict(enumerate(core.weights))
        cls = ensemble.ExtraTreesClassifier(
            n_estimators=512, n_jobs=2, verbose=10,
            bootstrap=True, min_samples_leaf=10,
            class_weight=weights)
        cls.fit(X, y)

        X, y = self.load_data('valid')
        preds = cls.predict_proba(X)[:, 1]
        weights = core.weights[y]
        loss = metrics.log_loss(y, preds, sample_weight=weights)
        print(colors.green | str(loss))

        X, y = self.load_data('merge')
        merge_pred = cls.predict_proba(X)[:, 1]
        pandas.Series(merge_pred).to_csv('cache/XTC/merge_predictions.csv')

        X = self.load_test_data()
        pred = cls.predict_proba(X)[:, 1]
        pandas.Series(pred).to_csv('cache/XTC/predictions_tmp.csv')
        os.rename('cache/XTC/predictions_tmp.csv', 'cache/XTC/predictions.csv')

    def load(self):
        assert self.complete()
        return pandas.read_csv('cache/XTC/merge_predictions.csv', names=['test_id', 'pred'], index_col='test_id').values

    def load_test(self):
        assert self.complete()
        return pandas.read_csv('cache/XTC/predictions.csv', names=['test_id', 'pred'], index_col='test_id').values

