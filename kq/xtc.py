import os

import luigi
import pandas
import numpy as np
import scipy.sparse as sp
from plumbum import colors

from kq import core, distances, dataset, question_vectors, shared_entites, tfidf_matrix, count_matrix, wordmat_distance
from sklearn import ensemble, metrics



class XTCBaseClassifier(luigi.Task):
    base_name = 'XXX'

    def requires(self):
        yield dataset.Dataset()
        yield shared_entites.SharedEntities()
        yield distances.AllDistances()
        yield count_matrix.CountFeature()
        yield wordmat_distance.WordMatDistance()

    def output(self):
        return luigi.LocalTarget('cache/XTC_%s/importance_report' % self.base_name)



    def run(self):
        self.output().makedirs()
        X, y, cols = self.load_data('train')
        weights = dict(enumerate(core.weights))
        cls = ensemble.ExtraTreesClassifier(
            n_estimators=512, n_jobs=2, verbose=10,
            bootstrap=True, min_samples_leaf=10,
            class_weight=weights)
        cls.fit(X, y)
        importances = pandas.Series(
            cls.feature_importances_,
            index=cols)

        report_data = str(importances.groupby([ix.split('.')[0] for ix in importances.index]).agg(['mean', 'max', 'min', 'sum']))
        print(report_data)

        X, y, _ = self.load_data('valid')
        preds = cls.predict_proba(X)[:, 1]
        weights = core.weights[y]
        loss = metrics.log_loss(y, preds, sample_weight=weights)
        print(colors.green | str(loss))

        X, y, _ = self.load_data('merge')
        merge_pred = cls.predict_proba(X)[:, 1]
        pandas.Series(merge_pred).to_csv('cache/XTC_%s/merge_predictions.csv' % self.base_name)

        X = self.load_test_data()
        pred = cls.predict_proba(X)[:, 1]
        pandas.Series(pred).to_csv('cache/XTC_%s/predictions.csv' % self.base_name)

        with self.output().open('w') as f:
            f.write(report_data)

    def load(self):
        assert self.complete()
        return pandas.read_csv('cache/XTC_%s/merge_predictions.csv' % self.base_name,
                               names=['test_id', 'pred'], index_col='test_id').values

    def load_test(self):
        assert self.complete()
        return pandas.read_csv('cache/XTC_%s/predictions.csv' % self.base_name,
                               names=['test_id', 'pred'], index_col='test_id').values


class XTCComplexClassifier(XTCBaseClassifier):
    base_name = 'complex'
    def load_data(self, subset):
        ix = {'train': 0, 'merge': 1, 'valid': 2}[subset]
        wv = count_matrix.CountFeature.load_dataset(subset)
        se = np.nan_to_num(shared_entites.SharedEntities().load()[ix])
        ad = distances.AllDistances().load()[ix]
        md = wordmat_distance.WordMatDistance().load(subset)
        y = dataset.Dataset().load()[ix].is_duplicate.values

        res = sp.hstack([wv, se, ad, md])
        cols = (
              ['count.%d' % i for i in range(wv.shape[1])]
            + ['shared_ent.%d' % i for i in range(se.shape[1])]
            + ['distance.%d' % i for i in range(ad.shape[1])]
            + ['wordmat.%d' % i for i in range(md.shape[1])]
        )
        return res, y, cols

    def load_test_data(self):
        wv = tfidf_matrix.TFIDFFeature.load_dataset('test')
        se = np.nan_to_num(shared_entites.SharedEntities().load_test())
        ad = distances.AllDistances().load_test()
        md = wordmat_distance.WordMatDistance().load('test')
        res = sp.hstack([wv, se, ad, md])
        return res

class XTCSimpleClassifier(XTCBaseClassifier):
    base_name = 'simple'

    def load_data(self, subset):
        ix = {'train': 0, 'merge': 1, 'valid': 2}[subset]
        se = np.nan_to_num(shared_entites.SharedEntities().load()[ix])
        ad = distances.AllDistances().load()[ix]
        md = wordmat_distance.WordMatDistance().load(subset)
        y = dataset.Dataset().load()[ix].is_duplicate.values

        res = np.hstack([se, ad, md])
        cols = (
            ['shared_ent.%d' % i for i in range(se.shape[1])]
            + ['distance.%d' % i for i in range(ad.shape[1])]
            + ['wordmat.%d' % i for i in range(md.shape[1])]
        )
        return res, y, cols

    def load_test_data(self):
        se = np.nan_to_num(shared_entites.SharedEntities().load_test())
        ad = distances.AllDistances().load_test()
        md = wordmat_distance.WordMatDistance().load('test')
        res = np.hstack([se, ad, md])
        return res
