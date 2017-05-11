import luigi
import pandas
import scipy.sparse as sp
from plumbum import colors
from sklearn import ensemble, metrics

from kq import core, dataset, count_matrix
from kq import feature_collection


class XTCBaseClassifier(luigi.Task):
    resources = {'cpu': 6}
    base_name = 'XXX'
    n_trees = 500

    def requires(self):
        yield dataset.Dataset()
        yield count_matrix.CountFeature()
        yield feature_collection.FeatureCollection()

    def output(self):
        return luigi.LocalTarget('cache/XTC_%s/importance_report' % self.base_name)

    def load_data(self, name):
        raise NotImplemented()

    def run(self):
        self.output().makedirs()
        X, y, cols = self.load_data('train')
        weights = dict(enumerate(core.weights))
        cls = ensemble.ExtraTreesClassifier(
            n_estimators=self.n_trees, n_jobs=-1, verbose=10,
            bootstrap=True, min_samples_leaf=10,
            oob_score=False, class_weight=weights)
        cls.fit(X, y)
        importances = pandas.Series(
            cls.feature_importances_,
            index=cols)

        report_data = str(
            importances.groupby([ix.split('.')[0] for ix in importances.index]).agg(['mean', 'max', 'min', 'sum']))
        print(report_data)

        X, y, _ = self.load_data('valid')
        preds = cls.predict_proba(X)[:, 1]
        weights = core.weights[y]
        loss = metrics.log_loss(y, preds, sample_weight=weights)
        print(colors.green | str(loss))

        X, y, _ = self.load_data('merge')
        merge_pred = cls.predict_proba(X)[:, 1]
        pandas.Series(merge_pred).to_csv('cache/XTC_%s/merge_predictions.csv' % self.base_name)

        X, y, _ = self.load_data('test')
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
    n_trees = 500

    def load_data(self, subset):
        X1 = feature_collection.FeatureCollection().load(subset)
        X2 = count_matrix.CountFeature.load_dataset(subset)
        y = dataset.Dataset().load_named(subset).is_duplicate.values

        res = sp.hstack([X1.values, X2])
        cols = list(X1.columns) + ['count.%d' % i for i in range(X2.shape[1])]
        return res, y, cols


class XTCSimpleClassifier(XTCBaseClassifier):
    base_name = 'simple'
    n_trees = 2000

    def load_data(self, subset):
        X = feature_collection.FeatureCollection().load_named(subset)
        y = dataset.Dataset().load_named(subset).is_duplicate.values
        cols = X.columns
        return X.values, y, cols

