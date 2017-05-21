import hyperopt
import luigi
import numpy as np
import sklearn.linear_model
from plumbum import colors
from sklearn import pipeline, preprocessing

from kq import core
from kq.feat_abhishek import FoldDependent, hyper_helper
from kq.refold import rf_dataset, rf_word_count_features, BaseTargetBuilder, rf_ab

__all__ = ['ABLinear']

class ABLinear(FoldDependent):
    resources = {'cpu': 1}

    C = hyper_helper.LuigiTuneableHyperparam(
        prior=hyperopt.hp.randint('ABLinear.C', 6),
        transform = lambda v: 10 ** ((v-2)/2),
        default=4,
        disable=False)

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_ab_lin',
            'C_{:f}'.format(self.C),
            str(self.fold)
        )
        return (base_path + fname).get()

    def _load(self, name):
        path = self.make_path(name + '.npz')
        return np.load(path)['data']

    def requires(self):
        yield rf_dataset.Dataset()
        yield rf_ab.ABDataset()

    def output(self):
        return luigi.LocalTarget(self.make_path('done'))

    def run(self):
        self.output().makedirs()
        ab_data = rf_ab.ABDataset()

        X = ab_data.load('train', self.fold, as_np=False)
        y = rf_dataset.Dataset().load('train', self.fold, as_np=False).is_duplicate

        preproc = pipeline.Pipeline([
            ('norm', preprocessing.MinMaxScaler(feature_range=(-1, 1))),
            ('poly', preprocessing.PolynomialFeatures(2))
        ])

        cls = sklearn.linear_model.LogisticRegression(C=self.C, solver='sag')
        cls.fit(preproc.fit_transform(X), y)

        X_val = ab_data.load('valid', self.fold, as_np=False)
        y_val = rf_dataset.Dataset().load('valid', self.fold, as_np=False).is_duplicate

        y_pred = cls.predict_proba(preproc.transform(X_val))[:, 1]
        np.savez_compressed(self.make_path('valid.npz'), data=y_pred)
        score = core.score_data(y_val, y_pred)

        del X, y, X_val, y_val
        X_test = ab_data.load('test', None, as_np=False)
        y_test_pred = cls.predict_proba(preproc.transform(X_test))[:, 1]
        np.savez_compressed(self.make_path('test.npz'), data=y_test_pred)

        print(colors.green | 'Score: {:s}: {:f}'.format(repr(self), score))

        with self.output().open('w') as f:
            f.write('Score: {:s}: {:f}'.format(repr(self), score))
        return score
