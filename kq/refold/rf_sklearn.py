import luigi
import numpy as np
import pandas
from plumbum import colors

from kq import core
from kq.feat_abhishek import FoldDependent, FoldIndependent
from kq.refold import rf_dataset

__all__ = ['RF_SKLearn',]


class RF_SKLearn(FoldDependent):
    def make_path(self, fname):
        raise NotImplementedError

    def xdataset(self) -> FoldIndependent:
        raise NotImplementedError

    def make_cls(self):
        raise NotImplementedError('Implement Sklearn compatible classifier')

    def _load(self, name, as_df):
        path = self.make_path(name + '.npz')
        if as_df:
            return pandas.DataFrame({self.__class__.__name__: np.load(path)['data']})
        else:
            return np.load(path)['data']

    def requires(self):
        yield rf_dataset.Dataset()
        yield self.xdataset()

    def output(self):
        return luigi.LocalTarget(self.make_path('done'))

    def run(self):
        self.output().makedirs()
        data = self.xdataset()

        X = data.load('train', self.fold)
        y = rf_dataset.Dataset().load('train', self.fold, as_df=True).is_duplicate

        cls = self.make_cls()
        print('Training classifier {:s} on data of size: {}'.format(repr(cls), X.shape))
        cls.fit(X, y)

        X_val = data.load('valid', self.fold)
        y_val = rf_dataset.Dataset().load('valid', self.fold, as_df=True).is_duplicate

        y_pred = cls.predict_proba(X_val)[:, 1]
        np.savez_compressed(self.make_path('valid.npz'), data=y_pred)
        score = core.score_data(y_val, y_pred)

        del X, y, X_val, y_val
        X_test = data.load('test', None)
        y_test_pred = cls.predict_proba(X_test)[:, 1]
        np.savez_compressed(self.make_path('test.npz'), data=y_test_pred)

        print(colors.green | 'Score: {:s}: {:f}'.format(repr(self), score))

        with self.output().open('w') as f:
            f.write('Score: {:s}: {:f}'.format(repr(self), score))
        return score