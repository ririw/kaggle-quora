import luigi
import nose.tools
import numpy as np
import os

import pandas
import pandas.core.generic


class FoldIndependent(luigi.Task):
    def load_all(self, name, as_df=True):
        assert name in {'train', 'test'}
        if name == 'train':
            return self._load(as_df)[0]
        else:
            return self._load_test(as_df)

    def load(self, name, fold, as_df=False):
        assert name in {'train', 'test', 'valid'}
        assert self.complete(), repr(self) + ' is not complete'
        if name == 'test':
            assert fold is None, 'If using test, fold should be None'
            res = self._load_test(as_df)
        else:
            features, folds = self._load(as_df)
            nose.tools.assert_is_instance(folds, np.ndarray, 'Error while loading: ' + repr(self))
            if fold is None:
                return features
            folds = (folds + fold) % fold_max
            if name == 'valid':
                fold_ix = folds == 0
            else:
                fold_ix = folds != 0
            res = features[fold_ix]

        if as_df:
            nose.tools.assert_is_instance(res, pandas.DataFrame)
        return res

    def _load(self, as_df):
        # Return features, folds
        raise NotImplementedError

    def _load_test(self, as_df):
        # return features
        raise NotImplementedError


class FoldDependent(luigi.Task):
    fold = luigi.IntParameter()

    def _load(self, name, as_df):
        raise NotImplementedError

    def load(self, name, as_df=False):
        res = self._load(name, as_df)
        if as_df:
            nose.tools.assert_is_instance(res, pandas.DataFrame)
        return res


class HyperTuneable:
    def score(self):
        raise NotImplementedError


fold_max = 6