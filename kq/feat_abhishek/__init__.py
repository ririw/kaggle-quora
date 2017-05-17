import luigi
import nose.tools
import numpy as np
import os

import pandas
import pandas.core.generic


class FoldIndependent(luigi.Task):
    def load(self, name, fold, as_np=True):
        assert name in {'train', 'test', 'valid'}
        assert self.complete()
        if name == 'test':
            assert fold is None, 'If using test, fold should be None'
            res = self._load_test()
        else:
            features, folds = self._load()
            nose.tools.assert_is_instance(folds, np.ndarray, 'Error while loading: ' + repr(self))
            folds = (folds + fold) % fold_max
            if name == 'valid':
                res = features[folds == 0]
            else:
                res = features[folds != 0]

        nose.tools.assert_is_instance(res, pandas.core.generic.NDFrame)
        if as_np:
            return res.values
        else:
            return res

    def _load(self):
        raise NotImplementedError

    def _load_test(self):
        raise NotImplementedError


class FoldDependent(luigi.Task):
    fold = luigi.IntParameter()

    def _load(self, name):
        raise NotImplementedError

    def load(self, name):
        res = self._load(name)
        nose.tools.assert_is_instance(res, pandas.DataFrame)

fold_max = 9