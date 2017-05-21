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
            #print(type(folds))
            #print(type(features))
            nose.tools.assert_is_instance(folds, np.ndarray, 'Error while loading: ' + repr(self))
            if fold is None:
                return features
            folds = (folds + fold) % fold_max
            if name == 'valid':
                fold_ix = folds == 0
            else:
                fold_ix = folds != 0
            res = features[fold_ix]

        if as_np:
            return res.values
        else:
            return res

    def _load(self):
        # Return features, folds
        raise NotImplementedError

    def _load_test(self):
        # return features
        raise NotImplementedError


class FoldDependent(luigi.Task):
    fold = luigi.IntParameter()

    def _load(self, name):
        raise NotImplementedError

    def load(self, name, as_np=True):
        res = self._load(name)
        nose.tools.assert_is_instance(res, pandas.DataFrame)
        if as_np:
            return res.values
        else:
            return res

class HyperTuneable:
    def score(self):
        raise NotImplementedError


fold_max = 9