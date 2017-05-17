import luigi
import numpy as np
import os


class FoldIndependent(luigi.Task):
    def load(self, name, fold):
        assert self.complete()
        if name == 'test':
            assert fold is None, 'If using test, fold should be None'
            return self._load_test()
        features, folds = self._load()
        folds = (folds + fold) % fold_max
        if name == 'valid':
            return features[folds == 0]
        if name == 'train':
            return features[folds != 0]
        assert False

    def _load(self):
        raise NotImplementedError

    def _load_test(self):
        raise NotImplementedError


class FoldDependent(luigi.Task):
    fold = luigi.IntParameter()

    def load(self, name):
        raise NotImplementedError


fold_max = 9