import logging

import coloredlogs
import luigi
import pandas
from tqdm import tqdm

from kq import count_matrix, dataset, core
from scipy import sparse as sp
import numpy as np
from scipy.spatial import distance
from sklearn import decomposition
import lightgbm.sklearn

__all__ = ['LDADecompositionFeatureVectors', 'NMFDecompositionFeatureVectors']


class DecompositionFeatureVectors(luigi.Task):
    resources = {'cpu': 7, 'mem': 2}

    def make_decomposer(self):
        raise NotImplementedError()

    def requires(self):
        yield count_matrix.CountFeature()
        yield dataset.Dataset()

    def output(self):
        return luigi.LocalTarget('cache/decomp_feature/{:s}/done'.format(self.__class__.__name__))

    def apply_model(self, name):
        data = count_matrix.CountFeature().load_full_mats(name)
        X = sp.vstack(data)
        transf = self.decomposer.transform(X)
        return self.dists_and_stuff(transf[:transf.shape[0] // 2], transf[transf.shape[0] // 2:])

    def dists_and_stuff(self, v1, v2):
        assert v1.shape == v2.shape
        deuc = np.asarray([distance.euclidean(a, b) for a, b in
                           tqdm(zip(v1, v2), desc='euclidean', total=v1.shape[0])])[:, None]
        dcit = np.asarray([distance.cityblock(a, b) for a, b in
                           tqdm(zip(v1, v2), desc='cityblock', total=v1.shape[0])])[:, None]
        dcos = np.asarray([distance.cosine(a, b) for a, b in
                           tqdm(zip(v1, v2), desc='cosine', total=v1.shape[0])])[:, None]
        dcor = np.asarray([distance.correlation(a, b) for a, b in
                           tqdm(zip(v1, v2), desc='correlation', total=v1.shape[0])])[:, None]
        absdiff = np.abs(v1 - v2)

        return np.concatenate([absdiff, deuc, dcit, dcos, dcor], 1)

    def run(self):
        coloredlogs.install(level=logging.INFO)
        self.output().makedirs()
        logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        logger.info('Loading train data')
        train_data = count_matrix.CountFeature().load_full_mats('train')
        X = sp.vstack(train_data)

        logger.info('Training decomposer')
        self.decomposer = self.make_decomposer()
        train_decomp = self.decomposer.fit_transform(X)
        train_d1, train_d2 = train_decomp[:train_decomp.shape[0] // 2], train_decomp[train_decomp.shape[0] // 2:]
        logger.info('Calculating train distances')
        train_dists = self.dists_and_stuff(train_d1, train_d2)
        logger.info('Saving train data')
        np.save('cache/decomp_feature/{:s}/train.npy'.format(self.__class__.__name__), train_dists)

        logger.info('Transforming valid')
        np.save('cache/decomp_feature/{:s}/valid.npy'.format(self.__class__.__name__), self.apply_model('valid'))
        logger.info('Transforming merge')
        np.save('cache/decomp_feature/{:s}/merge.npy'.format(self.__class__.__name__), self.apply_model('merge'))
        logger.info('Transforming test')
        np.save('cache/decomp_feature/{:s}/test.npy'.format(self.__class__.__name__), self.apply_model('test'))
        logger.info('Done')

        self.testicles()

    def testicles(self):
        X = self._load_named('train')
        y = dataset.Dataset().load_named('train').is_duplicate.values

        cls = lightgbm.LGBMClassifier(num_leaves=512, n_estimators=500)
        cls.fit(X.values, y)
        X_test = self._load_named('valid').values
        y_test = dataset.Dataset().load_named('valid').is_duplicate.values
        y_pred = cls.predict_proba(X_test)[:, 1]

        scoring = core.score_data(y_test, y_pred)
        importances = pandas.Series(cls.feature_importances_, index=X.columns)
        print(scoring)
        print(importances)
        with self.output().open('w') as f:
            f.write("Score: {:f}\n".format(scoring))
            f.write(str(importances))

    def load_named(self, name):
        assert self.complete()
        return self._load_named(name)

    def _load_named(self, name):
        fname = 'cache/decomp_feature/{:s}/{:s}.npy'.format(self.__class__.__name__, name)
        data = np.load(fname, mmap_mode='r')
        diff_cols = ['diff_{:d}'.format(v) for v in range(data.shape[1] - 4)]
        other_cols = ['euclidean', 'cityblock', 'cosine', 'correlation']
        return pandas.DataFrame(data, columns=diff_cols + other_cols)


class LDADecompositionFeatureVectors(DecompositionFeatureVectors):
    def make_decomposer(self):
        return decomposition.LatentDirichletAllocation(n_jobs=-4)


class NMFDecompositionFeatureVectors(DecompositionFeatureVectors):
    def make_decomposer(self):
        return decomposition.NMF(n_components=10)
