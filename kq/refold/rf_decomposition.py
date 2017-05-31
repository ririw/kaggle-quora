import multiprocessing

import luigi
import numpy as np
import pandas
import scipy.sparse as sp
from sklearn import decomposition, base
from tqdm import tqdm

from kq.feat_abhishek import FoldIndependent
import kq.core
from kq.refold import rf_word_count_features, BaseTargetBuilder, rf_vectorspaces, rf_dataset

__all__ = ['LDADecomposition', 'NMFDecomposition', 'AllDecompositions']


class Decomposition(FoldIndependent):
    def _load_test(self, as_df):
        feat = kq.core.fillna(np.load(self.make_path('test.npz'))['data'], 9999).clip(-10000, 10000)
        if as_df:
            return pandas.DataFrame(feat, columns=self.columns())
        else:
            return feat

    def _load(self, as_df):
        folds = rf_dataset.Dataset().load_dataset_folds()
        feat = kq.core.fillna(np.load(self.make_path('train.npz'))['data'], 9999).clip(-10000, 10000)
        if as_df:
            return pandas.DataFrame(feat, columns=self.columns()), folds
        else:
            return feat, folds

    def requires(self):
        yield rf_word_count_features.WordCountMatrix()

    def output(self):
        return luigi.LocalTarget(self.make_path('done'))

    def decomp_dist(self, v12):
        v1, v2 = v12
        d = np.hstack([
            np.asarray([distance(v1, v2) for distance in rf_vectorspaces.distances]),
            np.abs(v1 - v2)
        ])
        return d

    def columns(self):
        return ['euclidean',
                'sqeuclidean',
                'cityblock',
                'cosine',
                'correlation',
                'chebyshev',
                'canberra',
                'braycurtis'] + ['decomp_{:s}_{:0d}'.format(self.__class__.__name__, i)
                                 for i in range(self.n_components())]

    def run(self):
        self.output().makedirs()
        wc_mat_train = rf_word_count_features.WordCountMatrix().load_raw_vectors('train')
        wc_mat_test = rf_word_count_features.WordCountMatrix().load_raw_vectors('test')

        all_vecs = sp.vstack(list(wc_mat_train) + list(wc_mat_test))
        decomp = self.decomposition()
        decomposed = decomp.fit_transform(all_vecs)
        train_size = wc_mat_train[0].shape[0]
        test_size = wc_mat_test[0].shape[0]
        train_decomp = decomposed[:train_size * 2]
        test_decomp = decomposed[train_size * 2:]
        assert test_decomp.shape[0] == test_size * 2
        decomp_train = {
            'q1': train_decomp[:train_size],
            'q2': train_decomp[train_size:]
        }
        decomp_test = {
            'q1': test_decomp[:test_size],
            'q2': test_decomp[test_size:]
        }

        train_dists = list(tqdm(multiprocessing.Pool().imap(
            self.decomp_dist, zip(decomp_train['q1'], decomp_train['q2']), chunksize=50_000),
            total=train_size,
            desc='vectorizing the training data'
        ))

        test_dists = list(tqdm(multiprocessing.Pool().imap(
            self.decomp_dist, zip(decomp_test['q1'], decomp_test['q2']), chunksize=50_000),
            total=test_size,
            desc='vectorizing the testing data'
        ))

        train_dists = np.asarray(train_dists)
        test_dists = np.asarray(test_dists)

        print('Decomposition finished, shape of train result is: ' + str(train_dists.shape))
        print('Decomposition finished, shape of test result is: ' + str(test_dists.shape))

        np.savez_compressed(self.make_path('train.npz'), data=train_dists)
        np.savez_compressed(self.make_path('test.npz'), data=test_dists)
        with self.output().open('w'):
            pass

    def decomposition(self) -> base.TransformerMixin:
        raise NotImplementedError

    def n_components(self) -> int:
        raise NotImplementedError

    def make_path(self, fname):
        raise NotImplementedError


class NMFDecomposition(Decomposition):
    def n_components(self) -> int:
        return 10

    def decomposition(self) -> base.TransformerMixin:
        return decomposition.NMF(n_components=self.n_components())

    def make_path(self, fname):
        base_path = BaseTargetBuilder('rf_decompositions', 'nmf')
        return (base_path + fname).get()


class SVDDecomposition(Decomposition):
    def n_components(self) -> int:
        return 10

    def decomposition(self) -> base.TransformerMixin:
        return decomposition.TruncatedSVD(n_components=self.n_components())

    def make_path(self, fname):
        base_path = BaseTargetBuilder('rf_decompositions', 'svd')
        return (base_path + fname).get()


class LDADecomposition(Decomposition):
    def n_components(self) -> int:
        return 10

    def decomposition(self) -> base.TransformerMixin:
        return decomposition.LatentDirichletAllocation(n_topics=self.n_components())

    def make_path(self, fname):
        base_path = BaseTargetBuilder('rf_decompositions', 'lda')
        return (base_path + fname).get()


class AllDecompositions(FoldIndependent):
    def requires(self):
        yield SVDDecomposition()
        yield NMFDecomposition()

    def complete(self):
        for r in self.requires():
            if not r.complete():
                return False
        return True

    def _load_test(self, as_df):
        decomps = [r._load_test(as_df) for r in self.requires()]
        if as_df:
            for decomp, req in zip(decomps, self.requires()):
                self.wangjangle_columns(decomp, req)
            decomps = pandas.concat(decomps, 1)
        else:
            decomps = np.concatenate(decomps, 1)
        return decomps

    def _load(self, as_df):
        decomps = [r._load(as_df)[0] for r in self.requires()]
        if as_df:
            for d, r in zip(decomps, self.requires()):
                self.wangjangle_columns(d, r)
            decomps = pandas.concat(decomps, 1)
        else:
            decomps = np.concatenate(decomps, 1)
        folds = rf_dataset.Dataset().load_dataset_folds()

        return decomps, folds

    def wangjangle_columns(self, data, req):
        data.cols = [req.__class__.__name__ + c for c in data.columns]
