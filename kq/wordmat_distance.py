import luigi

from kq import count_matrix, tfidf_matrix
import numpy as np
import scipy.sparse as sp
from sklearn import decomposition

class WordMatDistance(luigi.Task):
    def requires(self):
        yield count_matrix.CountFeature()
        yield tfidf_matrix.TFIDFFeature()

    def output(self):
        return luigi.LocalTarget('cache/word_mat_distances/done')

    def shared_mean_transform(self, m1, m2):
        return np.mean(np.sqrt(m1.multiply(m2)), 1)

    def sparse_svm_transform(self, m1, m2, transformer):
        stackmat = sp.vstack([m1, m2])
        vecs = transformer.transform(stackmat)
        u1 = vecs[:m1.shape[0]]
        u2 = vecs[m1.shape[0]:]
        return np.concatenate([u1, u2], 1)

    def train_decompositions(self):
        m1, m2 = count_matrix.CountFeature().load_full_mats('train')
        stackmat = sp.vstack([m1, m2])
        self.pca_count = decomposition.TruncatedSVD(6)
        self.pca_count.fit(stackmat)

        m1, m2 = tfidf_matrix.TFIDFFeature().load_full_mats('train')
        stackmat = sp.vstack([m1, m2])
        self.pca_tfidf = decomposition.TruncatedSVD(6)
        self.pca_tfidf.fit(stackmat)

    def run(self):
        self.output().makedirs()
        self.train_decompositions()
        self.run_ds('train')
        self.run_ds('merge')
        self.run_ds('valid')
        self.run_ds('test')

        with self.output().open('w'):
            pass

    def run_ds(self, dataset):
        cm1, cm2 = count_matrix.CountFeature().load_full_mats(dataset)
        cd = self.shared_mean_transform(cm1, cm2)
        cu = self.sparse_svm_transform(cm1, cm2, self.pca_count)

        tm1, tm2 = tfidf_matrix.TFIDFFeature().load_full_mats(dataset)
        td = self.shared_mean_transform(tm1, tm2)
        tu =  self.sparse_svm_transform(tm1, tm2, self.pca_tfidf)

        print(type(cd), type(cu), type(td), type(tu))

        full_mat = np.concatenate([cd, cu, td, tu], 1)

        np.save('cache/word_mat_distances/%s.npy' % dataset, full_mat)

    def load(self, dataset):
        assert self.complete()
        assert dataset in {'train', 'test', 'merge', 'valid'}
        return np.load('cache/word_mat_distances/%s.npy' % dataset)