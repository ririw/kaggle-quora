import multiprocessing

import gensim
import luigi
import pandas
import spacy
from nltk.tokenize import treebank
from scipy import spatial
from tqdm import tqdm

from kq.feat_abhishek import FoldIndependent
from kq.refold import rf_dataset, BaseTargetBuilder
import numpy as np

from kq.utils import w2v_file

distances = [
    spatial.distance.euclidean,
    spatial.distance.sqeuclidean,
    spatial.distance.cityblock,
    spatial.distance.cosine,
    spatial.distance.correlation,
    spatial.distance.chebyshev,
    spatial.distance.canberra,
    spatial.distance.braycurtis]
English = None


class VectorSpaceTask(FoldIndependent):
    include_space = luigi.BoolParameter()

    def requires(self):
        yield rf_dataset.Dataset()

    def make_path(self, fname):
        base_path = BaseTargetBuilder('rf_vecspace')
        return (base_path + fname).get()

    def _load_test(self, as_df):
        res = np.load(self.make_path('test.npz'))['data']
        if self.include_space:
            res = res[:len(distances)]
            cols = self.colnames()[:len(distances)]
        else:
            cols = self.colnames()

        if as_df:
            res = pandas.DataFrame(res, columns=cols)
        return res

    def _load(self, as_df):
        assert not as_df
        res = np.load(self.make_path('train.npz'))['data']
        if self.include_space:
            res = res[:len(distances)]
            cols = self.colnames()[:len(distances)]
        else:
            cols = self.colnames()

        if as_df:
            res = pandas.DataFrame(res, columns=cols)
        folds = rf_dataset.Dataset().load_dataset_folds()
        return res, folds

    def output(self):
        return luigi.LocalTarget(self.make_path('done'))

    def vectorize(self, q12):
        q1, q2 = q12
        k1 = [self.kvecs[w.lower()] for w in self.tokenzier.tokenize(q1) if w.lower() in self.kvecs]
        k2 = [self.kvecs[w.lower()] for w in self.tokenzier.tokenize(q2) if w.lower() in self.kvecs]
        if len(k1) == 0 or len(k2) == 0:
            return np.ones(300+len(distances)) * 1000

        v1 = np.mean(k1, 0)
        v2 = np.mean(k2, 0)
        d = np.hstack([
            np.asarray([distance(v1, v2) for distance in distances]),
            np.abs(v1 - v2)
        ])
        return d

    @staticmethod
    def colnames():
        return ['euclidean',
                'sqeuclidean',
                'cityblock',
                'cosine',
                'correlation',
                'chebyshev',
                'canberra',
                'braycurtis'] + ['space_{:00d}'.format(i) for i in range(300)]

    def run(self):
        self.tokenzier = treebank.TreebankWordTokenizer()
        self.kvecs = gensim.models.KeyedVectors.load_word2vec_format(w2v_file)

        train_data = rf_dataset.Dataset().load_all('train', as_df=True)[['question1_clean', 'question2_clean']]
        test_data = rf_dataset.Dataset().load_all('test', as_df=True)[['question1_clean', 'question2_clean']]

        all_data = pandas.concat([train_data, test_data], 0)

        distances = list(tqdm(multiprocessing.Pool().imap(
            self.vectorize, zip(all_data['question1_clean'], all_data['question2_clean']), chunksize=50_000),
            total=all_data.shape[0],
            desc='vectorizing the words'
        ))
        distances = np.asarray(distances).astype(np.float16)

        self.output().makedirs()
        train_dists = distances[:train_data.shape[0]]
        test_dists = distances[train_data.shape[0]:]
        np.savez_compressed(self.make_path('train.npz'), data=train_dists)
        np.savez_compressed(self.make_path('test.npz'), data=test_dists)

        with self.output().open('w'):
            pass
