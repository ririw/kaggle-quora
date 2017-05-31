import gzip
import multiprocessing
import pickle

import luigi
import numpy as np
import scipy.io
import nose.tools
import scipy.sparse as sp
from nltk.stem import snowball
from nltk.tokenize import treebank
from plumbum import colors
from sklearn.feature_extraction.text import CountVectorizer

from kq.feat_abhishek import FoldIndependent
from kq.refold import rf_dataset, BaseTargetBuilder

__all__ = ['WordCountMatrix']


class WordCountMatrix(FoldIndependent):
    resources = {'cpu': 7, 'mem': 3}
    ngram_max = luigi.IntParameter(default=3)
    ngram_min_df = luigi.FloatParameter(default=0.0001)

    def _load(self, as_df):
        assert not as_df, 'Sparse matrix in word_count_features cannot be converted to dataframe'
        fn = self.make_path('train_mat.pkl')
        feat = self.read_mat_from(fn)
        fold = rf_dataset.Dataset().load_dataset_folds()
        return feat.tocsr(), fold

    def _load_test(self, as_df):
        assert not as_df, 'Sparse matrix in word_count_features cannot be converted to dataframe'
        fn = self.make_path('test_mat.pkl')
        feat = self.read_mat_from(fn)

        return feat.tocsr()

    def requires(self):
        yield rf_dataset.Dataset()

    def output(self):
        return luigi.LocalTarget(self.make_path('done_count'))

    def vectorize_question(self, q):
        tokens = self.tokenzier.tokenize(q)
        subtokens = [self.stemmer.stem(w) for w in tokens]
        return ' '.join(subtokens)

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_word_count_features',
            'ng_{:d}_df_{:f}'.format(self.ngram_max, self.ngram_min_df))
        return (base_path + fname).get()

    def run(self):
        self.output().makedirs()
        self.tokenzier = treebank.TreebankWordTokenizer()
        self.stemmer = snowball.SnowballStemmer('english')
        self.vectorizer = CountVectorizer(ngram_range=(1, self.ngram_max), min_df=self.ngram_min_df)
        train_data = rf_dataset.Dataset().load('train', fold=None, as_df=True)
        test_data = rf_dataset.Dataset().load('test', fold=None, as_df=True)

        all_questions = np.concatenate([
            train_data.question1_clean.values,
            test_data.question1_clean.values,

            train_data.question2_clean.values,
            test_data.question2_clean.values
        ])

        print(colors.lightblue | 'Tokenizing')
        all_tokens = multiprocessing.Pool(4).map(self.vectorize_question, all_questions)
        print(colors.lightblue | 'Finished tokenizing, now fitting')
        transformed_tokens = self.vectorizer.fit_transform(all_tokens)
        print(colors.lightblue | colors.bold | 'Gosh that takes a long time')
        transformed_tokens = transformed_tokens.tocsr()

        halfpt = transformed_tokens.shape[0] // 2
        assert halfpt == train_data.shape[0] + test_data.shape[0]
        q1s = transformed_tokens[:halfpt]
        q2s = transformed_tokens[halfpt:]

        train_q1s = q1s[:train_data.shape[0]]
        train_q2s = q2s[:train_data.shape[0]]
        test_q1s = q1s[train_data.shape[0]:]
        test_q2s = q2s[train_data.shape[0]:]
        nose.tools.assert_equal(test_q1s.shape[0], test_data.shape[0])
        nose.tools.assert_equal(test_q2s.shape[0], test_data.shape[0])
        nose.tools.assert_equal(train_q1s.shape[0], train_data.shape[0])
        nose.tools.assert_equal(train_q2s.shape[0], train_data.shape[0])

        self.write_mat_to(self.make_path('train_q1.pkl'), train_q1s)
        self.write_mat_to(self.make_path('train_q2.pkl'), train_q2s)
        self.write_mat_to(self.make_path('test_q1.pkl'), test_q1s)
        self.write_mat_to(self.make_path('test_q2.pkl'), test_q2s)

        diffs = sp.hstack([np.abs(q1s - q2s), q1s.multiply(q2s)]).tocsr()

        train_vecs = diffs[:train_data.shape[0]]
        test_vecs = diffs[train_data.shape[0]:]
        nose.tools.assert_equal(train_vecs.shape[0], train_data.shape[0])
        nose.tools.assert_equal(test_vecs.shape[0], test_data.shape[0])

        self.write_mat_to(self.make_path('train_mat.pkl'), train_vecs)
        self.write_mat_to(self.make_path('test_mat.pkl'), test_vecs)

        with self.output().open('w'):
            pass

    def write_mat_to(self, fname, mat):
        with gzip.open(fname, 'w') as f:
            pickle.dump(mat, f)

    def read_mat_from(self, fname):
        with gzip.open(fname) as f:
            return pickle.load(f)

    def load_raw_vectors(self, name):
        assert name in {'train', 'test'}

        q1 = self.read_mat_from(self.make_path('{}_q1.pkl'.format(name)))
        q2 = self.read_mat_from(self.make_path('{}_q2.pkl'.format(name)))

        return q1, q2