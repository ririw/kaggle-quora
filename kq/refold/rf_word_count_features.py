import multiprocessing

import luigi
import numpy as np
import scipy.io
import scipy.sparse as sp
from nltk.stem import snowball
from nltk.tokenize import treebank
from plumbum import colors
from sklearn.feature_extraction.text import CountVectorizer

from kq.feat_abhishek import FoldIndependent
from kq.refold import rf_dataset

__all__ = ['WordCountMatrix']
class WordCountMatrix(FoldIndependent):
    resources = {'cpu': 8, 'mem': 3}
    ngram_max = luigi.IntParameter(default=3)
    ngram_min_df = luigi.FloatParameter(default=0.001)

    def _load(self):
        feat = scipy.io.mmread('rf_cache/rf_word_count_features/ng_{:d}_df_{:f}/train_mat.mtx'.format(self.ngram_max, self.ngram_min_df))
        fold = rf_dataset.Dataset().load_dataset_folds()
        return feat.tocsr(), fold

    def _load_test(self):
        feat = scipy.io.mmread('rf_cache/rf_word_count_features/ng_{:d}_df_{:f}/test_mat.mtx'.format(self.ngram_max, self.ngram_min_df))
        return feat.tocsr()

    def requires(self):
        yield rf_dataset.Dataset()

    def output(self):
        return luigi.LocalTarget('rf_cache/rf_word_count_features/ng_{:d}_df_{:f}/done_count'.format(self.ngram_max, self.ngram_min_df))

    def vectorize_question(self, q):
        tokens = self.tokenzier.tokenize(q)
        subtokens = [self.stemmer.stem(w) for w in tokens]
        return ' '.join(subtokens)

    def run(self):
        self.output().makedirs()
        self.tokenzier = treebank.TreebankWordTokenizer()
        self.stemmer = snowball.SnowballStemmer('english')
        self.vectorizer = CountVectorizer(ngram_range=(1, self.ngram_max), min_df=self.ngram_min_df)
        train_data = rf_dataset.Dataset().load('train', fold=None, as_np=False)
        test_data = rf_dataset.Dataset().load('test', fold=None, as_np=False)

        all_questions = np.concatenate([
            train_data.question1.values,
            test_data.question1.values,

            train_data.question2.values,
            test_data.question2.values
        ])

        print(colors.lightblue | 'Tokenizing')
        all_tokens = multiprocessing.Pool(4).map(self.vectorize_question, all_questions)
        print(colors.lightblue | 'Finished tokenizing, now fitting')
        transformed_tokens = self.vectorizer.fit_transform(all_tokens)
        print(colors.lightblue | colors.bold | 'Gosh that takes a long time')
        transformed_tokens = transformed_tokens.tocsr()

        halfpt = transformed_tokens.shape[0]//2
        assert halfpt == train_data.shape[0] + test_data.shape[0]
        q1s = transformed_tokens[:halfpt]
        q2s = transformed_tokens[halfpt:]

        diffs = sp.hstack([q1s + q2s, q1s.multiply(q2s)]).tocsr()

        train_vecs = diffs[:train_data.shape[0]]
        test_vecs = diffs[train_data.shape[0]:]
        assert train_vecs.shape[0] == train_data.shape[0]
        assert test_vecs.shape[0] == test_data.shape[0]
        scipy.io.mmwrite('rf_cache/rf_word_count_features/ng_{:d}_df_{:f}/train_mat.mtx'.format(self.ngram_max, self.ngram_min_df), train_vecs)
        scipy.io.mmwrite('rf_cache/rf_word_count_features/ng_{:d}_df_{:f}/test_mat.mtx'.format(self.ngram_max, self.ngram_min_df), test_vecs)
        with self.output().open('w'):
            pass