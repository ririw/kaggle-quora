import logging
import luigi
import scipy.io
import scipy.sparse as sp
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from nltk.tokenize import treebank
from nltk.stem import snowball
from tqdm import tqdm

from kq.dataset import Dataset


class TFIDFFeature(luigi.Task):
    resources = {'cpu': 1}

    English = None
    def requires(self):
        return Dataset()

    def output(self):
        return luigi.LocalTarget('cache/tfidf/done')

    def vectorize_question(self, question_string):
        tokens = self.tokenzier.tokenize(question_string)
        subtokens = [self.stemmer.stem(w) for w in tokens]
        return ' '.join(subtokens)

    def fit(self, train_data):
        tokens1 = train_data.question1_raw.progress_apply(self.vectorize_question)
        tokens2 = train_data.question2_raw.progress_apply(self.vectorize_question)
        all_tokens = np.concatenate([tokens1, tokens2], 0)

        transformed_tokens = self.vectorizer.fit_transform(all_tokens)
        q1 = transformed_tokens[:transformed_tokens.shape[0]//2]
        q2 = transformed_tokens[transformed_tokens.shape[0]//2:]

        return sp.hstack([q1 + q2, q1.multiply(q2), np.abs(q1 - q2)]), q1, q2

    def transform(self, X):
        tokens1 = X.question1_raw.apply(self.vectorize_question)
        tokens2 = X.question2_raw.apply(self.vectorize_question)
        all_tokens = np.concatenate([tokens1, tokens2], 0)

        transformed_tokens = self.vectorizer.transform(all_tokens)
        q1 = transformed_tokens[:transformed_tokens.shape[0] // 2]
        q2 = transformed_tokens[transformed_tokens.shape[0] // 2:]

        return sp.hstack([q1 + q2, q1.multiply(q2), np.abs(q1 - q2)]), q1, q2

    def run(self):
        self.output().makedirs()
        tqdm.pandas(tqdm)
        self.tokenzier = treebank.TreebankWordTokenizer()
        self.stemmer = snowball.SnowballStemmer('english')
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=50)
        train, merge, valid = Dataset().load()

        logging.info('Vectorizing train')
        train_mat, q1, q2 = self.fit(train)
        train_cols = train_mat.shape[1]
        train_q1_cols, train_q2_cols = q1.shape[1], q2.shape[1]
        scipy.io.mmwrite('cache/tfidf/train.mtx', train_mat)
        scipy.io.mmwrite('cache/tfidf/train_q1.mtx', q1)
        scipy.io.mmwrite('cache/tfidf/train_q2.mtx', q2)
        del train, train_mat

        logging.info('Vectorizing valid')
        valid_mat, q1, q2 = self.transform(valid)
        assert valid_mat.shape[1] == train_cols
        assert q1.shape[1] == train_q1_cols and q2.shape[1] == train_q2_cols
        scipy.io.mmwrite('cache/tfidf/valid.mtx', valid_mat)
        scipy.io.mmwrite('cache/tfidf/valid_q1.mtx', q1)
        scipy.io.mmwrite('cache/tfidf/valid_q2.mtx', q2)
        del valid, valid_mat

        logging.info('Vectorizing merge')
        merge_mat, q1, q2 = self.transform(merge)
        assert merge_mat.shape[1] == train_cols
        assert q1.shape[1] == train_q1_cols and q2.shape[1] == train_q2_cols
        scipy.io.mmwrite('cache/tfidf/merge.mtx', merge_mat)
        scipy.io.mmwrite('cache/tfidf/merge_q1.mtx', q1)
        scipy.io.mmwrite('cache/tfidf/merge_q2.mtx', q2)
        del merge, merge_mat

        logging.info('Vectorizing test')
        test = Dataset().load_test()
        test_mat, q1, q2 = self.transform(test)
        assert test_mat.shape[1] == train_cols
        assert q1.shape[1] == train_q1_cols and q2.shape[1] == train_q2_cols
        scipy.io.mmwrite('cache/tfidf/test.mtx', test_mat)
        scipy.io.mmwrite('cache/tfidf/test_q1.mtx', q1)
        scipy.io.mmwrite('cache/tfidf/test_q2.mtx', q2)

        assert self.load_dataset('test').shape[1] == train_cols

        with self.output().open('w') as f:
            pass

    @staticmethod
    def load():
        tfidf = TFIDFFeature()
        assert tfidf.complete()
        train = tfidf.load_dataset('train')
        valid = tfidf.load_dataset('valid')
        merge = tfidf.load_dataset('merge')
        return train, merge, valid

    @staticmethod
    def load_test():
        tfidf = TFIDFFeature()
        assert tfidf.complete()
        return tfidf.load_dataset('test')

    @staticmethod
    def load_dataset(name):
        assert name in {'train', 'merge', 'valid', 'test'}, 'Name %s was not one of train/test/merge/valid' % name
        if name == 'train':
            return scipy.io.mmread('cache/tfidf/train.mtx').tocsr()
        elif name == 'valid':
            return scipy.io.mmread('cache/tfidf/valid.mtx').tocsr()
        elif name == 'merge':
            return scipy.io.mmread('cache/tfidf/merge.mtx').tocsr()
        elif name == 'test':
            return scipy.io.mmread('cache/tfidf/test.mtx').tocsr()

    @staticmethod
    def load_full_mats(name):
        assert name in {'train', 'merge', 'valid', 'test'}, 'Name %s was not one of train/test/merge/valid' % name
        if name == 'train':
            q1 = scipy.io.mmread('cache/tfidf/train_q1.mtx').tocsr()
            q2 = scipy.io.mmread('cache/tfidf/train_q2.mtx').tocsr()
        elif name == 'valid':
            q1 = scipy.io.mmread('cache/tfidf/valid_q1.mtx').tocsr()
            q2 = scipy.io.mmread('cache/tfidf/valid_q2.mtx').tocsr()
        elif name == 'merge':
            q1 = scipy.io.mmread('cache/tfidf/merge_q1.mtx').tocsr()
            q2 = scipy.io.mmread('cache/tfidf/merge_q2.mtx').tocsr()
        else:
            q1 = scipy.io.mmread('cache/tfidf/test_q1.mtx').tocsr()
            q2 = scipy.io.mmread('cache/tfidf/test_q2.mtx').tocsr()
        return q1, q2