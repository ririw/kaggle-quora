import logging

import luigi
import numpy as np
import scipy.io
import scipy.sparse as sp
from nltk.stem import snowball
from nltk.tokenize import treebank
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from kq.dataset import Dataset

__all__ = ['CountFeature']

class CountFeature(luigi.Task):
    resources = {'cpu': 1}
    English = None
    def requires(self):
        return Dataset()

    def output(self):
        return luigi.LocalTarget('cache/count/done')

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
        self.vectorizer = CountVectorizer(ngram_range=(1,2), min_df=50)
        train, merge, valid = Dataset().load()

        logging.info('Vectorizing train')
        train_mat, q1, q2 = self.fit(train)
        scipy.io.mmwrite('cache/count/train.mtx', train_mat)
        scipy.io.mmwrite('cache/count/train_q1.mtx', q1)
        scipy.io.mmwrite('cache/count/train_q2.mtx', q2)
        del train, train_mat

        logging.info('Vectorizing valid')
        valid_mat, q1, q2 = self.transform(valid)
        scipy.io.mmwrite('cache/count/valid.mtx', valid_mat)
        scipy.io.mmwrite('cache/count/valid_q1.mtx', q1)
        scipy.io.mmwrite('cache/count/valid_q2.mtx', q2)

        del valid, valid_mat

        logging.info('Vectorizing merge')
        merge_mat, q1, q2 = self.transform(merge)
        scipy.io.mmwrite('cache/count/merge.mtx', merge_mat)
        scipy.io.mmwrite('cache/count/merge_q1.mtx', q1)
        scipy.io.mmwrite('cache/count/merge_q2.mtx', q2)
        del merge, merge_mat

        logging.info('Vectorizing test')
        test = Dataset().load_test()
        test_mat, q1, q2 = self.transform(test)
        scipy.io.mmwrite('cache/count/test.mtx', test_mat)
        scipy.io.mmwrite('cache/count/test_q1.mtx', q1)
        scipy.io.mmwrite('cache/count/test_q2.mtx', q2)

        with self.output().open('w') as f:
            pass

    @staticmethod
    def load():
        counts = CountFeature()
        assert counts.complete()
        train = counts.load_dataset('train')
        valid = counts.load_dataset('valid')
        merge = counts.load_dataset('merge')
        return train, merge, valid

    @staticmethod
    def load_test():
        counts = CountFeature()
        assert counts.complete()
        return counts.load_dataset('test')

    @staticmethod
    def load_dataset(name):
        assert name in {'train', 'merge', 'valid', 'test'}, 'Name %s was not one of train/test/merge/valid' % name
        if name == 'train':
            return scipy.io.mmread('cache/count/train.mtx').tocsr()
        elif name == 'valid':
            return scipy.io.mmread('cache/count/valid.mtx').tocsr()
        elif name == 'merge':
            return scipy.io.mmread('cache/count/merge.mtx').tocsr()
        elif name == 'test':
            return scipy.io.mmread('cache/count/test.mtx').tocsr()

    @staticmethod
    def load_full_mats(name):
        assert name in {'train', 'merge', 'valid', 'test'}, 'Name %s was not one of train/test/merge/valid' % name
        if name == 'train':
            q1 = scipy.io.mmread('cache/count/train_q1.mtx').tocsr()
            q2 = scipy.io.mmread('cache/count/train_q2.mtx').tocsr()
        elif name == 'valid':
            q1 = scipy.io.mmread('cache/count/valid_q1.mtx').tocsr()
            q2 = scipy.io.mmread('cache/count/valid_q2.mtx').tocsr()
        elif name == 'merge':
            q1 = scipy.io.mmread('cache/count/merge_q1.mtx').tocsr()
            q2 = scipy.io.mmread('cache/count/merge_q2.mtx').tocsr()
        elif name == 'test':
            q1 = scipy.io.mmread('cache/count/test_q1.mtx').tocsr()
            q2 = scipy.io.mmread('cache/count/test_q2.mtx').tocsr()
        return q1, q2