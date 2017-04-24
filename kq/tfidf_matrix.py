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

        return sp.hstack([q1 + q2, q1.multiply(q2), np.abs(q1 - q2)])

    def transform(self, X):
        tokens1 = X.question1_raw.apply(self.vectorize_question)
        tokens2 = X.question2_raw.apply(self.vectorize_question)
        all_tokens = np.concatenate([tokens1, tokens2], 0)

        transformed_tokens = self.vectorizer.transform(all_tokens)
        q1 = transformed_tokens[:transformed_tokens.shape[0] // 2]
        q2 = transformed_tokens[transformed_tokens.shape[0] // 2:]

        return sp.hstack([q1 + q2, q1.multiply(q2), np.abs(q1 - q2)])

    def run(self):
        self.output().makedirs()
        tqdm.pandas(tqdm)
        self.tokenzier = treebank.TreebankWordTokenizer()
        self.stemmer = snowball.SnowballStemmer('english')
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=10)
        train, merge, valid = Dataset().load()

        logging.info('Vectorizing train')
        train_mat = self.fit(train)
        scipy.io.mmwrite('cache/tfidf/train.mtx', train_mat)
        del train, train_mat

        logging.info('Vectorizing valid')
        valid_mat = self.transform(valid)
        scipy.io.mmwrite('cache/tfidf/valid.mtx', valid_mat)
        del valid, valid_mat

        logging.info('Vectorizing merge')
        merge_mat = self.transform(merge)
        scipy.io.mmwrite('cache/tfidf/merge.mtx', merge_mat)
        del merge, merge_mat

        logging.info('Vectorizing test')
        test = Dataset().load_test()
        test_mat = self.transform(test)
        scipy.io.mmwrite('cache/tfidf/test.mtx', test_mat)

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
            return scipy.io.mmread('cache/tfidf/train.mtx')
        elif name == 'valid':
            return scipy.io.mmread('cache/tfidf/valid.mtx')
        elif name == 'merge':
            return scipy.io.mmread('cache/tfidf/merge.mtx')
        elif name == 'test':
            return scipy.io.mmread('cache/tfidf/test.mtx')