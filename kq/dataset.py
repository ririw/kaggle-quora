"""
Datasources, with standardized train/test splits.
"""

import os.path

import luigi
import mmh3
import numpy as np
import pandas
import re

from nltk import SnowballStemmer
from nltk.corpus import stopwords


def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


class Dataset(luigi.Task):
    def output(self):
        return luigi.LocalTarget('./cache/dataset-done-%d' % self.seed)

    def run(self):
        kaggle_train_data = pandas.read_csv(os.path.expanduser('~/Datasets/Kaggle-Quora/train.csv'))
        num_pos = kaggle_train_data.is_duplicate.sum()
        num_neg = kaggle_train_data.shape[0] - kaggle_train_data.is_duplicate.sum()
        x = (num_pos - (0.165 * (num_pos + num_neg))) / 0.165
        x = int(x)
        # Pull in an extra x rows from the data,
        # then shuffle with the sample call
        # then reset the index, dropping the old
        resampled_data = kaggle_train_data.query('is_duplicate == 0').sample(x, replace=True)
        kaggle_train_data = pandas.concat([kaggle_train_data, resampled_data]) \
            .sample(frac=1.) \
            .reset_index(drop=True)
        assert 0.15 < (kaggle_train_data.is_duplicate.mean()) < 0.17

        kaggle_train_data['question1'] = kaggle_train_data['question1'].fillna('')
        kaggle_train_data['question2'] = kaggle_train_data['question2'].fillna('')
        # Make sure the valid set contains _zero_ question from the train set.
        a = kaggle_train_data.qid1.apply(lambda v: mmh3.hash(str(v).encode('ascii'), 2213))
        b = kaggle_train_data.qid2.apply(lambda v: mmh3.hash(str(v).encode('ascii'), 6663))
        train_data = kaggle_train_data[(a > 1) & (b > 1)].reset_index(drop=True)
        merge_data = kaggle_train_data[(a == 0) & (b == 0)].reset_index(drop=True)
        valid_data = kaggle_train_data[(a == 1) & (b == 1)].reset_index(drop=True)

        self.output().makedirs()
        train_data.to_msgpack('cache/dataset-train.msg')
        merge_data.to_msgpack('cache/dataset-merge.msg')
        valid_data.to_msgpack('cache/dataset-valid.msg')
        del train_data, valid_data, kaggle_train_data

        kaggle_test_data = pandas.read_csv(os.path.expanduser('~/Datasets/Kaggle-Quora/test.csv'))
        kaggle_test_data['question1'] = kaggle_test_data['question1'].fillna('')
        kaggle_test_data['question2'] = kaggle_test_data['question2'].fillna('')
        kaggle_test_data.to_msgpack('cache/dataset-test.msg')

        with self.output().open('w') as f:
            f.write('done')

    def load(self, only_get=None):
        train_data = pandas.read_msgpack('cache/dataset-train.msg')
        merge_data = pandas.read_msgpack('cache/dataset-merge.msg')
        valid_data = pandas.read_msgpack('cache/dataset-valid.msg')
        if only_get is not None:
            return train_data.head(only_get), merge_data.head(only_get), valid_data.head(only_get)
        else:
            return train_data, merge_data, valid_data

    def load_test(self, only_get=None):
        test_data = pandas.read_msgpack('cache/dataset-test.msg')
        if only_get is not None:
            return test_data.head(only_get)
        else:
            return test_data
