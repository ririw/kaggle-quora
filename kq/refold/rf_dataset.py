"""
Datasources, with standardized train/test splits.
"""
import gzip
import os.path

import luigi
import mmh3
import numpy as np
import pandas
import re

import spacy
from tqdm import tqdm

from kq.feat_abhishek import FoldIndependent

__all__ = ['Dataset']

def clean_text(text):
    text = text.lower()
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


class Dataset(FoldIndependent):
    resources = {'cpu': 1}

    def _load(self, as_df):
        with gzip.open('rf_cache/dataset/train.msg.gz', 'r') as f:
            if as_df:
                td = pandas.read_msgpack(f)
            else:
                td = pandas.read_msgpack(f).values

        f = np.load('rf_cache/folds.npz')['data']

        return td, f

    def load_dataset_folds(self):
        return np.load('rf_cache/folds.npz')['data']

    def _load_test(self, as_df):
        with gzip.open('rf_cache/dataset/test.msg.gz', 'r') as f:
            if as_df:
                return pandas.read_msgpack(f)
            else:
                return pandas.read_msgpack(f).values

    def output(self):
        return luigi.LocalTarget('./rf_cache/dataset/done')

    def run(self):
        self.output().makedirs()
        tqdm.pandas(tqdm)

        kaggle_train_data = pandas.read_csv(os.path.expanduser('~/Datasets/Kaggle-Quora/train.csv')).drop('id', 1)

        kaggle_train_data['question1'] = kaggle_train_data['question1'].fillna('')
        kaggle_train_data['question2'] = kaggle_train_data['question2'].fillna('')
        kaggle_train_data['question1_clean'] = kaggle_train_data['question1'].progress_apply(clean_text)
        kaggle_train_data['question2_clean'] = kaggle_train_data['question2'].progress_apply(clean_text)

        q1_fold = kaggle_train_data.qid1.apply(lambda qid: mmh3.hash(str(qid)) % 3)
        q2_fold = kaggle_train_data.qid2.apply(lambda qid: mmh3.hash(str(qid)) % 2 * 3)
        fold_n = (q1_fold + q2_fold).values

        with gzip.open('rf_cache/dataset/train.msg.gz', 'w') as f:
            kaggle_train_data.reset_index(drop=True).to_msgpack(f)
        np.savez_compressed('rf_cache/folds.npz', data=fold_n)
        del kaggle_train_data
        ########
        # TEST #
        ########
        kaggle_test_data = pandas.read_csv(os.path.expanduser('~/Datasets/Kaggle-Quora/test.csv'))
        kaggle_test_data['question1']       = kaggle_test_data['question1'].fillna('')
        kaggle_test_data['question2']       = kaggle_test_data['question2'].fillna('')
        kaggle_test_data['question1_clean'] = kaggle_test_data['question1'].progress_apply(clean_text)
        kaggle_test_data['question2_clean'] = kaggle_test_data['question2'].progress_apply(clean_text)
        kaggle_test_data['is_duplicate'] = -1

        with gzip.open('rf_cache/dataset/test.msg.gz', 'w') as f:
            kaggle_test_data.reset_index(drop=True).to_msgpack(f)

        with self.output().open('w') as f:
            f.write('done')
