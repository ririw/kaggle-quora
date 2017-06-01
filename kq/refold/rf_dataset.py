"""
Datasources, with standardized train/test splits.
"""
import gzip
import os.path

import luigi
import mmh3
import nose.tools
import numpy as np
import pandas
import re

import networkx as nx
from tqdm import tqdm

from kq.feat_abhishek import FoldIndependent
import kq.core

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


class GraphSynthesizingDataset(FoldIndependent):
    """
    Synthesize a dataset. Do so by:

    - Writing all questions to a graph
    - For each question:
      - If label 1, create entries for all question descendents
      - If label 0, create entries for all counter-question's descendents
    - Also add in 25% random question pairings, marked as not related.
    - This also builds out its own folds.
    """

    def _load_test(self, as_df):
        return Dataset().load_all('test', as_df=True)

    def _load(self, as_df):
        pass

    def requires(self):
        yield Dataset()

    def output(self):
        return luigi.LocalTarget('./rf_cache/synth_dataset/done')

    def run(self):
        train_data = Dataset().load_all('train', as_df=True)
        question_text = [(qid, qtext) for qid, qtext in zip(train_data.qid1, train_data.question1)] + \
                        [(qid, qtext) for qid, qtext in zip(train_data.qid2, train_data.question2)]
        question_text = dict(question_text)

        train_graph = nx.Graph()
        gid_counter = 0
        # Build a bipartite graph of questions
        q_iter = zip(train_data.qid1, train_data.qid2, train_data.is_duplicate)
        for qid1, qid2, is_dup in tqdm(q_iter, total=train_data.shape[0], desc='Reading data into graph'):
            if is_dup:
                if qid1 in train_graph and qid2 in train_graph:
                    if qid1 in set(nx.descendants(train_graph, qid2)):
                        continue
                    else:
                        gid = nx.neighbors(train_graph, qid1)[0]
                        assert gid.startswith('g_')
                        train_graph.add_edge(qid1, gid)
                if qid1 not in train_graph and qid2 not in train_graph:
                    gid = 'g_' + str(gid_counter)
                    gid_counter += 1
                    train_graph.add_edge(qid1, gid)
                    train_graph.add_edge(qid2, gid)
                if qid1 in train_graph:
                    gid = nx.neighbors(train_graph, qid1)[0]
                    assert gid.startswith('g_')
                    train_graph.add_edge(qid2, gid)
                if qid2 in train_graph:
                    gid = nx.neighbors(train_graph, qid2)[0]
                    assert gid.startswith('g_')
                    train_graph.add_edge(qid1, gid)
            else:
                if qid1 not in train_graph:
                    gid = 'g_' + str(gid_counter)
                    train_graph.add_edge(qid1, gid)
                    gid_counter += 1
                if qid2 not in train_graph:
                    gid = 'g_' + str(gid_counter)
                    train_graph.add_edge(qid2, gid)
                    gid_counter += 1

        synthetic_rows = {}
        dups_counts = np.asarray([0, 0])

        q_iter = zip(train_data.qid1, train_data.qid2, train_data.is_duplicate)
        for qid1, qid2, is_dup in tqdm(q_iter, total=train_data.shape[0], desc='Building synthetic data'):
            g1 = nx.neighbors(train_graph, qid1)[0]
            g2 = nx.neighbors(train_graph, qid2)[0]
            nose.tools.assert_is_instance(g1, str)
            nose.tools.assert_is_instance(g2, str)
            all_q1s = [qid for qid in nx.descendants(train_graph, g1) if not isinstance(qid, str)]
            all_q2s = [qid for qid in nx.descendants(train_graph, g2) if not isinstance(qid, str)]

            for qid1 in all_q1s:
                for qid2 in all_q2s:
                    if qid1 != qid2 and (qid1, qid2) not in synthetic_rows:
                        synthetic_rows[(qid1, qid2)] = {
                            'qid1': qid1,
                            'qid2': qid2,
                            'question1': question_text[qid1],
                            'question2': question_text[qid2],
                            'is_duplicate': is_dup
                        }
                        dups_counts[is_dup] += 1

        qids = np.asarray([v for v in question_text.keys()])
        n_required_to_balance = int(kq.core.weights[0] * dups_counts[1] / kq.core.weights[1] - dups_counts[0])
        for _ in tqdm(range(n_required_to_balance), desc='Building random pairs data'):
            qid1 = 0
            qid2 = 0
            while (qid1, qid2) in synthetic_rows or qid1 == qid2:
                qid1 = np.random.choice(qids)
                qid2 = np.random.choice(qids)
            synthetic_rows[(qid1, qid2)] = {
                'qid1': qid1,
                'qid2': qid2,
                'question1': question_text[qid1],
                'question2': question_text[qid2],
                'is_duplicate': 0
            }
            dups_counts[0] += 1

        for q1, q2 in zip(train_data.qid1, train_data.qid2):
            assert (q1, q2) in synthetic_rows

        print('Synthetic', dups_counts / dups_counts.sum())
        print('Real     ', kq.core.weights / kq.core.weights.sum())
        print(train_data.shape[0])