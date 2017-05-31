import multiprocessing
import os

import luigi
import pandas
import scipy.spatial.distance
import spacy.en
import numpy as np
import nose.tools
from tqdm import tqdm

from kq.feat_abhishek import FoldIndependent
from kq.refold import BaseTargetBuilder, rf_dataset

__all__ = ['RF_POS_Distance']


class RF_POS_Distance(FoldIndependent):
    resources = {'cpu': 1}

    def make_path(self, fname):
        base_path = BaseTargetBuilder('rf_pos_dist')
        return (base_path + fname).get()

    def _load(self, as_df):
        res = np.load(self.output().path)['train']
        if as_df:
            res = pandas.DataFrame(res, columns=['ent', 'nent', 'ent_diff', 'ent_ratio'])
        folds = rf_dataset.Dataset().load_dataset_folds()
        return res, folds

    def _load_test(self, as_df):
        res = np.load(self.output().path)['test']
        if as_df:
            res = pandas.DataFrame(res, columns=['ent', 'nent', 'ent_diff', 'ent_ratio'])
        return res

    def requires(self):
        yield rf_dataset.Dataset()

    def output(self):
        return luigi.LocalTarget(self.make_path('done.npz'))

    def entity_diffs(self, q12):
        q1, q2 = q12
        doc1 = self.English(q1.lower())
        doc2 = self.English(q2.lower())

        doc1_ent_vecs = [np.zeros(300)]
        doc1_other_vecs = [np.zeros(300)]
        for tok in doc1:
            if tok.ent_type != 0:
                doc1_ent_vecs.append(tok.vector)
            else:
                doc1_other_vecs.append(tok.vector)
        doc2_ent_vecs = [np.zeros(300)]
        doc2_other_vecs = [np.zeros(300)]
        for tok in doc2:
            if tok.ent_type != 0:
                doc2_ent_vecs.append(tok.vector)
            else:
                doc2_other_vecs.append(tok.vector)

        mean_ent_1 = np.mean(np.asarray(doc1_ent_vecs), 0)
        mean_other_1 = np.mean(np.asarray(doc1_other_vecs), 0)
        mean_ent_2 = np.mean(np.asarray(doc2_ent_vecs), 0)
        mean_other_2 = np.mean(np.asarray(doc2_other_vecs), 0)

        ent_diff = scipy.spatial.distance.euclidean(mean_ent_1, mean_ent_2)
        nent_diff = scipy.spatial.distance.euclidean(mean_other_1, mean_other_2)

        return [
            ent_diff,
            nent_diff,
            np.abs(nent_diff - ent_diff),
            ent_diff / (nent_diff + 1),
        ]

    def run(self):
        self.English = spacy.en.English()
        train_data = rf_dataset.Dataset().load_all('train')
        test_data = rf_dataset.Dataset().load_all('test')

        train_q12 = zip(train_data.question1_clean, train_data.question2_clean)
        test_q12 = zip(test_data.question1_clean, test_data.question2_clean)
        all_ent_train =[self.entity_diffs(v) for v in tqdm(train_q12, total=train_data.shape[0], desc='ents - train')]
        all_ent_test =[self.entity_diffs(v) for v in tqdm(test_q12, total=test_data.shape[0], desc='ents - test')]

        all_ent_train = np.asarray(all_ent_train)
        all_ent_test = np.asarray(all_ent_test)
        nose.tools.assert_equal(all_ent_train.shape[1], all_ent_test.shape[1])
        nose.tools.assert_equal(all_ent_train.shape[0], train_data.shape[0])
        nose.tools.assert_equal(all_ent_test.shape[0], test_data.shape[0])


        self.output().makedirs()
        np.savez_compressed(self.make_path('done_tmp.npz'), train=all_ent_train, test=all_ent_test)
        os.rename(self.make_path('done_tmp.npz'), self.output().path)
