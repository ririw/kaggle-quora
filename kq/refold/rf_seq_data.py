import gensim
import luigi
import nose.tools
import numpy as np
import pandas
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import treebank

from kq.feat_abhishek import FoldIndependent, fold_max
from kq.refold import BaseTargetBuilder, rf_dataset, rf_small_features
from kq.utils import w2v_file

__all__ = ['RFWordSequenceDataset']


class RFWordSequenceDataset(FoldIndependent):
    resources = {'cpu': 1, 'mem': 2}

    def load_all(self, name, as_df=False, include_smallfeat=True):
        nose.tools.assert_in(name, {'train', 'test'})
        assert not as_df
        assert include_smallfeat
        if name == 'train':
            questions = np.load(self.make_path('train.npz'))
            smallfeat, _ = rf_small_features.SmallFeaturesTask()._load(False)
            return questions['q1'], questions['q2'], smallfeat
        else:
            questions = np.load(self.make_path('test.npz'))
            smallfeat = rf_small_features.SmallFeaturesTask()._load_test(False)
            return questions['q1'], questions['q2'], smallfeat

    def load(self, name, fold, as_df=False, include_smallfeat=True):
        assert self.complete()
        assert not as_df, 'Dataframe mode not supported'
        assert include_smallfeat, 'implement in load_all then remove assert.'
        assert name in {'train', 'test', 'valid'}
        if name == 'test':
            res = np.load(self.make_path('test.npz'))
            if include_smallfeat:
                smallfeat = rf_small_features.SmallFeaturesTask()._load_test(False)
                return res['q1'], res['q2'], smallfeat
            else:
                return res['q1'], res['q2']
        else:
            res = np.load(self.make_path('train.npz'))
            smallfeat, sm_folds = rf_small_features.SmallFeaturesTask()._load(False)
            folds = (rf_dataset.Dataset().load_dataset_folds() + fold) % fold_max
            if name == 'valid':
                selection = folds == 0
            else:
                selection = folds != 0
            if include_smallfeat:
                return res['q1'][selection], res['q2'][selection], smallfeat[selection]
            else:
                return res['q1'][selection], res['q2'][selection]

    def load_embedding_mat(self):
        assert self.complete()
        return np.load(self.make_path('embedding.npz'))['data']

    def make_path(self, fname):
        base_path = BaseTargetBuilder('rf_ab_sequences')
        return (base_path + fname).get()

    def requires(self):
        yield rf_dataset.Dataset()
        yield rf_small_features.SmallFeaturesTask()

    def output(self):
        return luigi.LocalTarget(self.make_path('done'))

    def run(self):
        self.output().makedirs()
        kvecs = gensim.models.KeyedVectors.load_word2vec_format(w2v_file)
        train_dataset = rf_dataset.Dataset().load_all('train', as_df=True)
        test_dataset = rf_dataset.Dataset().load_all('train', as_df=True)
        self.tokenzier = treebank.TreebankWordTokenizer()

        all_words = pandas.concat([
            train_dataset.question1_clean.str.lower(),
            train_dataset.question2_clean.str.lower(),
            test_dataset.question1_clean.str.lower(),
            test_dataset.question2_clean.str.lower(),
        ])

        tokenizer = Tokenizer(num_words=250_000)
        tokenizer.fit_on_texts(all_words)
        all_seqs = tokenizer.texts_to_sequences(all_words)
        all_padded_seqs = pad_sequences(all_seqs, 32)

        train_seqs = all_padded_seqs[:train_dataset.shape[0] * 2]
        test_seqs = all_padded_seqs[train_dataset.shape[0] * 2:]
        nose.tools.assert_equal(test_seqs.shape[0], test_dataset.shape[0] * 2)

        train_q1 = train_seqs[:train_dataset.shape[0]]
        train_q2 = train_seqs[train_dataset.shape[0]:]
        test_q1 = test_seqs[:test_dataset.shape[0]]
        test_q2 = test_seqs[test_dataset.shape[0]:]

        np.savez_compressed(self.make_path('train.npz'), q1=train_q1, q2=train_q2)
        np.savez_compressed(self.make_path('test.npz'), q1=test_q1, q2=test_q2)

        embedding_matrix = np.zeros((250_000, 300))
        for word, ix in tokenizer.word_index.items():
            if word in kvecs:
                embedding_matrix[ix, :] = kvecs[word]
        np.savez_compressed(self.make_path('embedding.npz'), data=embedding_matrix)

        with self.output().open('w'):
            pass
