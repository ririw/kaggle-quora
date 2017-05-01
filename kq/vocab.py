from collections import Counter

import luigi
import pandas
import tqdm
import numpy as np
from kq import dataset


class Vocab(luigi.Task):
    def requires(self):
        return dataset.Dataset()

    def output(self):
        return luigi.LocalTarget('./cache/vocab.msg')

    def run(self):
        train_data, _, _ = dataset.Dataset().load()
        vocab_count = Counter()
        for sent in tqdm.tqdm(train_data.question1_tokens,
                              desc='Counting questions one',
                              total=train_data.shape[0]):
            for tok in sent:
                vocab_count[tok] += 1

        for sent in tqdm.tqdm(train_data.question1_tokens,
                              desc='Counting questions two',
                              total=train_data.shape[0]):
            for tok in sent:
                vocab_count[tok] += 1

        vocab_counts = pandas.Series(vocab_count)
        self.output().makedirs()
        vocab_counts.to_msgpack(self.output().path)

    def load_vocab(self, min_occurances=10, max_words=None):
        assert self.complete()
        vocab_counts = pandas.read_msgpack('./cache/vocab.msg')
        admissible_vocab = vocab_counts[vocab_counts > min_occurances].copy()
        admissible_vocab.index = admissible_vocab.index.rename('word')
        admissible_vocab = admissible_vocab.to_frame('count').sort_values('count', ascending=False)
        admissible_vocab['word_id'] = np.arange(admissible_vocab.shape[0]) + 1
        return admissible_vocab