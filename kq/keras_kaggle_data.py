import re

import luigi
import numpy as np
import spacy
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk import SnowballStemmer
from tqdm import tqdm

from kq import dataset

__all__ = ['KaggleDataset']

class KaggleDataset(luigi.Task):
    resources = {'cpu': 2}

    MAX_SEQUENCE_LENGTH = 32
    max_nb_words = 250000

    def requires(self):
        return dataset.Dataset()

    def output(self):
        return luigi.LocalTarget('cache/kaggledata/done')

    def run(self):
        def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
            # Clean the text, with the option to remove stopwords and to stem words.

            # Convert words to lower case and split them
            text = text.lower().split()

            text = " ".join(text)

            # Clean the text
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
            text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
            text = re.sub(r":", " : ", text)
            text = re.sub(r" e g ", " eg ", text)
            text = re.sub(r" b g ", " bg ", text)
            text = re.sub(r" u s ", " american ", text)
            text = re.sub(r"\0s", "0", text)
            text = re.sub(r" 9 11 ", "911", text)
            text = re.sub(r"e - mail", "email", text)
            text = re.sub(r"j k", "jk", text)
            text = re.sub(r"\s{2,}", " ", text)

            # Optionally, shorten words to their stems
            if stem_words:
                text = text.split()
                stemmer = SnowballStemmer('english')
                stemmed_words = [stemmer.stem(word) for word in text]
                text = " ".join(stemmed_words)

            # Return a list of words
            return (text)

        train_texts_1 = []
        train_texts_2 = []
        train_labels = []

        train_dataset = dataset.Dataset().load_named('train')
        for _, row in tqdm(train_dataset.iterrows(), total=train_dataset.shape[0]):
            train_texts_1.append(text_to_wordlist(row.question1_raw))
            train_texts_2.append(text_to_wordlist(row.question2_raw))
            train_labels.append(row.is_duplicate)
        print('Found %s texts in train.csv' % len(train_texts_1))

        valid_texts_1 = []
        valid_texts_2 = []
        valid_labels = []

        valid_dataset = dataset.Dataset().load_named('valid')
        for _, row in tqdm(valid_dataset.iterrows(), total=valid_dataset.shape[0]):
            valid_texts_1.append(text_to_wordlist(row.question1_raw))
            valid_texts_2.append(text_to_wordlist(row.question2_raw))
            valid_labels.append(row.is_duplicate)
        print('Found %s texts in valid.csv' % len(valid_texts_1))

        merge_texts_1 = []
        merge_texts_2 = []
        merge_labels = []

        merge_dataset = dataset.Dataset().load_named('merge')
        for _, row in tqdm(merge_dataset.iterrows(), total=merge_dataset.shape[0]):
            merge_texts_1.append(text_to_wordlist(row.question1_raw))
            merge_texts_2.append(text_to_wordlist(row.question2_raw))
            merge_labels.append(row.is_duplicate)
        print('Found %s texts in merge.csv' % len(merge_texts_1))

        test_texts_1 = []
        test_texts_2 = []

        test_dataset = dataset.Dataset().load_named('test')
        for _, row in tqdm(test_dataset.iterrows(), total=test_dataset.shape[0]):
            test_texts_1.append(text_to_wordlist(row.question1_raw))
            test_texts_2.append(text_to_wordlist(row.question2_raw))
        print('Found %s texts in test.csv' % len(test_texts_1))

        tokenizer = Tokenizer(num_words=self.max_nb_words)
        tokenizer.fit_on_texts(train_texts_1 + valid_texts_1 + merge_texts_1 + test_texts_1 +
                               train_texts_2 + valid_texts_2 + merge_texts_2 + test_texts_2)

        train_sequences_1 = tokenizer.texts_to_sequences(train_texts_1)
        train_sequences_2 = tokenizer.texts_to_sequences(train_texts_2)
        valid_sequences_1 = tokenizer.texts_to_sequences(valid_texts_1)
        valid_sequences_2 = tokenizer.texts_to_sequences(valid_texts_2)
        merge_sequences_1 = tokenizer.texts_to_sequences(merge_texts_1)
        merge_sequences_2 = tokenizer.texts_to_sequences(merge_texts_2)
        test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
        test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

        word_index = tokenizer.word_index
        print('Found %s unique tokens' % len(word_index))

        train_data_1 = pad_sequences(train_sequences_1, maxlen=self.MAX_SEQUENCE_LENGTH)
        train_data_2 = pad_sequences(train_sequences_2, maxlen=self.MAX_SEQUENCE_LENGTH)
        valid_data_1 = pad_sequences(valid_sequences_1, maxlen=self.MAX_SEQUENCE_LENGTH)
        valid_data_2 = pad_sequences(valid_sequences_2, maxlen=self.MAX_SEQUENCE_LENGTH)
        merge_data_1 = pad_sequences(merge_sequences_1, maxlen=self.MAX_SEQUENCE_LENGTH)
        merge_data_2 = pad_sequences(merge_sequences_2, maxlen=self.MAX_SEQUENCE_LENGTH)
        test_data_1 = pad_sequences(test_sequences_1, maxlen=self.MAX_SEQUENCE_LENGTH)
        test_data_2 = pad_sequences(test_sequences_2, maxlen=self.MAX_SEQUENCE_LENGTH)

        train_labels = np.array(train_labels)
        valid_labels = np.array(valid_labels)
        merge_labels = np.array(merge_labels)

        English = spacy.en.English()
        embedding_matrix = np.zeros((self.max_nb_words, 300))
        for word, i in word_index.items():
            lex = English.vocab[word]
            if not lex.is_oov:
                embedding_matrix[i] = lex.vector
        print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

        self.output().makedirs()
        np.save('cache/kaggledata/train_1.npy', train_data_1)
        np.save('cache/kaggledata/train_2.npy', train_data_2)
        np.save('cache/kaggledata/train_labels.npy', train_labels)

        np.save('cache/kaggledata/valid_1.npy', valid_data_1)
        np.save('cache/kaggledata/valid_2.npy', valid_data_2)
        np.save('cache/kaggledata/valid_labels.npy', valid_labels)

        np.save('cache/kaggledata/merge_1.npy', merge_data_1)
        np.save('cache/kaggledata/merge_2.npy', merge_data_2)
        np.save('cache/kaggledata/merge_labels.npy', merge_labels)

        np.save('cache/kaggledata/test_1.npy', test_data_1)
        np.save('cache/kaggledata/test_2.npy', test_data_2)

        np.save('cache/kaggledata/embedding.npy', embedding_matrix)

        with self.output().open('w'):
            pass

    def load_named(self, name, load_only=None):
        assert self.complete()
        assert name in {'train', 'test', 'valid', 'merge'}
        f1 = np.load('cache/kaggledata/%s_1.npy' % name, mmap_mode='r').astype(np.int64)
        f2 = np.load('cache/kaggledata/%s_2.npy' % name, mmap_mode='r').astype(np.int64)
        if name == 'test':
            l = np.zeros(f2.shape[0]) - 1
        else:
            l = np.load('cache/kaggledata/%s_labels.npy' % name, mmap_mode='r')
        if load_only:
            f1 = f1[:load_only]
            f2 = f2[:load_only]
            l = l[:load_only]

        return ([f1, f2], l)

    def load_embedding(self):
        assert self.complete()
        return np.load('cache/kaggledata/embedding.npy', mmap_mode='r').astype(np.float32)