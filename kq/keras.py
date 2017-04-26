import glob

import keras
import nltk
import pandas
import spacy
import numpy as np
from plumbum import colors
import os
from kq import core, dataset
import luigi

class DatasetIterator:
    def __init__(self, train_mode, tokenizer, English, dataset, vec_len, batch_size=128):
        self.train_mode = train_mode
        self.vec_len = vec_len
        self.dataset = dataset
        self.English = English
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.batch = 0
        self.n_batches = int(np.ceil(self.dataset.shape[0] / batch_size))
        self.ordering = np.random.permutation(self.dataset.shape[0])

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches

    def text_transform(self, text):
        tokens = self.tokenizer.tokenize(text.lower())
        vecs = np.zeros((1, self.vec_len, 300))
        i = 0
        for tok in tokens:
            lex = self.English.vocab[tok]
            if not lex.is_oov and not lex.is_punct and not lex.is_stop:
                vecs[0, i, :] = lex.vector
                i += 1
            if i == self.vec_len:
                break
        if i < self.vec_len:
            # Move the zeros to the beginning of the sequence.
            np.roll(vecs, self.vec_len, 1)
        return vecs

    def __next__(self):
        start = self.batch_size * self.batch
        end = min(start + self.batch_size, self.dataset.shape[0])
        batch_data = self.dataset.iloc[self.ordering[start:end]]
        self.batch += 1
        if self.batch >= self.n_batches:
            self.batch = 0
            self.ordering = np.random.permutation(self.dataset.shape[0])

        try:
            v1 = np.concatenate(batch_data.question1_raw.apply(self.text_transform).values, 0).astype(np.float32)
            v2 = np.concatenate(batch_data.question2_raw.apply(self.text_transform).values, 0).astype(np.float32)
            y = batch_data.is_duplicate.astype(np.int32).values
            weights = core.weights[y]
        except ValueError:
            import ipdb; ipdb.set_trace()
        if self.train_mode:
            return [v1, v2], y, weights
        else:
            return [v1, v2]


class KerasModel(luigi.Task):
    vec_len = 31
    nb_epoch = 1

    def requires(self):
        return dataset.Dataset()

    def output(self):
        return luigi.LocalTarget('cache/keras/classifications.npy')


    def model(self):
        num_lstm = np.random.randint(175, 275)
        num_dense = np.random.randint(100, 150)
        rate_drop_lstm = 0.15 + np.random.rand() * 0.25
        rate_drop_dense = 0.15 + np.random.rand() * 0.25

        lstm_layer = keras.layers.LSTM(
            num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

        sequence_1_input = keras.layers.Input(shape=(self.vec_len, 300))
        x1 = lstm_layer(sequence_1_input)

        sequence_2_input = keras.layers.Input(shape=(self.vec_len, 300))
        x2 = lstm_layer(sequence_2_input)

        merged = keras.layers.concatenate([x1, x2])
        merged = keras.layers.Dropout(rate_drop_dense)(merged)
        merged = keras.layers.BatchNormalization()(merged)

        merged = keras.layers.Dense(num_dense, activation='relu')(merged)
        merged = keras.layers.Dropout(rate_drop_dense)(merged)
        merged = keras.layers.BatchNormalization()(merged)

        preds = keras.layers.Dense(1, activation='sigmoid')(merged)
        model = keras.models.Model(
            inputs=[sequence_1_input, sequence_2_input], outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

        return model

    def run(self):
        self.output().makedirs()
        self.tokenizer = nltk.TreebankWordTokenizer()
        self.en = spacy.en.English()


        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        model = self.model()

        valid_data = dataset.Dataset().load_named('valid')
        train_data = dataset.Dataset().load_named('train')
        train_iter = DatasetIterator(True, self.tokenizer, self.en, train_data, self.vec_len)
        valid_iter = DatasetIterator(True, self.tokenizer, self.en, valid_data, self.vec_len)

        hist = model.fit_generator(train_iter, len(train_iter) // 5,
                                   self.nb_epoch, callbacks=[early_stopping],
                                   validation_steps=32, validation_data=valid_iter)

        valid_iter = DatasetIterator(False, self.tokenizer, self.en, valid_data, self.vec_len)
        valid_len = len(valid_iter)
        valid_pred = model.predict_generator(valid_iter, valid_len)
        loss = core.score_data(valid_data.is_duplicate, valid_pred)
        print(colors.green | "Performance (single): " + str(loss))

        valid_iter = DatasetIterator(False, self.tokenizer, self.en, valid_data, self.vec_len)
        valid_pred += model.predict_generator(map(lambda a: (a[1], a[0]), valid_iter), valid_len)
        valid_pred = valid_pred / 2
        loss = core.score_data(valid_data.is_duplicate, valid_pred)
        print(colors.green | colors.bold | "Performance (doubleback): " + str(loss))


        merge_iter = DatasetIterator(False, self.tokenizer, self.en, dataset.Dataset().load_named('merge'), self.vec_len)
        merge_len = len(merge_iter)
        merge_pred1 = model.predict_generator(merge_iter, merge_len, verbose=1)[:, 0]
        merge_iter = map(
            lambda a: (a[1], a[0]),
            DatasetIterator(False, self.tokenizer, self.en, dataset.Dataset().load_named('merge'), self.vec_len))
        merge_pred2 = model.predict_generator(merge_iter, merge_len, verbose=1)[:, 0]
        merge_pred = (merge_pred1 + merge_pred2) / 2

        np.save('cache/keras/merge.npy', merge_pred)

        test_iter = DatasetIterator(False, self.tokenizer, self.en, dataset.Dataset().load_named('test'), self.vec_len)
        test_len = len(test_iter)
        test_pred = model.predict_generator(test_iter, test_len, verbose=1)[:, 0]
        test_iter = DatasetIterator(False, self.tokenizer, self.en, dataset.Dataset().load_named('test'), self.vec_len)
        test_pred += model.predict_generator(test_iter, test_len, verbose=1)[:, 0]
        test_pred = test_pred / 2

        np.save('cache/keras/classifications_tmp.npy', test_pred)
        os.rename('cache/keras/classifications_tmp.npy', 'cache/keras/classifications.npy')

    @staticmethod
    def load():
        assert KerasModel().complete()
        return np.load('cache/keras/merge.npy')

    @staticmethod
    def load_test():
        assert KerasModel().complete()
        return np.load('cache/keras/classifications.npy')


class KaggleKeras(luigi.Task):
    def complete(self):
        return len(glob.glob('cache/kagglekeras/*.csv')) > 0

    def load(self):
        assert self.complete()
        return np.load('cache/kagglekeras/merge.npy')

    def load_test(self):
        assert self.complete()
        cols = []
        for csv in glob.glob('cache/kagglekeras/*.csv'):
            cols.append(pandas.read_csv(csv).is_duplicate.values[:, None])
        mat = np.concatenate(cols, 1)
        return mat.mean(1)
