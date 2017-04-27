import glob
import os
import uuid
from collections import Counter

import keras
import luigi
import nltk
import numpy as np
import pandas
import spacy
from plumbum import colors
from tqdm import tqdm

from kq import core, dataset


class DatasetIterator:
    def __init__(self, train_mode, tokenizer, English, dataset, vec_len, batch_size=128, reverse_order=False, vocab=None):
        self.reverse_order = reverse_order
        self.train_mode = train_mode
        self.vec_len = vec_len
        self.dataset = dataset
        self.English = English
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.batch = 0
        self.n_batches = int(np.ceil(self.dataset.shape[0] / batch_size))
        self.ordering = np.random.permutation(self.dataset.shape[0])
        self.vocab = vocab

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
            if (self.vocab is not None and tok in self.vocab or
                    (not lex.is_oov and not lex.is_punct and not lex.is_stop)):
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

        v1 = np.concatenate(batch_data.question1_raw.apply(self.text_transform).values, 0).astype(np.float32)
        v2 = np.concatenate(batch_data.question2_raw.apply(self.text_transform).values, 0).astype(np.float32)
        if not self.train_mode:
            if self.reverse_order:
                return [v2, v1]
            else:
                return [v1, v2]
        else:
            y = batch_data.is_duplicate.astype(np.int32).values
            weights = core.weights[y]
            if self.reverse_order:
                return [v2, v1], y, weights
            else:
                return [v1, v2], y, weights


class KerasConvModel(luigi.Task):
    vec_len = 31
    nb_epoch = 1000

    def requires(self):
        return dataset.Dataset()

    def complete(self):
        return bool(glob.glob('cache/keras_conv/*/*.test.npy'))

    def model(self):
        input1 = keras.layers.Input(shape=[self.vec_len, 300])
        input2 = keras.layers.Input(shape=[self.vec_len, 300])

        conv = keras.models.Sequential(name='input_convolution')
        conv.add(keras.layers.Conv1D(300, 3, input_shape=[self.vec_len, 300]))
        conv.add(keras.layers.MaxPool1D())
        conv.add(keras.layers.PReLU())
        conv.add(keras.layers.Conv1D(150, 5))
        conv.add(keras.layers.MaxPool1D())
        conv.add(keras.layers.PReLU())
        conv.add(keras.layers.Flatten())

        conv1 = conv(input1)
        conv2 = conv(input2)

        merged_vecs = keras.layers.concatenate([conv1, conv2])

        merge_model = keras.models.Sequential(name='merge_transformations')
        merge_model.add(keras.layers.Dropout(0.25, input_shape=[1500]))
        merge_model.add(keras.layers.BatchNormalization())
        merge_model.add(keras.layers.Dense(300))
        merge_model.add(keras.layers.PReLU())

        merge_model.add(keras.layers.Dropout(0.25))
        merge_model.add(keras.layers.BatchNormalization())
        merge_model.add(keras.layers.Dense(100))
        merge_model.add(keras.layers.PReLU())

        merge_model.add(keras.layers.Dropout(0.25))
        merge_model.add(keras.layers.BatchNormalization())
        merge_model.add(keras.layers.Dense(1, activation='sigmoid'))

        res = merge_model(merged_vecs)

        model = keras.models.Model([input1, input2], res)
        model.compile('adam', 'binary_crossentropy')

        model.summary()

        return model

    def run(self):
        run_id = str(uuid.uuid4())
        os.makedirs('cache/keras_conv/%s' % run_id, exist_ok=True)

        self.tokenizer = nltk.TreebankWordTokenizer()
        self.en = spacy.en.English()

        model = self.model()

        valid_data = dataset.Dataset().load_named('valid')
        train_data = dataset.Dataset().load_named('train')
        train_iter = DatasetIterator(True, self.tokenizer, self.en, train_data, self.vec_len)
        valid_iter = DatasetIterator(True, self.tokenizer, self.en, valid_data, self.vec_len)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        hist = model.fit_generator(train_iter, len(train_iter),
                                   self.nb_epoch, callbacks=[early_stopping],
                                   validation_steps=32, validation_data=valid_iter)

        valid_iter = DatasetIterator(False, self.tokenizer, self.en, valid_data, self.vec_len)
        valid_len = len(valid_iter)
        valid_pred = model.predict_generator(valid_iter, valid_len)
        loss = core.score_data(valid_data.is_duplicate, valid_pred)
        print(colors.green | "Performance (single): " + str(loss))

        valid_iter = DatasetIterator(False, self.tokenizer, self.en, valid_data, self.vec_len)
        valid_pred += model.predict_generator(map(lambda a: [a[1], a[0]], valid_iter), valid_len)
        valid_pred = valid_pred / 2
        loss = core.score_data(valid_data.is_duplicate, valid_pred)
        print(colors.green | colors.bold | "Performance (doubleback): " + str(loss))

        merge_data = dataset.Dataset().load_named('merge')
        merge_iter = DatasetIterator(False, self.tokenizer, self.en, merge_data, self.vec_len)
        merge_pred1 = model.predict_generator(merge_iter, len(merge_iter), verbose=1)[:, 0]
        merge_iter = DatasetIterator(False, self.tokenizer, self.en, merge_data, self.vec_len, reverse_order=True)
        merge_pred2 = model.predict_generator(merge_iter, len(merge_iter), verbose=1)[:, 0]
        merge_pred = (merge_pred1 + merge_pred2) / 2

        np.save('cache/keras_conv/%s/merge.npy' % run_id, merge_pred)

        test_data = dataset.Dataset().load_named('test')
        test_iter = DatasetIterator(False, self.tokenizer, self.en, test_data, self.vec_len)
        test_len = len(test_iter)
        test_pred = model.predict_generator(test_iter, test_len, verbose=1)[:, 0]
        np.save('cache/keras_conv/%s/classifications1.npy' % run_id, test_pred)
        test_iter = DatasetIterator(False, self.tokenizer, self.en, test_data, self.vec_len, reverse_order=True)
        test_pred += model.predict_generator(test_iter, test_len, verbose=1)[:, 0]
        test_pred = test_pred / 2

        np.save('cache/keras_conv/%s/classifications_tmp.npy' % run_id, test_pred)
        os.rename('cache/keras_conv/%s/classifications_tmp.npy' % run_id,
                  'cache/keras_conv/%s/classifications.npy' % run_id)


class KerasLSTMModel(luigi.Task):
    vec_len = 31
    nb_epoch = 1000
    force = luigi.BoolParameter()

    def requires(self):
        return dataset.Dataset()

    def output(self):
        if self.force:
            return False
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
        model = keras.models.Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

        model.summary()

        return model

    def count_vocab(self, dataset):
        counter = Counter()
        for qn in tqdm(dataset.question1_raw, total=dataset.shape[0]):
            for word in self.tokenizer.tokenize(qn):
                lex = self.en.vocab[word]
                if lex.is_alpha and not lex.is_stop and not lex.is_oov:
                    counter[word] += 1
        for qn in tqdm(dataset.question2_raw, total=dataset.shape[0]):
            for word in self.tokenizer.tokenize(qn):
                lex = self.en.vocab[word]
                if lex.is_alpha and not lex.is_stop and not lex.is_oov:
                    counter[word] += 1
        return counter


    def run(self):
        self.output().makedirs()
        self.tokenizer = nltk.TreebankWordTokenizer()
        self.en = spacy.en.English()
        ds = dataset.Dataset()
        vocab_count = self.count_vocab(ds.load_named('train'))
        self.vocab = {w for w, _ in vocab_count.most_common(200000)}

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        model = self.model()

        valid_data = ds.load_named('valid')
        train_data = ds.load_named('train')
        train_iter = DatasetIterator(True, self.tokenizer, self.en, train_data, self.vec_len)
        valid_iter = DatasetIterator(True, self.tokenizer, self.en, valid_data, self.vec_len)

        hist = model.fit_generator(train_iter, len(train_iter),
                                   self.nb_epoch, callbacks=[early_stopping],
                                   validation_steps=32, validation_data=valid_iter)

        valid_iter = DatasetIterator(False, self.tokenizer, self.en, valid_data, self.vec_len)
        valid_len = len(valid_iter)
        valid_pred = model.predict_generator(valid_iter, valid_len)
        loss = core.score_data(valid_data.is_duplicate, valid_pred)
        print(colors.green | "Performance (single): " + str(loss))

        valid_iter = DatasetIterator(False, self.tokenizer, self.en, valid_data, self.vec_len)
        valid_pred += model.predict_generator(map(lambda a: [a[1], a[0]], valid_iter), valid_len)
        valid_pred = valid_pred / 2
        loss = core.score_data(valid_data.is_duplicate, valid_pred)
        print(colors.green | colors.bold | "Performance (doubleback): " + str(loss))


        merge_iter = DatasetIterator(False, self.tokenizer, self.en, ds.load_named('merge'), self.vec_len)
        merge_len = len(merge_iter)
        merge_pred1 = model.predict_generator(merge_iter, merge_len, verbose=1)[:, 0]
        merge_iter = map(
            lambda a: [a[1], a[0]],
            DatasetIterator(False, self.tokenizer, self.en, ds.load_named('merge'), self.vec_len))
        merge_pred2 = model.predict_generator(merge_iter, merge_len, verbose=1)[:, 0]
        merge_pred = (merge_pred1 + merge_pred2) / 2

        np.save('cache/keras/merge.npy', merge_pred)

        test_iter = DatasetIterator(False, self.tokenizer, self.en, ds.load_named('test'), self.vec_len)
        test_len = len(test_iter)
        test_pred = model.predict_generator(test_iter, test_len, verbose=1)[:, 0]
        test_iter = map(
            lambda a: [a[1], a[0]],
            DatasetIterator(False, self.tokenizer, self.en, ds.load_named('test'), self.vec_len)
        )

        test_pred += model.predict_generator(test_iter, test_len, verbose=1)[:, 0]
        test_pred = test_pred / 2

        np.save('cache/keras/classifications_tmp.npy', test_pred)
        os.rename('cache/keras/classifications_tmp.npy', 'cache/keras/classifications.npy')

    @staticmethod
    def load():
        assert KerasLSTMModel().complete()
        return np.load('cache/keras/merge.npy')

    @staticmethod
    def load_test():
        assert KerasLSTMModel().complete()
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


