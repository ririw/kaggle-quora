import keras
import nltk
import spacy
import numpy as np
from plumbum import colors
import os
from collections import Counter
from kq import core, dataset
import luigi

class DatasetIterator:
    def __init__(self, tokenizer, stemmer, English, dataset, vec_len, batch_size=128):
        self.vec_len = vec_len
        self.dataset = dataset
        self.English = English
        self.stemmer = stemmer
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
        subtokens = [self.stemmer.stem(w) for w in tokens]
        vecs = np.zeros((1, self.vec_len, 300))
        i = 0
        for tok in subtokens:
            lex = self.English.vocab[tok]
            if not lex.is_oov and not lex.is_punct and not lex.is_stop:
                vecs[0, i, :] = lex.vector
                i += 1
            if i == self.vec_len:
                break
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
        except ValueError:
            import ipdb; ipdb.set_trace()

        return [v1, v2], y


class KerasModel(luigi.Task):
    vec_len = 31
    nb_epoch = 1000

    def requires(self):
        return dataset.Dataset()

    def output(self):
        return luigi.LocalTarget('cache/keras/classifications.npy')


    def model(self):
        nb_words = min(MAX_NB_WORDS, len(word_index)) + 1
        embedding_matrix = np.zeros((nb_words, 300))
        for ix, word in enumerate(self.vocab):
            lex = self.en.vocab[word]
            if not lex.is_oov:
                embedding_matrix[ix] = lex.vector
        print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))


        num_lstm = np.random.randint(175, 275)
        num_dense = np.random.randint(100, 150)
        rate_drop_lstm = 0.15 + np.random.rand() * 0.25
        rate_drop_dense = 0.15 + np.random.rand() * 0.25

        embedding_layer = keras.layers.Embedding(
            nb_words, 300, weights=[embedding_matrix],
            input_length=self.vec_len, trainable=False)
        lstm_layer = keras.layers.LSTM(
            num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

        sequence_1_input = keras.layers.Input(shape=(self.vec_len,), dtype=int32)
        embedded_sequences_1 = embedding_layer(sequence_1_input)
        x1 = lstm_layer(embedded_sequences_1)

        sequence_2_input = keras.layers.Input(shape=(self.vec_len,), dtype=int32)
        embedded_sequences_2 = embedding_layer(sequence_1_input)
        x2 = lstm_layer(embedded_sequences_2)

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

    def top_words(self, max_words=200000):
        word_counter = Counter()
        train = dataset.Dataset().load_named('train')
        test = dataset.Dataset().load_named('test')
        all_questions = np.concatenate([
            train.question1_raw.values,
            train.question2_raw.values,
            test.question1_raw.values,
            test.question2_raw.values,
        ])
        for sent in all_questions:
            tokens = self.tokenizer.tokenize(sent.lower())
            for w in [self.stemmer.stem(w) for w in tokens]:
                word_counter[w] += 1
        return [w[0] for w in word_counter.most_common(max_words)]

    def run(self):
        self.output().makedirs()
        self.tokenizer = nltk.TreebankWordTokenizer()
        self.stemmer = nltk.SnowballStemmer('english')
        self.en = spacy.en.English()

        self.vocab = self.top_words()

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

        model = self.model()

        valid_data = dataset.Dataset().load_named('valid')
        train_data = dataset.Dataset().load_named('train')
        train_iter = DatasetIterator(self.tokenizer, self.stemmer, self.en, train_data, self.vec_len)
        valid_iter = DatasetIterator(self.tokenizer, self.stemmer, self.en, valid_data, self.vec_len)

        hist = model.fit_generator(train_iter, len(train_iter) // 5,
                                   self.nb_epoch, callbacks=[early_stopping],
                                   validation_steps=32, validation_data=valid_iter)

        valid_iter = DatasetIterator(self.tokenizer, self.stemmer, self.en, valid_data, self.vec_len)
        valid_pred = model.predict_generator(valid_iter, len(valid_iter))
        loss = core.score_data(valid_data.is_duplicate, valid_pred)
        print(colors.green | "Performance: " + str(loss))


        merge_iter = DatasetIterator(self.tokenizer, self.stemmer, self.en, dataset.Dataset().load_named('merge'), self.vec_len)
        merge_pred = model.predict_generator(merge_iter, len(merge_iter), verbose=1)[:, 0]

        np.save('cache/keras/merge.npy', merge_pred)

        test_iter = DatasetIterator(self.tokenizer, self.stemmer, self.en, dataset.Dataset().load_named('test'), self.vec_len)
        test_pred = model.predict_generator(merge_iter, len(test_iter), verbose=1)[:, 0]

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
