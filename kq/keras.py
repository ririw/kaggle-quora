import glob

import keras
import luigi
import numpy as np
import pandas
from plumbum import colors

from kq import core, keras_kaggle_data, distances


class KerasModel(luigi.Task):
    resources = {'cpu': 8}

    force = luigi.BoolParameter(default=False)

    def requires(self):
        yield keras_kaggle_data.KaggleDataset()
        yield distances.AllDistances()

    def output(self):
        if self.force:
            return False
        return luigi.LocalTarget('cache/%s/classifications.npy' % self.base_name)

    def load_dataset(self, name):
        [d1, d2], labels = keras_kaggle_data.KaggleDataset().load(name)
        ds = distances.AllDistances().load_named(name)
        return [d1, d2, ds], labels

    def run(self):
        self.output().makedirs()
        train_data, train_labels = self.load_dataset('train')
        valid_data, valid_labels = self.load_dataset('valid')
        valid_weights = core.weights[valid_labels]
        class_weights = dict(enumerate(core.weights))
        embedding = keras_kaggle_data.KaggleDataset().load_embedding()

        model = self.model(embedding,
                           keras_kaggle_data.KaggleDataset().MAX_SEQUENCE_LENGTH,
                           train_data[2].shape[1])
        model.summary()

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

        model.fit(
            train_data, train_labels,
            validation_data=(valid_data, valid_labels, valid_weights),
            epochs=200, batch_size=2048, shuffle=True,
            class_weight=class_weights, callbacks=[early_stopping])

        valid_preds = model.predict(valid_data, batch_size=1024)
        print(colors.green | ('Valid loss: %f ' % core.score_data(valid_labels, valid_preds)))
        valid_preds += model.predict([valid_data[1], valid_data[0]. valid_data[2]], batch_size=1024)
        valid_preds /= 2
        print(colors.green | colors.bold | ('Valid loss (crossed): %f' % core.score_data(valid_labels, valid_preds)))
        del valid_labels, valid_data
        del train_labels, train_data

        merge_data, merge_labels = self.load_dataset('merge')
        merge_preds = model.predict(merge_data, batch_size=1024)
        merge_preds += model.predict([merge_data[1], merge_data[0], merge_data[2]], batch_size=1024)
        merge_preds /= 2

        np.save('cache/%s/merge.npy' % self.base_name, merge_preds)

        test_data, _ = self.load_dataset('test')
        test_preds = model.predict(test_data, batch_size=1024)
        test_preds += model.predict([test_data[1], test_data[0], test_data[2]], batch_size=1024)
        test_preds /= 2

        np.save('cache/%s/classifications.npy' % self.base_name, test_preds)

    def load(self):
        assert self.complete()
        return np.load('cache/%s/merge.npy' % self.base_name)

    def load_test(self):
        assert self.complete()
        return np.load('cache/%s/classifications.npy' % self.base_name)

class KerasLSTMModel(KerasModel):
    base_name = "keras_lstm"

    def model(self, embedding_matrix, vec_len, distance_width):
        num_lstm = np.random.randint(175, 275)
        num_dense = np.random.randint(100, 150)
        rate_drop_lstm = 0.15 + np.random.rand() * 0.25
        rate_drop_dense = 0.15 + np.random.rand() * 0.25


        embedding_layer = keras.layers.Embedding(
            embedding_matrix.shape[0], embedding_matrix.shape[1],
            weights=[embedding_matrix], input_length=vec_len, trainable=False)

        lstm_layer1 = keras.layers.Bidirectional(
            keras.layers.LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)
        )

        sequence_1_input = keras.layers.Input(shape=[vec_len], dtype='int32')
        embed_1 = embedding_layer(sequence_1_input)
        x1 = lstm_layer1(embed_1)

        sequence_2_input = keras.layers.Input(shape=[vec_len], dtype='int32')
        embed_2 = embedding_layer(sequence_2_input)
        y1 = lstm_layer1(embed_2)

        distance_input = keras.layers.Input(shape=[distance_width])
        di = keras.layers.Dense(num_dense, activation='relu')(distance_input)
        di = keras.layers.Dropout(num_lstm)(di)
        di = keras.layers.BatchNormalization()(di)

        di = keras.layers.Dense(num_dense, activation='relu')(di)
        di = keras.layers.Dropout(num_lstm)(di)
        di = keras.layers.BatchNormalization()(di)

        merged = keras.layers.concatenate([x1, y1, di])
        merged = keras.layers.Dropout(rate_drop_dense)(merged)
        merged = keras.layers.BatchNormalization()(merged)

        merged = keras.layers.Dense(num_dense, activation='relu')(merged)
        merged = keras.layers.Dropout(rate_drop_dense)(merged)
        merged = keras.layers.BatchNormalization()(merged)

        preds = keras.layers.Dense(1, activation='sigmoid')(merged)
        model = keras.models.Model(inputs=[sequence_1_input, sequence_2_input, distance_input], outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

        return model


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


