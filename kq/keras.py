import glob

import keras
import luigi
import numpy as np
import pandas
from plumbum import colors

from kq import core, keras_kaggle_data


class KerasModel(luigi.Task):
    force = luigi.BoolParameter(default=False)

    def requires(self):
        return keras_kaggle_data.KaggleDataset()

    def output(self):
        if self.force:
            return False
        return luigi.LocalTarget('cache/%s/classifications.npy' % self.base_name)

    def run(self):
        self.output().makedirs()
        train_data, train_labels = self.requires().load('train')
        valid_data, valid_labels = self.requires().load('valid')
        valid_weights = core.weights[valid_labels]
        class_weights = dict(enumerate(core.weights))
        embedding = self.requires().load_embedding()

        model = self.model(embedding, self.requires().MAX_SEQUENCE_LENGTH)
        model.summary()

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

        model.fit(
            train_data, train_labels,
            validation_data=(valid_data, valid_labels, valid_weights),
            epochs=200, batch_size=2048, shuffle=True,
            class_weight=class_weights, callbacks=[early_stopping])

        valid_preds = model.predict(valid_data, batch_size=1024)
        print(colors.green | ('Valid loss: %f ' % core.score_data(valid_labels, valid_preds)))
        valid_preds += model.predict([valid_data[1], valid_data[0]], batch_size=1024)
        valid_preds /= 2
        print(colors.green | colors.bold | ('Valid loss (crossed): %f' % core.score_data(valid_labels, valid_preds)))
        del valid_labels, valid_data
        del train_labels, train_data

        merge_data, merge_labels = self.requires().load('merge')
        merge_preds = model.predict(merge_data, batch_size=1024)
        merge_preds += model.predict([merge_data[1], merge_data[0]], batch_size=1024)
        merge_preds /= 2

        np.save('cache/%s/merge.npy' % self.base_name, merge_preds)

        test_data, _ = self.requires().load('test', 10000)
        test_preds = model.predict(test_data, batch_size=1024)
        test_preds += model.predict([test_data[1], test_data[0]], batch_size=1024)
        test_preds /= 2

        np.save('cache/%s/classifications.npy' % self.base_name, test_preds)

    def load(self):
        assert self.complete()
        return np.load('cache/%s/merge.npy' % self.base_name)

    def load_test(self):
        assert self.complete()
        return np.load('cache/%s/classifications.npy' % self.base_name)

class KerasConvModel(KerasModel):
    base_name = "keras_conv"

    def model(self, embedding_matrix, vec_len):
        num_dense = np.random.randint(100, 150)
        rate_drop_dense = 0.15 + np.random.rand() * 0.25
        embedding_layer = keras.layers.Embedding(
            embedding_matrix.shape[0], embedding_matrix.shape[1],
            weights=[embedding_matrix], input_length=vec_len, trainable=False)
        conv1 = keras.layers.Conv1D(100, 3)
        conv2 = keras.layers.Conv1D(50, 5)

        convnet = keras.models.Sequential()
        convnet.add(embedding_layer)
        convnet.add(conv1)
        convnet.add(keras.layers.MaxPool1D())
        convnet.add(keras.layers.PReLU())
        convnet.add(conv2)
        convnet.add(keras.layers.MaxPool1D())
        convnet.add(keras.layers.PReLU())
        convnet.add(keras.layers.Flatten())

        sequence_1_input = keras.layers.Input(shape=[vec_len], dtype='int32')
        sequence_2_input = keras.layers.Input(shape=[vec_len], dtype='int32')
        x1 = convnet(sequence_1_input)
        x2 = convnet(sequence_2_input)

        merged = keras.layers.concatenate([x1, x2])
        merged = keras.layers.Dropout(rate_drop_dense)(merged)
        merged = keras.layers.BatchNormalization()(merged)

        merged = keras.layers.Dense(num_dense)(merged)
        merged = keras.layers.PReLU()(merged)
        merged = keras.layers.Dropout(rate_drop_dense)(merged)
        merged = keras.layers.BatchNormalization()(merged)

        preds = keras.layers.Dense(1, activation='sigmoid')(merged)
        model = keras.models.Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

        return model

class KerasLSTMModel(KerasModel):
    base_name = "keras_lstm"

    def model(self, embedding_matrix, vec_len):
        num_lstm = np.random.randint(175, 275)
        num_dense = np.random.randint(100, 150)
        rate_drop_lstm = 0.15 + np.random.rand() * 0.25
        rate_drop_dense = 0.15 + np.random.rand() * 0.25


        lstm_layer1 = keras.layers.Bidirectional(
            keras.layers.LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)
        )

        embedding_layer = keras.layers.Embedding(
            embedding_matrix.shape[0], embedding_matrix.shape[1],
            weights=[embedding_matrix], input_length=vec_len, trainable=False)

        sequence_1_input = keras.layers.Input(shape=[vec_len], dtype='int32')
        embed_1 = embedding_layer(sequence_1_input)
        x1 = lstm_layer1(embed_1)

        sequence_2_input = keras.layers.Input(shape=[vec_len], dtype='int32')
        embed_2 = embedding_layer(sequence_2_input)
        y1 = lstm_layer1(embed_2)

        merged = keras.layers.concatenate([x1, y1])
        merged = keras.layers.Dropout(rate_drop_dense)(merged)
        merged = keras.layers.BatchNormalization()(merged)

        merged = keras.layers.Dense(num_dense, activation='relu')(merged)
        merged = keras.layers.Dropout(rate_drop_dense)(merged)
        merged = keras.layers.BatchNormalization()(merged)

        preds = keras.layers.Dense(1, activation='sigmoid')(merged)
        model = keras.models.Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
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


