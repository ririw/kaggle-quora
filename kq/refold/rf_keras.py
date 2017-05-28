import keras
import luigi
import numpy as np
import os
import pandas
import sklearn
import tensorflow as tf
from plumbum import colors
from sklearn import preprocessing, pipeline, base

from kq.core import dictweights, score_data
from kq.feat_abhishek import FoldDependent
from kq.refold import rf_seq_data, rf_dataset, BaseTargetBuilder

__all__ = ['ReaderModel', 'SiameseModel', 'SequenceTask', 'TestModel']


class Clipping(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, lower, upper):
        self.upper = upper
        self.lower = lower

    def fit(self, X):
        pass

    def transform(self, X):
        return X.clip(self.lower, self.upper)

class SequenceTask(FoldDependent):
    def model(self, embedding_mat, seq_len, otherdata_size) -> keras.models.Model:
        raise NotImplementedError

    def _load(self, name, as_df):
        res = np.load(self.output().path)[name]
        if as_df:
            res = pandas.Series(res, name=repr(self))
        return res

    def make_path(self, fname):
        raise NotImplementedError

    def requires(self):
        yield rf_seq_data.RFWordSequenceDataset()
        yield rf_dataset.Dataset()

    def output(self):
        return luigi.LocalTarget(self.make_path('done.npz'))

    def run(self):
        self.output().makedirs()
        batch_size = 128
        normalizer = pipeline.Pipeline([
            ('normalize', preprocessing.Normalizer()),
            ('truncate', Clipping(-10, 10))])

        train_q1, train_q2, train_other = rf_seq_data.RFWordSequenceDataset().load('train', fold=self.fold)
        train_other = normalizer.fit_transform(train_other)
        train_labels = rf_dataset.Dataset().load('train', fold=self.fold, as_df=True).is_duplicate
        print(train_q1.shape, train_q2.shape, train_other.shape)
        embedding = rf_seq_data.RFWordSequenceDataset().load_embedding_mat()

        model = self.model(embedding, train_q2.shape[1], train_other.shape[1])

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
        slow_plateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3)
        model_path = self.make_path('model.h5')
        model_checkpointer = keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, save_weights_only=True)

        train_data = [train_q1, train_q2, train_other]

        model.fit(
           train_data, train_labels,
           validation_split=0.05,
           epochs=20, batch_size=batch_size, shuffle=True,
           class_weight=dictweights,
           callbacks=[early_stopping, slow_plateau, model_checkpointer])
        model.load_weights(model_path)

        valid_q1, valid_q2, valid_other = rf_seq_data.RFWordSequenceDataset().load('valid', fold=self.fold)
        valid_other = normalizer.transform(valid_other)
        valid_labels = rf_dataset.Dataset().load('valid', fold=self.fold, as_df=True).is_duplicate
        valid_data = [valid_q1, valid_q2, valid_other]
        valid_preds = model.predict(valid_data, verbose=1, batch_size=batch_size)

        score = score_data(valid_labels, valid_preds)
        print(colors.green | "Score for {:s}: {:f}".format(repr(self), score))

        test_q1, test_q2, test_other = rf_seq_data.RFWordSequenceDataset().load('test', None)
        test_other = normalizer.transform(test_other)
        test_data = [test_q1, test_q2, test_other]
        test_preds = model.predict(test_data, verbose=1, batch_size=batch_size)

        np.savez_compressed(self.make_path('done_tmp.npz'), valid=valid_preds, test=test_preds)
        os.rename(self.make_path('done_tmp.npz'), self.output().path)
        return score


class TestModel(SequenceTask):
    resources = {'gpu': 1}

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_keras',
            'test',
            str(self.fold)
        )
        return (base_path + fname).get()

    def model(self, embedding_mat, seq_len, otherdata_size) -> keras.models.Model:
        cast_l = keras.layers.Lambda(lambda l: tf.cast(l, tf.float32))
        sequence_1_input = keras.layers.Input(shape=[seq_len], dtype='int32')
        sequence_2_input = keras.layers.Input(shape=[seq_len], dtype='int32')
        s1 = cast_l(sequence_1_input)
        s2 = cast_l(sequence_2_input)
        distance_input = keras.layers.Input(shape=[otherdata_size])
        d = keras.layers.concatenate([s1, s2, distance_input])
        d = keras.layers.Dense(512)(d)
        d = keras.layers.PReLU()(d)
        d = keras.layers.Dropout(0.5)(d)
        d = keras.layers.Dense(256)(d)
        d = keras.layers.PReLU()(d)
        d = keras.layers.Dropout(0.5)(d)
        d = keras.layers.Dense(128)(d)
        d = keras.layers.PReLU()(d)
        d = keras.layers.Dropout(0.5)(d)

        d = keras.layers.Dense(1, activation='sigmoid')(d)
        model = keras.models.Model(inputs=[sequence_1_input, sequence_2_input, distance_input], outputs=d)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

        return model


class SiameseModel(SequenceTask):
    resources = {'gpu': 1}

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_keras',
            'siamese',
            str(self.fold)
        )
        return (base_path + fname).get()

    def model(self, embedding_mat, seq_len, otherdata_size) -> keras.models.Model:
        num_lstm = np.random.randint(175, 275)
        num_dense = np.random.randint(100, 150)
        rate_drop_lstm = 0.15 + np.random.rand() * 0.25
        rate_drop_dense = 0.15 + np.random.rand() * 0.25

        embedding_layer = keras.layers.Embedding(
            embedding_mat.shape[0],
            embedding_mat.shape[1],
            weights=[embedding_mat], input_length=seq_len, trainable=False)

        lstm_layer1 = keras.layers.Bidirectional(
            keras.layers.LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)
        )

        sequence_1_input = keras.layers.Input(shape=[seq_len], dtype='int32')
        embed_1 = embedding_layer(sequence_1_input)
        x1 = lstm_layer1(embed_1)

        sequence_2_input = keras.layers.Input(shape=[seq_len], dtype='int32')
        embed_2 = embedding_layer(sequence_2_input)
        y1 = lstm_layer1(embed_2)

        distance_input = keras.layers.Input(shape=[otherdata_size])
        di = keras.layers.Dense(num_dense, activation='relu')(distance_input)
        di = keras.layers.Dropout(num_lstm)(di)

        di = keras.layers.Dense(num_dense, activation='relu')(di)

        merged = keras.layers.concatenate([x1, y1, di])
        merged = keras.layers.Dropout(rate_drop_dense)(merged)
        merged = keras.layers.Dense(num_dense, activation='relu')(merged)
        merged = keras.layers.Dropout(rate_drop_dense)(merged)
        merged = keras.layers.Dense(num_dense//2, activation='relu')(merged)

        preds = keras.layers.Dense(1, activation='sigmoid')(merged)
        preds = keras.layers.Lambda(lambda v: tf.clip_by_value(v, 1e-10, 1-1e-10))(preds)

        model = keras.models.Model(inputs=[sequence_1_input, sequence_2_input, distance_input], outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

        return model


class ReaderModel(SequenceTask):
    resources = {'gpu': 1}

    def make_path(self, fname):
        base_path = BaseTargetBuilder(
            'rf_keras',
            'reader',
            str(self.fold)
        )
        return (base_path + fname).get()

    def model(self, embedding_matrix, vec_len, distance_width) -> keras.models.Model:
        num_lstm = np.random.randint(250, 400)
        num_dense = np.random.randint(100, 150)
        rate_drop_lstm = 0.15 + np.random.rand() * 0.25
        rate_drop_dense = 0.15 + np.random.rand() * 0.25

        embedding_layer = keras.layers.Embedding(
            embedding_matrix.shape[0], embedding_matrix.shape[1],
            weights=[embedding_matrix], input_length=vec_len * 2, trainable=False)

        lstm_layer1 = keras.layers.Bidirectional(
            keras.layers.LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)
        )

        sequence_1_input = keras.layers.Input(shape=[vec_len], dtype='int32')
        sequence_2_input = keras.layers.Input(shape=[vec_len], dtype='int32')
        x1x2 = keras.layers.concatenate([sequence_1_input, sequence_2_input])
        embed = embedding_layer(x1x2)
        l1 = lstm_layer1(embed)

        distance_input = keras.layers.Input(shape=[distance_width])
        di = keras.layers.Dense(num_dense, activation='relu')(distance_input)
        di = keras.layers.Dropout(rate_drop_dense)(di)
        di = keras.layers.BatchNormalization()(di)

        di = keras.layers.Dense(num_dense, activation='relu')(di)
        di = keras.layers.Dropout(rate_drop_dense)(di)
        di = keras.layers.BatchNormalization()(di)

        merged = keras.layers.concatenate([l1, di])
        merged = keras.layers.Dropout(rate_drop_dense)(merged)
        merged = keras.layers.BatchNormalization()(merged)

        merged = keras.layers.Dense(1, activation='sigmoid')(merged)
        merged = keras.layers.Lambda(lambda v: tf.clip_by_value(v, 1e-10, 1-1e-10))(merged)

        model = keras.models.Model([sequence_1_input, sequence_2_input, distance_input], merged)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

        return model
