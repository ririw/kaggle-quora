import luigi
import os
import numpy as np
import spacy.en
import keras
from keras.layers.merge import concatenate
from sklearn import metrics

from kq.dataset import Dataset
from kq.shared_words import Vocab

English = None
class QuestionReaderTask(luigi.Task):
    max_words = 32

    def requires(self):
        yield Dataset()

    def output(self):
        return luigi.LocalTarget('./cache/keras/classifier_pred.csv')

    def vectorize(self, sent):
        global English
        if English is None:
            English = spacy.en.English()

        res = np.zeros([self.max_words, 300])
        for ix, tok in enumerate(English(sent)):
            if ix >= self.max_words:
                break
            if tok.is_oov:
                continue
            res[ix, :] = tok.vector
        return res[:, None, :]

    def make_data_vecs(self, frame):
        while True:
            samp = frame.sample(256)
            X1 = np.concatenate(samp.question1_raw.apply(self.vectorize).values, 1).transpose(1, 0, 2)
            X2 = np.concatenate(samp.question2_raw.apply(self.vectorize).values, 1).transpose(1, 0, 2)
            y = samp.is_duplicate.values.astype(np.float32)

            yield [X1, X2], y

    def make_seq_dataset(self, frame):
        while True:
            batches = int(np.ceil(frame.shape[0] / 512))
            for ix in range(batches):
                start = ix * 512
                end = min(frame.shape[0], (ix + 1) * 512)
                samp = frame.iloc[start:end]

                X1 = np.concatenate(samp.question1_raw.apply(self.vectorize).values, 1)
                X2 = np.concatenate(samp.question2_raw.apply(self.vectorize).values, 1)

                yield [X1, X2]
            print("!!!!!!!")

    def run(self):
        self.output().makedirs()
        train, merge, valid = Dataset().load()

        in1 = keras.layers.Input([self.max_words, 300])
        in2 = keras.layers.Input([self.max_words, 300])
        lstm = keras.layers.LSTM(200, dropout=0.25, recurrent_dropout=0.25)
        v1 = lstm(in1)
        v2 = lstm(in2)

        merged = concatenate([v1, v2])

        linmodel = keras.models.Sequential()
        linmodel.add(keras.layers.Dense(200, input_shape=[400]))
        linmodel.add(keras.layers.PReLU())
        linmodel.add(keras.layers.Dropout(0.25))
        linmodel.add(keras.layers.BatchNormalization())
        linmodel.add(keras.layers.Dense(150))
        linmodel.add(keras.layers.PReLU())
        linmodel.add(keras.layers.Dropout(0.25))
        linmodel.add(keras.layers.BatchNormalization())
        linmodel.add(keras.layers.Dense(1, activation='sigmoid'))

        l = linmodel(merged)

        model = keras.models.Model([in1, in2], l)
        model.compile('nadam', 'binary_crossentropy')
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

        #batches = np.ceil(train.shape[0] / 512)
        model.fit_generator(
            self.make_data_vecs(train), 1024,
            epochs=100,
            validation_data=self.make_data_vecs(valid),
            validation_steps=32,
            class_weight={0: 1.309028344, 1: 0.472001959},
            callbacks=[early_stopping])

        batches = np.ceil(valid.shape[0] / 512)
        valid_predictions = model.predict_generator(self.make_seq_dataset(valid), int(batches), verbose=1)
        print(metrics.log_loss(valid.is_duplicate.values, valid_predictions))

        batches = np.ceil(merge.shape[0] / 512)
        merge_predictions = model.predict_generator(self.make_seq_dataset(merge), int(batches), verbose=1)
        print(merge_predictions.shape)
        np.save('cache/keras/merge_predictions.csv', merge_predictions)

        test = Dataset().load_test()
        batches = np.ceil(test.shape[0] / 512)
        test_predictions = model.predict_generator(self.make_seq_dataset(test), int(batches), verbose=1)
        print(test_predictions.shape)
        np.save('cache/keras/classifier_pred_tmp.csv', merge_predictions)
        os.rename('cache/keras/classifier_pred_tmp.csv', self.output().path)
