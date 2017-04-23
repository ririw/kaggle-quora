import luigi
import numpy as np
import spacy.en
import keras

from kq.dataset import Dataset
from kq.shared_words import Vocab

English = None
class QuestionReaderTask(luigi.Task):
    max_words = 32

    def requires(self):
        yield Dataset()

    def output(self):
        return luigi.LocalTarget('./cache/classifier_pred.csv.gz')

    def vectorize(self, sent):
        global English
        if English is None:
            English = spacy.en.English()

        res = np.zeros([self.max_words, 300])
        for ix, tok in enumerate(English(sent)):
            if ix >= self.max_words:
                break
            res[ix, :] = tok.vector
        return res[:, None, :]

    def make_data_vecs(self, frame):
        while True:
            samp = frame.sample(64)
            X1 = np.concatenate(samp.question1_raw.apply(self.vectorize).values, 1)
            X2 = np.concatenate(samp.question2_raw.apply(self.vectorize).values, 1)
            y = samp.is_duplicate.values.astype(np.float32)

            yield [X1.transpose(1, 0, 2), X2.transpose(1, 0, 2)], y

    def run(self):
        train, merge, valid = Dataset().load()

        in1 = keras.layers.Input([self.max_words, 300])
        in2 = keras.layers.Input([self.max_words, 300])
        lstm = keras.layers.LSTM(300, dropout=0.25, recurrent_dropout=0.25)
        v1 = lstm(in1)
        v2 = lstm(in2)
        paired = keras.layers.BatchNormalization()(
            keras.layers.Dropout(0.25)(keras.layers.merge([v1, v2], mode='concat')))

        linmodel = keras.models.Sequential()
        linmodel.add(keras.layers.Dense(300, input_shape=[600]))
        linmodel.add(keras.layers.PReLU())
        linmodel.add(keras.layers.Dropout(0.25))
        linmodel.add(keras.layers.BatchNormalization())
        linmodel.add(keras.layers.Dense(150))
        linmodel.add(keras.layers.PReLU())
        linmodel.add(keras.layers.Dropout(0.25))
        linmodel.add(keras.layers.BatchNormalization())
        linmodel.add(keras.layers.Dense(1, activation='sigmoid'))

        l = linmodel(paired)

        model = keras.models.Model([in1, in2], l)
        model.compile('nadam', 'binary_crossentropy')

        model.fit_generator(
            self.make_data_vecs(train), 128, epochs=1000,
            validation_data=self.make_data_vecs(valid),
            validation_steps=32, class_weight={0: 1.309028344, 1: 0.472001959})
