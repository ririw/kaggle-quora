import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas
import spacy.en
import torch
import torch.optim
from sklearn import metrics
from torch.autograd import Variable
from tqdm import tqdm

from kq import rwa, utils
from kq.dataset import Dataset
from kq.shared_words import Vocab


class QuestionReader(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = rwa.RWA(300, 64)
        self.lt1 = torch.nn.Linear(64*2, 64)
        self.lt2 = torch.nn.Linear(64, 1)

    def forward(self, in1, in2):
        _, hn1 = self.rnn(in1)
        _, hn2 = self.rnn(in2)
        hn = torch.cat([hn1, hn2], 1)
        hn = torch.nn.BatchNorm1d(64*2)(hn)
        lin = torch.nn.PReLU()(self.lt1(hn))
        lin = self.lt2(lin)
        return torch.nn.Sigmoid()(lin)


class QuestionReaderTask(luigi.Task):
    max_words = 64

    def requires(self):
        yield Vocab()
        yield Dataset()

    def output(self):
        return luigi.LocalTarget('./cache/classifier_pred.csv.gz')

    def vectorize(self, words, English):
        res = np.zeros([self.max_words, 300])
        for ix, tok in enumerate(English(words)):
            if ix >= self.max_words:
                break
            res[ix, :] = tok.vector
        return res[:, None, :]

    def make_data_vecs(self, frame, English):
        while True:
            samp = frame.sample(128)
            X1 = np.concatenate(samp.question1.apply(lambda v: self.vectorize(v, English)).values, 1)
            X2 = np.concatenate(samp.question2.apply(lambda v: self.vectorize(v, English)).values, 1)
            y = samp.is_duplicate.values.astype(np.float32)

            yield X1, X2, y

    def run(self):
        English = spacy.en.English()
        train, valid = Dataset().load()
        qr = QuestionReader()
        opt = torch.optim.Adam(qr.parameters(), betas=[0.95, 0.99])

        bar = tqdm(range(20000))
        last_valid_loss = np.NaN
        scores = []
        valid_scores = []
        train_gen = utils.GeneratorEnqueuer(self.make_data_vecs(train, English), pickle_safe=True)
        valid_gen = utils.GeneratorEnqueuer(self.make_data_vecs(valid, English), pickle_safe=True)
        train_gen.start()
        valid_gen.start()
        scores_file = open('./cache/dmn_scores.csv', 'w')
        for i in bar:
            opt.zero_grad()
            X1, X2, y = train_gen.queue.get()
            X1 = Variable(torch.from_numpy(X1), requires_grad=True).float()
            X2 = Variable(torch.from_numpy(X2), requires_grad=True).float()
            y = Variable(torch.from_numpy(y))

            pred = qr(X1, X2)
            loss = torch.nn.BCELoss()(pred, y)
            loss.backward()
            l = loss.data.numpy()[0]
            opt.step()

            if np.random.uniform() < 0.1:
                X1, X2, y = valid_gen.queue.get()
                X1 = Variable(torch.from_numpy(X1), requires_grad=True).float()
                X2 = Variable(torch.from_numpy(X2), requires_grad=True).float()
                y = Variable(torch.from_numpy(y))
                loss = torch.nn.BCELoss()(qr(X1, X2), y)
                last_valid_loss = loss.data.numpy()[0]
            scores_file.write('%f,%f\n' % (l, last_valid_loss))
            scores_file.flush()
            scores.append(l)
            valid_scores.append(last_valid_loss)

            bar.set_description('%02f -- %02f' % (l, last_valid_loss))
        train_gen.stop()
        valid_gen.stop()
        scoring = pandas.DataFrame({
            'scores': scores,
            'valid_scores': valid_scores
        })
        scoring.to_csv('./scores.csv')
        scoring.plot()
        plt.semilogy()
        plt.savefig('./scores.png')


def test_for_fun():
    # seq_len, batch, input_size
    qr = QuestionReader()
    opt = torch.optim.Adam(qr.parameters())

    def gen_data():
        X = np.random.normal(size=[7, 32, 50]).astype(dtype=np.float32)
        y = np.random.choice(2, size=32).astype(np.float32)
        X[6, :, 0] = y
        Xvar = Variable(torch.from_numpy(X), requires_grad=True)
        yvar = Variable(torch.from_numpy(y))

        return Xvar, yvar

    bar = tqdm(range(200))
    for i in bar:
        Xvar, yvar = gen_data()
        opt.zero_grad()
        v = qr(Xvar)
        loss = torch.nn.BCELoss()(v, yvar)
        loss.backward()
        opt.step()
        bar.set_description('%02f' % loss.data.numpy()[0])

    print(metrics.log_loss(y, v.data.numpy()))

