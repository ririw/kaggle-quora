import luigi
import spacy
import torch
import numpy as np
from torch.autograd import Variable
import itertools

from tqdm import tqdm

from kq import core
from kq.dataset import Dataset


class MaxoutReduction(torch.nn.Module):
    """
    Input size: [BATCH_SIZE, 300, NUM_WORDS]
    """

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(300, 100, 3)
        self.conv2 = torch.nn.Conv1d(100, 100, 5)

    def forward(self, words_vecs):
        batch_size, word_vec_size, num_words = words_vecs.size()
        assert word_vec_size == 300, 'Word vec should be 300'

        c1 = torch.nn.PReLU()(torch.nn.MaxPool1d(2)(self.conv1(words_vecs)))
        c2 = torch.nn.PReLU()(torch.nn.MaxPool1d(2)(self.conv2(c1)))

        return c2.resize(batch_size, 500)


class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(500, 250)
        self.l2 = torch.nn.Linear(250, 250)
        self.l3 = torch.nn.Linear(250, 50)
        self.l4 = torch.nn.Linear(50, 2)

    def forward(self, X):
        X = torch.nn.PReLU()(self.l1(torch.nn.Dropout(0.25)(X)))
        X = torch.nn.PReLU()(self.l2(torch.nn.Dropout(0.25)(X)))
        X = torch.nn.PReLU()(self.l3(torch.nn.Dropout(0.25)(X)))
        X = torch.nn.PReLU()(self.l4(torch.nn.Dropout(0.25)(X)))
        return X


class SiameseMaxout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxout = MaxoutReduction()
        self.linear = Linear()

    def forward(self, v1, v2):
        v1 = self.maxout(v1)
        v2 = self.maxout(v2)
        diff = v1 - v2
        return self.linear(diff)


class MaxoutTask(luigi.Task):
    max_words = 31
    epochs = 500
    batch_size = 32
    weights = core.weights.astype(np.float32)

    def requires(self):
        yield Dataset()

    def output(self):
        return luigi.LocalTarget('cache/maxout/done')

    def dataset_iterator(self, dataset, requires_grad=True):
        def vectorize(words):
            res = np.zeros([1, 300, self.max_words])
            j = 0
            for word in self.English(words):
                if word.is_stop or word.is_punct:
                    continue
                if j >= self.max_words:
                    break
                res[:, :, j] = word.vector
                j += 1
            return res

        traverse_order = np.random.permutation(dataset.shape[0])
        for ix in range(0, dataset.shape[0], self.batch_size):
            ixs = traverse_order[ix:ix + self.batch_size]
            q = dataset.iloc[ixs]
            X1 = q.question1.fillna('').apply(vectorize).values
            X2 = q.question2.fillna('').apply(vectorize).values
            #y = np.zeros([q.shape[0], 2], dtype=np.float32)
            #y[np.arange(q.shape[0]), q.is_duplicate.values] = 1
            y = q.is_duplicate.values.astype(np.int64)
            weight = np.zeros(self.batch_size, dtype=np.float32)
            for i, v in enumerate(q.is_duplicate.values):
                weight[i] = self.weights[v]

            X1 = np.concatenate(X1, 0).astype(np.float32)
            X2 = np.concatenate(X2, 0).astype(np.float32)

            yield Variable(torch.from_numpy(X1), requires_grad=requires_grad), \
                  Variable(torch.from_numpy(X2), requires_grad=requires_grad), \
                  Variable(torch.from_numpy(y)), \
                  torch.from_numpy(weight)
        raise StopIteration()

    def run(self):
        self.output().makedirs()
        self.English = spacy.en.English()
        train, merge, valid = Dataset().load()

        maxout = SiameseMaxout()
        opt = torch.optim.Adam(maxout.parameters())
        test_loss = np.NaN
        train_loss = None
        weights = torch.from_numpy(core.weights.astype(np.float32))

        score_file = open('cache/maxout/scores.csv', 'w')
        score_file.write('train_loss,test_loss\n')

        for i in range(self.epochs):
            bar = tqdm(itertools.islice(self.dataset_iterator(train, True), 256), total=256)
            for (v1, v2, y, _) in bar:
                opt.zero_grad()
                pred = maxout(v1, v2)
                loss = torch.nn.CrossEntropyLoss(weight=weights)(pred, y)
                if train_loss is None:
                    train_loss = loss.data.numpy()[0]
                else:
                    train_loss = 0.95 * train_loss + 0.05 * loss.data.numpy()[0]
                bar.set_description('%f -- %f' % (train_loss, test_loss))
                loss.backward()
                opt.step()
                score_file.write('%f,%f\n' % (train_loss, test_loss))
                score_file.flush()

            bar = tqdm(itertools.islice(self.dataset_iterator(valid, False), 32), total=32)
            losses = []
            for (v1, v2, y, _) in bar:
                pred = maxout(v1, v2)
                loss = torch.nn.CrossEntropyLoss(weight=weights)(pred, y)
                losses.append(loss.data.numpy()[0])
            test_loss = np.mean(losses)
        score_file.close()