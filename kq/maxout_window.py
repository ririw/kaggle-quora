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
        self.convolution = torch.nn.Conv1d(300, 100, 3)

    def forward(self, words_vecs):
        batch_size, word_vec_size, num_words = words_vecs.size()
        assert word_vec_size == 300, 'Word vec should be 300'

        convolved = self.convolution(words_vecs)
        total = torch.sum(convolved, 2).squeeze() / batch_size
        absmax = torch.max(convolved, 2)[0].squeeze()

        return torch.cat([total, absmax], 1)


class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(200, 50)
        self.l2 = torch.nn.Linear(50, 1)

    def forward(self, X):
        X = self.l1(X)
        X = torch.nn.PReLU()(X)
        X = torch.nn.Dropout(0.25)(X)
        X = self.l2(X)
        return torch.nn.Sigmoid()(X)


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
    max_words = 64
    epochs = 10
    batch_size = 32
    weights = core.weights.astype(np.float32)

    def requires(self):
        yield Dataset()

    def dataset_iterator(self, dataset, requires_grad=True):
        def vectorize(words):
            res = np.zeros([1, 300, self.max_words])
            for ix, word in enumerate(self.English(' '.join(words))):
                if ix == self.max_words:
                    break
                res[:, :, ix] = word.vector
            return res

        traverse_order = np.random.permutation(dataset.shape[0])
        for ix in range(0, dataset.shape[0], self.batch_size):
            ixs = traverse_order[ix:ix + self.batch_size]
            q = dataset.iloc[ixs]
            X1 = q.question1.apply(vectorize).values
            X2 = q.question2.apply(vectorize).values
            y = q.is_duplicate.values
            weight = np.zeros(self.batch_size, dtype=np.float32)
            for i, v in enumerate(y):
                weight[i] = self.weights[v]

            X1 = np.concatenate(X1, 0).astype(np.float32)
            X2 = np.concatenate(X2, 0).astype(np.float32)

            yield Variable(torch.from_numpy(X1), requires_grad=requires_grad), \
                  Variable(torch.from_numpy(X2), requires_grad=requires_grad), \
                  Variable(torch.from_numpy(y.astype(np.float32))), \
                  torch.from_numpy(weight)
        raise StopIteration()

    def run(self):
        self.English = spacy.en.English()
        train, merge, valid = Dataset().load()

        maxout = SiameseMaxout()
        opt = torch.optim.Adam(maxout.parameters())
        test_loss = 0

        for i in range(self.epochs):
            bar = tqdm(itertools.islice(self.dataset_iterator(train, True), 1024), total=1024)
            for (v1, v2, y, w) in bar:
                opt.zero_grad()
                pred = maxout(v1, v2)
                loss = torch.nn.BCELoss(weight=w)(pred, y)
                bar.set_description('%f -- %f' % (loss.data.numpy()[0], test_loss))
                loss.backward()
                opt.step()
            bar = tqdm(itertools.islice(self.dataset_iterator(valid, False), 32), total=32)
            losses = []
            for (v1, v2, y, w) in bar:
                pred = maxout(v1, v2)
                loss = torch.nn.BCELoss(weight=w)(pred, y)
                losses.append(loss.data.numpy()[0])
            test_loss = np.mean(losses)

