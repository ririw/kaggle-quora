import sys
import torch
import torch.nn
import torch.nn.functional
import numpy as np
from . import torchtask


class RNNReadModel(torchtask.TorchTask):
    def model(self, embedding_mat, vector_input_shape, otherfeature_shape, batch_size):
        return RNNReadModule(embedding_mat, vector_input_shape, otherfeature_shape)

class RNNReadModule(torch.nn.Module):
    def __init__(self, embedding_mat, vector_input_shape, otherfeature_shape):
        super().__init__()
        self.embedding = torch.nn.Embedding(embedding_mat.shape[0], embedding_mat.shape[1])
        self.embedding.weight = torch.nn.Parameter(torch.from_numpy(embedding_mat), requires_grad=False)
        self.rnn = torch.nn.LSTM(300, 128, 1, dropout=0.25)

        self.h_transform = torch.nn.Linear(otherfeature_shape, 128*1)
        self.c_transform = torch.nn.Linear(otherfeature_shape, 128*1)

        self.input_transforms = [
            torch.nn.Linear(otherfeature_shape, 32),
            torch.nn.PReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(32, 32),
            torch.nn.PReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(32, 32),
            torch.nn.PReLU(),
            torch.nn.Dropout(0.25),
        ]

        self.output_transforms = [
            torch.nn.Linear(256+32, 128),
            torch.nn.PReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        ]
        for ix, l in enumerate(self.output_transforms + self.input_transforms):
            self.add_module('Layer {} {}'.format(ix, str(l)), l)

    def forward(self, x1, x2, features):
        x1 = x1.transpose(0, 1)
        x2 = x2.transpose(0, 1)
        n_words, batch_size = x1.size()
        x12 = torch.cat([x1, x2], 0)
        embedded = self.embedding(x12)
        h = self.h_transform(features)
        c = self.c_transform(features)
        _, outputs = self.rnn(embedded, (h.view(1, batch_size, -1), c.view(1, batch_size, -1)))

        f = features
        for l in self.input_transforms:
            f = l(f)

        r = torch.cat([outputs[0].squeeze(), f], 1)
        for l in self.output_transforms:
            r = l(r)
        return r
