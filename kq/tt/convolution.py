import sys
import torch
import torch.nn
import torch.nn.functional
import numpy as np
from . import torchtask


class ConvModel(torchtask.TorchTask):
    def model(self, embedding_mat, vector_input_shape, otherfeature_shape, batch_size):
        return ConvModule(embedding_mat, vector_input_shape, otherfeature_shape)

class ConvModule(torch.nn.Module):
    def __init__(self, embedding_mat, vector_input_shape, otherfeature_shape):
        super().__init__()
        self.embedding = torch.nn.Embedding(embedding_mat.shape[0], embedding_mat.shape[1])
        self.embedding.weight = torch.nn.Parameter(torch.from_numpy(embedding_mat), requires_grad=False)
        self.marker = torch.nn.Parameter(torch.from_numpy(np.arange(vector_input_shape, dtype=np.float32)),
                                         requires_grad=False)
        self.conv1 = torch.nn.Conv1d(301, 150, 3)
        self.conv2 = torch.nn.Conv1d(150, 100, 5)
        self.conv3 = torch.nn.Conv1d(100, 100, 5)
        self.conv_prelu1 = torch.nn.PReLU()
        self.conv_prelu2 = torch.nn.PReLU()
        self.conv_prelu3 = torch.nn.PReLU()
        self.dense1 = torch.nn.Linear(100*24*2+otherfeature_shape, 32 * 4 * 2)
        self.dense2 = torch.nn.Linear(32 * 4 * 2, 32)
        self.dense3 = torch.nn.Linear(32, 1)
        self.dense_prelu1 = torch.nn.PReLU()
        self.dense_prelu2 = torch.nn.PReLU()

    def forward(self, x1, x2, features):
        x1 = self.process_vec(x1)
        x2 = self.process_vec(x2)
        v = torch.cat([x1, x2, features], 1)
        v = self.dense_prelu1(self.dense1(v))
        v = self.dense_prelu2(self.dense2(v))
        v = torch.nn.functional.sigmoid(self.dense3(v))

        return v

    def process_vec(self, v):
        batch_size, vec_len = v.size()
        bigmarker = self.marker.expand([batch_size, vec_len]).unsqueeze(2)

        v = self.embedding(v)
        v = torch.cat([v, bigmarker], 2).transpose(1, 2)
        v = self.conv_prelu1(self.conv1(v))
        v = self.conv_prelu2(self.conv2(v))
        v = self.conv_prelu3(self.conv3(v))
        v = v.resize(batch_size, 32 * 24)
        return v