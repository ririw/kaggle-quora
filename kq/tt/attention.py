"""
Learn how to pay attention by creating a summary vector.

This model reads both sentences, with a siamese network, and forms two summary matricies
These two matricies are dot-producted together, and the resulting weights are used to 
form inputs into a final comparative RNN, which makes a decision about the overall scentence.
"""
import sys
import torch
import torch.nn
import torch.nn.functional
import numpy as np
from . import torchtask


class AttentionReadModel(torchtask.TorchTask):
    def model(self, embedding_mat, vector_input_shape, otherfeature_shape, batch_size):
        return AttentionReadNetwork(embedding_mat, vector_input_shape, otherfeature_shape)

class AttentionReadNetwork(torch.nn.Module):
    def __init__(self, embedding_mat, vector_input_shape, otherfeature_shape):
        super().__init__()
        self.embedding = torch.nn.Embedding(embedding_mat.shape[0], embedding_mat.shape[1])
        self.embedding.weight = torch.nn.Parameter(torch.from_numpy(embedding_mat), requires_grad=False)

        self.summarizer = AttentionSummaryNetwork()

    def forward(self, x1, x2, features):
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        attention1 = self.summarizer(x1)
        attention2 = self.summarizer(x2)

        prod = torch.dot(attention1, attention2)
        #weights = torch.nn.Softmax()(torch.sum(prod, 2).squeeze())
        print(prod.size())
        print(x1.size())
        print(x2.size())

class AttentionSummaryNetwork(torch.nn.Module):
    attention_size = 25
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.GRU(300, self.attention_size, 1, batch_first=True, bidirectional=True)
        self.h = torch.nn.Parameter(torch.from_numpy(np.random.normal(size=[2, self.attention_size]).astype(np.float32)))


    def forward(self, x):
        batch_size, vec_wdith, _ = x.size()
        hm = self.h.unsqueeze(1)
        hm = hm.expand(2, batch_size, self.attention_size)
        attention_vecs, _ = self.rnn(x, hm.contiguous())

        return attention_vecs

