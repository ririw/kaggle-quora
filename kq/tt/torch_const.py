"""
A dummy testing task
"""
import luigi
import torch
import numpy as np
from . import torchtask, to_torchvar


class TorchConst(torchtask.TorchTask):
    def model(self, embedding_mat, vector_input_shape, otherfeature_shape, batch_size):
        return ConstModel()

class ConstModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.const = torch.nn.Parameter(torch.from_numpy(0.5 * np.ones([1], dtype=np.float32)))

    def forward(self, x1, x2, features):
        batch_size = features.size()[0]
        return self.const.expand(batch_size)
