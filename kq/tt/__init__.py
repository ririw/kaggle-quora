import torch

def to_torchvar(var, requires_grad=True):
    return torch.autograd.Variable(torch.from_numpy(var), requires_grad=requires_grad)