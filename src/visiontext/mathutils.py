import torch


def torch_stable_softmax(inp, temp=1.0):
    x = inp / temp
    z = x - x.max()
    numerator = torch.exp(z)
    denominator = torch.sum(numerator)
    softmax = numerator / denominator
    return softmax
