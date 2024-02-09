import torch


def torch_stable_softmax(inp, temp=1.0, dim=-1):
    x: torch.Tensor = inp / temp
    max_values, max_indices = x.max(dim=dim, keepdim=True)
    z = x - max_values
    numerator = torch.exp(z)
    denominator = torch.sum(numerator, dim=dim, keepdim=True)
    softmax = numerator / denominator
    return softmax
