import torch

a = torch.tensor([[0.5, 0.5]])

c = torch.matmul(a, a.T)
print(c)