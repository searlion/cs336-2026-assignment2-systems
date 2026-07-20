import torch

a = torch.tensor([[1, 2, 3],[4,5,6],[7,8,9]])
b = torch.tensor([4, 5, 6]).unsqueeze(-1)

# All three expressions yield the same result: tensor([4, 10, 18])
res1 = a * b
res2 = torch.mul(a, b)
res3 = a.mul(b)
print(a.shape[:-1])
# print(res1, res2, res3)