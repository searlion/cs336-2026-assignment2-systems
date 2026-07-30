import torch
from torch import nn
from naive_ddp import Naive_DDP

toy_model = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10,5)
)

Naive_DDP(toy_model)