from torch import nn
import torch

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        assert torch.backends.mps.is_available(), "MPS not available!"
        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            print(x.dtype)
            x = self.relu(self.fc1(x))
            print(x.dtype)
            x = self.ln(x)
            print(x.dtype)
            x = self.fc2(x)
            print(x.dtype)
            return x
         
toy_example = ToyModel(5,1).to("mps")
print(next(toy_example.parameters()).device)
input = torch.ones(5).to("mps")
print(input.device)
toy_example.forward(input)