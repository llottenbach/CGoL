import torch
import torch.nn as NN
import torch.nn.functional as F

class MinimalArchitectureModel(NN.Module):
    def __init__(self, *args, device=torch.device('cpu'), dtype=torch.double, derivable=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.derivable = derivable
        self.dtype = dtype
        self.device = device

        self.conv1 = NN.Conv2d(1, 2, kernel_size=3, padding=1, dtype=dtype, device=device)
        self.conv2 = NN.Conv2d(2, 1, kernel_size=1, bias=False, dtype=dtype, device=device)

        self.conv1.weight = NN.Parameter(torch.tensor([
            [[[1,1,1], [1,.1,1], [1,1,1]]],
            [[[1,1,1], [1,1,1], [1,1,1.]]]
        ],dtype=dtype, device=device))
        self.conv1.bias = NN.Parameter(torch.tensor([
            -3, 
            -2.
        ],dtype=dtype, device=device))

        self.conv2.weight = NN.Parameter(torch.tensor([
            [[[-10]], [[1.]]]
        ],dtype=dtype, device=device))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        if not self.derivable:
            x = (x > 0).float()
        
        return x

