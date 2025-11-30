import torch

torch.nn.MSELoss()

class CompletionLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((((input - target)**2).sum((-1,-2)) ** 2).sum() 
                / (input.size().numel()))