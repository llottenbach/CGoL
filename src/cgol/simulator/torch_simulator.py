import torch

from cgol.simulator.simulator import Simulator

class TorchSimulator(Simulator):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')

    def stepTensor(self, state: torch.Tensor) -> torch.Tensor:
        pass

    def stepBatchTensor(self, states: torch.Tensor) -> torch.Tensor:
        pass