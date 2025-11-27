import torch

from cgol.simulator.simulator import Simulator

class TorchSimulator(Simulator):
    def __init__(self, device=torch.device('cpu'), dtype=torch.double):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def step_tensor(self, state: torch.Tensor) -> torch.Tensor:
        pass

    def step_batch_tensor(self, states: torch.Tensor) -> torch.Tensor:
        pass

    def get_config(self) -> dict:
        return {}