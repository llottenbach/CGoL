import torch

from cgol.generator.generator import Generator

class TorchGenerator(Generator):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')

    def generate_tensor(self, width: int, height: int) -> torch.Tensor:
        pass

    def generate_batch_tensor(self, width: int, height: int, batch_size: int) -> torch.Tensor:
        pass
