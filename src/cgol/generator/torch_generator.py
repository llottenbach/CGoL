import torch

from cgol.generator.generator import Generator

class TorchGenerator(Generator):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')

    def generateTensor(self) -> torch.Tensor:
        pass

    def generateBatchTensor(self) -> torch.Tensor:
        pass
