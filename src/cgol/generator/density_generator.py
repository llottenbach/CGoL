import numpy as np
import torch

from cgol.generator.torch_generator import TorchGenerator

class DensityGenerator(TorchGenerator):
    def __init__(self, seed, device, density=0.5, width=0, height=0, batchsize=0):
        super().__init__()
        self.seed = seed
        self.device = device
        self.density = torch.tensor(density, dtype=torch.double, device=device)
        self.width = width
        self.height = height
        self.batch_size = batchsize
        
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)

    def generateTensor(self) -> torch.Tensor:
        return (torch.rand((self.width, self.height), generator=self.rng, device=self.device, dtype=torch.double) <= self.density).to(device=self.device, dtype=torch.double)

    def generateBatchTensor(self) -> torch.Tensor:
        return (torch.rand((self.batch_size, self.width, self.height), generator=self.rng, device=self.device, dtype=torch.double) <= self.density).to(device=self.device, dtype=torch.double)

    def generate(self) -> np.ndarray:
        return self.generateTensor().cpu().numpy()
        
    def generateBatch(self) -> np.ndarray:
        return self.generateBatchTensor().cpu().numpy()