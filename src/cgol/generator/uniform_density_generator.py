import numpy as np
import torch

from cgol.generator.torch_generator import TorchGenerator

class UniformDensityGenerator(TorchGenerator):
    def __init__(self, seed, device, width=0, height=0, batchsize=0):
        super().__init__()
        self.seed = seed
        self.device = device
        self.width = width
        self.height = height
        self.batch_size = batchsize
        
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)

    def generateTensor(self) -> torch.Tensor:
        density = torch.rand((1,), generator=self.rng, device=self.device, dtype=torch.double)
        return (torch.rand((self.width, self.height), generator=self.rng, device=self.device, dtype=torch.double) <= density).to(device=self.device, dtype=torch.double)

    def generateBatchTensor(self) -> torch.Tensor:
        densities = torch.rand((self.batch_size), generator=self.rng, device=self.device, dtype=torch.double)
        return (torch.rand((self.batch_size, self.width, self.height), generator=self.rng, device=self.device, dtype=torch.double) <= densities[:,None,None]).to(device=self.device, dtype=torch.double)

    def generate(self) -> np.ndarray:
        return self.generateTensor().cpu().numpy()
        
    def generateBatch(self) -> np.ndarray:
        return self.generateBatchTensor().cpu().numpy()