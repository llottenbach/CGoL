import numpy as np
import torch

from cgol.generator.torch_generator import TorchGenerator

class UniformDensityGenerator(TorchGenerator):
    def __init__(self, seed, device, dtype):
        super().__init__()
        self.seed = seed
        self.device = device
        self.dtype = dtype
        
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)

    def generate_tensor(self, width: int, height: int) -> torch.Tensor:
        density = torch.rand((1,), generator=self.rng, device=self.device, dtype=self.dtype)
        return (torch.rand((width, height), generator=self.rng, device=self.device, dtype=self.dtype) <= density).to(device=self.device, dtype=self.dtype)

    def generate_batch_tensor(self, width: int, height: int, batch_size: int) -> torch.Tensor:
        densities = torch.rand((batch_size), generator=self.rng, device=self.device, dtype=self.dtype)
        return (torch.rand((batch_size, width, height), generator=self.rng, device=self.device, dtype=self.dtype) <= densities[:,None,None]).to(device=self.device, dtype=self.dtype)

    def generate(self, width: int, height: int) -> np.ndarray:
        return self.generate_tensor(width, height).cpu().numpy()
        
    def generate_batch(self, width: int, height: int, batch_size: int) -> np.ndarray:
        return self.generate_batch_tensor(width, height, batch_size).cpu().numpy()
    
    def get_state(self) -> dict:
        return super().get_state() | {
            "rng": self.rng.get_state()
        }