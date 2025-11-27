import numpy as np
import torch

from cgol.generator.torch_generator import TorchGenerator

class DensityGenerator(TorchGenerator):
    def __init__(self, seed, device, density=0.5):
        super().__init__()
        self.seed = seed
        self.device = device
        self.density = torch.tensor(density, dtype=torch.double, device=device)
        
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)

    def generate_tensor(self, width: int, height: int) -> torch.Tensor:
        return (torch.rand((width, height), generator=self.rng, device=self.device, dtype=torch.double) <= self.density).to(device=self.device, dtype=torch.double)

    def generate_batch_tensor(self, width: int, height: int, batch_size: int) -> torch.Tensor:
        return (torch.rand((batch_size, width, height), generator=self.rng, device=self.device, dtype=torch.double) <= self.density).to(device=self.device, dtype=torch.double)

    def generate(self, width: int, height: int) -> np.ndarray:
        return self.generate_tensor(width, height).cpu().numpy()
        
    def generate_batch(self, width: int, height: int, batch_size: int) -> np.ndarray:
        return self.generate_batch_tensor(width, height, batch_size).cpu().numpy()
    
    def get_config(self):
        return super().get_config() | {
            "density": self.density
        }
    
    def get_state(self) -> dict:
        return super().get_state() | {
            "rng": self.rng.get_state()
        }