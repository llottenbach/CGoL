import torch

from cgol.simulator.torch_simulator import TorchSimulator
from cgol.simulator.minimal_architecture_model import MinimalArchitectureModel

class MinimalArchitectureSimulator(TorchSimulator):
    def __init__(self, device=torch.device('cpu'), dtype=torch.double):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model = MinimalArchitectureModel(derivable=False, device=device, dtype=dtype)

    def step(self, state):
        self.stepTensor(torch.from_numpy(state).to(device=self.device,dtype=self.dtype)).detach().cpu().numpy()
        
    def stepBatch(self, states):
        self.stepBatchTensor(torch.from_numpy(states).to(device=self.device, dtype=self.dtype)).detach().cpu().numpy()
    
    def stepTensor(self, state):
        return self.model.forward(state[None,:,:])[0]
    
    def stepBatchTensor(self, states):
        return self.model.forward(states[:,None,:,:]).flatten(0,1)