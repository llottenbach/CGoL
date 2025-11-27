import torch

from cgol.simulator.torch_simulator import TorchSimulator
from cgol.simulator.minimal_architecture_model import MinimalArchitectureModel

class DerivableMinimalArchitectureSimulator(TorchSimulator):
    def __init__(self, device=torch.device('cpu')):
        super().__init__()
        self.model = MinimalArchitectureModel(device=device)
        self.device = device

    def step(self, state):
        self.step_tensor(torch.from_numpy(state).to(device=self.device)).detach().cpu().numpy()
        
    def step_batch(self, states):
        self.step_batch_tensor(torch.from_numpy(states).to(device=self.device)).detach().cpu().numpy()
    
    def step_tensor(self, state):
        return self.model.forward(torch.Tensor([state]))
    
    def step_batch_tensor(self, states):
        return self.model.forward(states)