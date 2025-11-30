from typing import Iterable

import torch
from torch import Tensor
from cgol.generator.torch_generator import TorchGenerator
from cgol.simulator.torch_simulator import TorchSimulator

class Dataloader2:
    def __init__(self, 
                 generator: TorchGenerator, 
                 simulator: TorchSimulator,
                 batch_size: int,
                 width: int,
                 height: int,
                 preprocess_device = "cpu",
                 output_device = "cpu",
                 dtype = torch.double,
                 min_change_threshold: float = 0.1,
                 max_sequence_age: int = 150):
        
        self.generator: TorchGenerator = generator
        self.simulator: TorchSimulator = simulator

        self.batch_size: int = batch_size
        self.width: int = width
        self.height: int = height
        self.preprocess_device = preprocess_device
        self.output_device = output_device
        self.dtype = dtype
        self.min_change_threshold = min_change_threshold
        self.max_sequence_age = max_sequence_age
        self.step = 0

        self.last_batch = torch.zeros(
            (self.batch_size, self.height, self.width), 
            dtype=self.dtype, 
            device=self.preprocess_device)

        self.sequence_ages = torch.zeros(
            (self.batch_size, ),
            dtype=torch.int,
            device=self.preprocess_device)

    def get_config(self) -> dict:
        return {
            "type": "Dataloader1",
            "generator": self.generator.get_config(),
            "width": self.width,
            "height": self.height,
            "batch_size": self.batch_size,
            "min_change_threshold": self.min_change_threshold,
            "max_sequence_age": self.max_sequence_age
        }
    
    def get_state(self) -> dict:
        return {
            "generator": self.generator.get_state(),
            "last_batch": self.last_batch,
            "sequence_ages": self.sequence_ages,
            "step": self.step
        }

    def __iter__(self) -> Iterable[Tensor]:
        return self
    
    def __next__(self) -> Tensor:
        new_batch = self.simulator.step_batch_tensor(self.last_batch)
        self.sequence_ages += 1
        self.batch_diffs_per_cell = self._get_batch_diffs_per_cell(self.last_batch, new_batch)
        i_reset = self._get_reseting_indexes(self.batch_diffs_per_cell)
        n_reset = i_reset.sum()

        while n_reset > 0:
            self.last_batch[i_reset] = self.generator.generate_batch_tensor(self.width, self.height, n_reset)
            self.sequence_ages[i_reset] = 0

            new_batch[i_reset] = self.simulator.step_batch_tensor(self.last_batch[i_reset])
            self.sequence_ages[i_reset] += 1
            self.batch_diffs_per_cell = self._get_batch_diffs_per_cell(self.last_batch, new_batch)
            i_reset = self._get_reseting_indexes(self.batch_diffs_per_cell)
            n_reset = i_reset.sum()

        output = torch.stack((new_batch, self.last_batch), dim=0)
        self.last_batch = new_batch
        self.step += 1
        return output.to(device=self.output_device, dtype=self.dtype)
        #raise StopIteration()

    def _get_batch_diffs_per_cell(self, old_batch: Tensor, new_batch: Tensor) -> Tensor:
        batch_diffs = (old_batch - new_batch).abs()
        return (batch_diffs.sum(dim=(-1,-2),dtype=self.dtype).detach() 
                            / (old_batch.shape[-1] * old_batch.shape[-2]))

    def _get_reseting_indexes(self, batch_diffs_per_cell: Tensor):
        return ((batch_diffs_per_cell < self.min_change_threshold)
                .logical_or(self.max_sequence_age < self.sequence_ages))


