import numpy as np

class Generator:
    def __init__(self):
        self.seed = 0

    def generate(self, width: int, height: int) -> np.ndarray:
        pass
        
    def generate_batch(self, width: int, height: int, batch_size: int) -> np.ndarray:
        pass

    def get_config(self) -> dict:
        return {
            "type": type(self).__name__,
            "seed": self.seed
        }
    
    def get_state(self) -> dict:
        return {}

    def set_state(dict):
        pass