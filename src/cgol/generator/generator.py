import numpy as np

class Generator:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.batch_size = 0
        self.seed = 0

    def generate(self) -> np.ndarray:
        pass
        
    def generateBatch(self) -> np.ndarray:
        pass