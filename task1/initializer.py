import numpy as np 
import random
class BaseInitializer():
    def __init__(self):
        self.seed = random.randint(1, 10000)
        pass

    def init(self, dim):
        pass  

    def seed(self, seed):
        self.seed = seed

class ZeroInitializer(BaseInitializer):
    def __init__(self):
        super().__init__()
    
    def init(self, shape):
        return np.zeros(shape)

class OneInitializer(BaseInitializer):
    def __init__(self):
        super().__init__()
    
    def init(self, shape):
        return np.ones(shape)

class UniformInitializer(BaseInitializer):
    def __init__(self, low, high):
        super().__init__()
        self.low = low 
        self.high = high
    
    def init(self, shape):
        dim = shape[0] * shape[1]
        ret = [np.random.uniform(self.low, self.high) for i in range(dim)]
        return np.array(ret).reshape(shape)