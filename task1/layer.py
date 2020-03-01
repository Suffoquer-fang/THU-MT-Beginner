import numpy as np 

from initializer import *

class Layer(): 
    def __init__(self, x):
        super().__init__()
        self.x = x
    def forward(self):
        pass
    def backward(self):
        pass
    def _reshape_(self, x): 
        if not isinstance(self.x, Layer):
            x = np.array(x)
            if len(x.shape) < len(self.x): 
                x = np.array([x])
        x = np.transpose(x)
        return x

class DenseLayer(Layer): 
    def __init__(self, x, dim, w_initializer=OneInitializer(), b_initializer=ZeroInitializer()):
        super().__init__(x)
        self.pre_layer = x 
        self.dim = dim 
        if isinstance(self.pre_layer, Layer): 
            pre_dim = x.dim
        else: 
            pre_dim = x[-1]
        self.W = w_initializer.init((dim, pre_dim))
        self.B = b_initializer.init((dim, 1))
    
    def forward(self, x): 
        if isinstance(self.pre_layer, Layer): 
            x = self.pre_layer.forward(x)
        else: 
            x = self._reshape_(x)

        return np.dot(self.W, x) + self.B



class TanhLayer(Layer):
    def __init__(self, x):
        super().__init__(x)
        self.pre_layer = x 

    def forward(self, x):
        if isinstance(self.pre_layer, Layer): 
            x = self.pre_layer.forward(x)
        else: 
            x = self._reshape_(x)
        return np.tanh(x)
        

class SoftmaxLayer(Layer):
    def __init__(self, x):
        super().__init__(x)
        self.pre_layer = x 

    def forward(self, x):
        if isinstance(self.pre_layer, Layer): 
            x = self.pre_layer.forward(x)
        else: 
            x = self._reshape_(x)
        y = np.exp(x)
        s = np.sum(y)
        return y / s

