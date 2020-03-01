import numpy as np 

from initializer import *

class Layer(): 
    def __init__(self):
        super().__init__()
    def forward(self):
        pass
    def backward(self):
        pass

class DenseLayer(Layer): 
    def __init__(self, x, dim, w_initializer=OneInitializer(), b_initializer=ZeroInitializer()):
        super().__init__()
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
            x = np.array([x])
            x = np.transpose(x)
        return np.dot(self.W, x) + self.B



class TanhLayer(Layer):
    def __init__(self, x):
        super().__init__()
        self.pre_layer = x 

    def forward(self, x):
        if isinstance(self.pre_layer, Layer): 
            x = self.pre_layer.forward(x)
        else: 
            x = np.array([x])
            x = np.transpose(x)
        return np.tanh(x)
        

class SoftmaxLayer(Layer):
    def __init__(self, x):
        super().__init__()
        self.pre_layer = x 

    def forward(self, x):
        if isinstance(self.pre_layer, Layer): 
            x = self.pre_layer.forward(x)
        else: 
            x = np.array([x])
            x = np.transpose(x)
        y = np.exp(x)
        s = np.sum(y)
        return y / s