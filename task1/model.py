from layer import *
from loss import *
from utils import *


class Model(object):
    def __init__(self):
        super().__init__()
        self.layers = []
        self.num_layers = 0
        self.loss = None
        self.learning_rate = 0.1

    def addLayer(self, layer):
        self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, grad_output):
        grad_input = grad_output
        for layer in self.layers[::-1]:
            grad_input = layer.backward(grad_input)
        return grad_input

    def update(self):
        for layer in self.layers:
            if layer.trainable:
                layer.update(self.learning_rate)
