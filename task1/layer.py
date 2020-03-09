import numpy as np

class Layer():
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self.saved_input = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def save_input(self, tensor):
        self.saved_input = tensor


class DenseLayer(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(DenseLayer, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

    def forward(self, input):
        self.save_input(input)
        output = np.dot(input, self.W) + self.b
        return output
        
    def backward(self, grad_output):
        tensor = self.saved_input
        
        self.grad_W = np.dot(tensor.T, grad_output)
        self.grad_b = np.sum(grad_output, axis=0)

        return np.dot(grad_output, self.W.T)

    def update(self, learning_rate):
        lr = learning_rate

        self.W = self.W - lr * self.grad_W
        self.b = self.b - lr * self.grad_b



class TanhLayer(Layer):
    def __init__(self, name):
        super(TanhLayer, self).__init__(name)

    def forward(self, input):
        output = np.tanh(input)
        self.save_input(output)
        return output


    def backward(self, grad_output):
        tensor = self.saved_input
        output = (1 - tensor ** 2) * grad_output
        return np.array(output)

class SoftmaxLayer(Layer):
    def __init__(self, name):
        super(SoftmaxLayer, self).__init__(name)

    def forward(self, input):
        
        exp = np.exp(input)
        exp_sum = np.sum(exp, axis=1)
        exp_sum = np.expand_dims(exp_sum, axis=1)
        h = exp / exp_sum
        self.save_input(h)
        return h


    def backward(self, grad_output):
       
        return grad_output