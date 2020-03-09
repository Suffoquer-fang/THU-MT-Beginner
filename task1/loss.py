import numpy as np 
class MSELoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        error = np.sum(((input - target) ** 2), axis=1)
        return 0.5 * np.mean(error, axis=0)

    def backward(self, input, target):
        return (input - target) / len(target)

class CrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        h = input
        error = np.sum(-1 * target * np.log(h), axis=1)
        output = np.mean(error)
        return output

    def backward(self, input, target):
        h = -1 / input

        output = h * target / len(target)

        return np.array(output)


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):

        exp = np.exp(input)
        exp_sum = np.sum(exp, axis=1)
        exp_sum = np.expand_dims(exp_sum, axis=1)
        h = exp / exp_sum
        
        error = np.sum(-1 * target * np.log(h), axis=1)
        output = np.mean(error)
        return output

    def backward(self, input, target):
        exp = np.exp(input)
        exp_sum = np.sum(exp, axis=1)
        exp_sum = np.expand_dims(exp_sum, axis=1)
        h = exp / exp_sum
      
        output = -1 * (target - h) / len(target)

        return output