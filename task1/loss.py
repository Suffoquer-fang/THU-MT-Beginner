import numpy as np 
def MSELoss(output, label):
    output = np.transpose(output)

    N = output.shape[0]

    error = output - label 

    loss = (error * error)
    loss = np.sum(loss, axis=1) / 2
    loss = np.sum(loss) / N

    return loss 