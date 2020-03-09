import numpy as np 
from layer import * 
from loss import MSELoss, SoftmaxCrossEntropyLoss, CrossEntropyLoss
from model import Model
from utils import *
import pickle


def evaluate(model, data):
    x_data = data['x']
    y_data = data['y']

    batch_size = 100
    size = len(x_data)
    correct = 0
    for start_idx in range(0, size, batch_size):
        end_idx = min(start_idx + batch_size, size)
        x = np.array(x_data[start_idx: end_idx])
        y = y_data[start_idx: end_idx]
        output = softmax(model.forward(x))
        correct += len(y) * calculate_acc(output, y)

    return correct / size



if __name__ == "__main__":
    np.random.seed(13)
    

    with open('data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    with open('validation.pkl', 'rb') as f:
        val_data = pickle.load(f)

    with open('test.pkl', 'rb') as f:
        test_data = pickle.load(f)

    x_data = train_data['x']
    y_data = train_data['y']

    model = Model()
    f1 = DenseLayer('f1', 5, 5, 1)
    f2 = SoftmaxLayer('f2')
    f3 = TanhLayer('f3')
    model.addLayer(f1)
    # model.addLayer(f3)
    # model.addLayer(f2)


    for i in range(10000):
        start = i % 100
        batch_size = 100
        x = x_data[start*batch_size:start*batch_size+batch_size]
        label = y_data[start*batch_size:start*batch_size+batch_size]
        
        y = onehot_encoding(label, 5)

        x = np.array(x)
        
        ans = model.forward(x)
        # print('x', x)
        loss = SoftmaxCrossEntropyLoss('loss')
        print('acc', calculate_acc(softmax(ans), label), end='  ')
        print('loss', loss.forward(ans, y))
        # print('ans', softmax(ans))
        grad = loss.backward(ans, y)
        model.backward(grad)
        model.update()

    print(evaluate(model, test_data))

    


