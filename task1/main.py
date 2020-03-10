import numpy as np 
from layer import * 
from loss import MSELoss, SoftmaxCrossEntropyLoss, CrossEntropyLoss
from model import Model
from utils import *
import matplotlib.pyplot as plt
import pickle


def evaluate(model, data):
    x_data = data['x']
    y_data = data['y']

    batch_size = 100
    size = len(x_data)
    correct = 0
    loss_value = 0
    loss = SoftmaxCrossEntropyLoss('loss')
    for start_idx in range(0, size, batch_size):
        end_idx = min(start_idx + batch_size, size)
        x = np.array(x_data[start_idx: end_idx])
        y = y_data[start_idx: end_idx]

        ans = model.forward(x)
        output = softmax(ans)

        loss_value += len(y) * loss.forward(ans, onehot_encoding(y, 5))
        correct += len(y) * calculate_acc(output, y)

    return loss_value / size, correct / size

def build_model():
    model = Model()
    f1 = DenseLayer('f1', 5, 5, 1)
    f2 = SoftmaxLayer('f2')
    f3 = TanhLayer('f3')
    model.addLayer(f1)
    # model.addLayer(f3)
    return model

def plot(logs):
    x = [i['step'] for i in logs]
    y = [i['train_acc'] for i in logs]
    plt.plot(x, y, label='train_acc')
    y = [i['val_acc'] for i in logs]
    plt.plot(x, y, label='val_acc')
    plt.legend()
    plt.show()
    y = [i['train_loss'] for i in logs]
    plt.plot(x, y, label='train_loss')
    y = [i['val_loss'] for i in logs]
    plt.plot(x, y, label='val_loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    np.random.seed(13)
    

    with open('train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    with open('validation.pkl', 'rb') as f:
        val_data = pickle.load(f)

    with open('data.pkl', 'rb') as f:
        test_data = pickle.load(f)

    x_data = train_data['x']
    y_data = train_data['y']

    model = build_model()
    batch_size = 128
    size = len(x_data)
    global_step = 0

    epoch = 2

    loss = SoftmaxCrossEntropyLoss('loss')

    logs = []
    for i in range(epoch):
        
        
        for start_idx in range(0, size, batch_size):
            end_idx = min(start_idx + batch_size, size)
            x = np.array(x_data[start_idx: end_idx])
            label = y_data[start_idx: end_idx]
            y = onehot_encoding(label, 5)

            x = np.array(x)
            
            

            
            
            val_loss, val_acc = 0, 0
            val_loss, val_acc = evaluate(model, val_data)

            ans = model.forward(x)
            train_acc = calculate_acc(softmax(ans), label)
            train_loss = loss.forward(ans, y)

            log_dict = {'step': global_step, 'train_loss': train_loss, 'train_acc':train_acc, 'val_loss':val_loss, 'val_acc':val_acc}

            logs.append(log_dict)

            msg = 'epoch: %d  steps: %d \n train_loss: %.3f   train_acc: %.3f \n valid_loss: %.3f   valid_acc: %.3f'%(i, global_step, train_loss, train_acc, val_loss, val_acc)
            print(msg)

            grad = loss.backward(ans, y)
            model.backward(grad)
            model.update()

            global_step += len(label)

    plot(logs)
    


