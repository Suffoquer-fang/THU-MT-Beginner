import numpy as np 
import random
import pickle

def generate_data(size=10000, name='data'):
    ret = {'x':[], 'y':[]}
    for d in range(size):
        x = [np.random.uniform(-1, 1) for i in range(5)]
        x = np.around(x, decimals=3)
        
        y = np.argmax(x)
        print(x)
        ret['x'].append(x)
        ret['y'].append(y)
        print('generate %d done'%d)
    with open('%s.pkl'%name, 'wb') as f:
        pickle.dump(ret, f)

def onehot_encoding(label, max_num_class):
    encoding = np.eye(max_num_class)
    encoding = encoding[label]
    return encoding


def calculate_acc(output, label):
    correct = np.sum(np.argmax(output, axis=1) == label)
    return correct / len(label)

def softmax(input):
    exp = np.exp(input)
    exp_sum = np.sum(exp, axis=1)
    exp_sum = np.expand_dims(exp_sum, axis=1)
    return exp / exp_sum

if __name__ == "__main__":
    np.random.seed(12345)
    # np.random.seed(54321) #val
    # np.random.seed(66666) #test
    generate_data(50000, 'train')