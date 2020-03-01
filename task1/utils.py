import numpy as np 
import random
import pickle

def generate_data(size=10000):
    ret = {'x':[], 'y':[]}
    for d in range(size):
        x = [np.random.uniform(-1, 1) for i in range(5)]
        x = np.around(x, decimals=3)
        
        y = np.argmax(x)
        print(x)
        ret['x'].append(x)
        ret['y'].append(y)
        print('generate %d done'%d)
    with open('data.pkl', 'wb') as f:
        pickle.dump(ret, f)


    

if __name__ == "__main__":
    np.random.seed(12345)
    generate_data()