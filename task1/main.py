import numpy as np 
from layer import * 
from loss import MSELoss

if __name__ == "__main__":
    np.random.seed(13)
    x_shape = [None, 5]
    # f1 = DenseLayer(x_shape, 5, UniformInitializer(-4, 4))
    # f2 = DenseLayer(f1, 128, UniformInitializer(-4, 4))
    # f2 = DenseLayer(f2, 5, UniformInitializer(-4, 4))
    # f3 = TanhLayer(f2)
    # f4 = SoftmaxLayer(f3)
    
    f4 = DenseLayer(x_shape, 5)


    x1 = [0.01, 0.01, 0.03, 0.04, 0.05]
    x2 = [1, 2, 3, 4, 5]

    x = [x1, x2]
    
    print(np.array(x).shape)
    # print(f3.forward(x1))
    print(f4.forward(x1))

    
    for i in range(100):
        ans = f4.forward(x1)
        

        print(np.transpose(ans))

        label1 = [0, 0, 0, 0, 1]
        label2 = [0, 0, 0, 1, 1]

        label = [label1, label2]

        print(MSELoss(ans, label1))
        print(np.transpose([label1]))
        f4.backward(np.transpose([label1]))


