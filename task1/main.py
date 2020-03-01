import numpy as np 
from layer import * 


if __name__ == "__main__":
    np.random.seed(13)
    x_shape = [None, 5]
    f1 = DenseLayer(x_shape, 256, UniformInitializer(-4, 4))
    f2 = DenseLayer(f1, 128, UniformInitializer(-4, 4))
    f2 = DenseLayer(f2, 5, UniformInitializer(-4, 4))
    f3 = TanhLayer(f2)
    f4 = SoftmaxLayer(f3)

    
    x1 = [0.01, 0.01, 0.03, 0.04, 0.05]
    
    print(np.array(x1).shape)
    # print(f3.forward(x1))
    print(f4.forward(x1))


