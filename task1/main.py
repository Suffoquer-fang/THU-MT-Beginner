import numpy as np 
from layer import * 


if __name__ == "__main__":
    np.random.seed(123)
    x_shape = [None, 5]
    f1 = DenseLayer(x_shape, 3, UniformInitializer(-4, 4))
    f2 = DenseLayer(f1, 5, UniformInitializer(-4, 4))
    f3 = TanhLayer(f2)
    f4 = SoftmaxLayer(f3)

    
    x = [0.01, 0.01, 0.03, 0.04, 0.05]
    # print(f1.forward(x))
    # print(f2.forward(x))
    # print(f3.forward(x))
    print(f4.forward(x))


