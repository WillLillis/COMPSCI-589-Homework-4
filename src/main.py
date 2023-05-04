import numpy as np
import math
from neural_nets import neural_net


def main():
    #test_nn = neural_net(3, 1, [4])
    #test_nn.print_layers()

    #test_instance = np.array([1,2,3])
    #print(f"{test_instance=}")
    #preds = test_nn.forward_propagation(test_instance)
    #print(f"{preds=}, {np.shape(preds)=}")
    for i in range(5, -1, -1):
        print(i)

    test_matrix = np.array([[1,2,3], [4,5,6], [7,8,9]])
    print(f"{test_matrix=}")
    test_matrix *= 0
    print(f"{test_matrix=}")
    


if __name__ == "__main__":
    main()