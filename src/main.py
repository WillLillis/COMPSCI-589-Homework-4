import numpy as np
import math
from neural_nets import neural_net


def main():
    test_nn = neural_net(3, 1, [4])
    test_nn.print_layers()

    test_instance = np.array([1,2,3])
    print(f"{test_instance=}")
    preds = test_nn.forward_propagation(test_instance)
    print(f"{preds=}, {np.shape(preds)=}")


if __name__ == "__main__":
    main()