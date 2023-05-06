from tracemalloc import start
import numpy as np
from neural_nets import neural_net

def test_examples():
    #
    # Example 1:
    #
    #   Forward Propagation:
    print("Example 1:")
    test_nn = neural_net(1, 1, [2], 0.0)
    starting_weights = list([np.array([[0.4, 0.1], [0.3, 0.2]]), np.array([np.array([0.7, 0.5, 0.6])])])
    test_nn.set_weights(starting_weights)
    print("Neural net layers:")
    test_nn.print_layers()

    instances = list([np.array([0.13]), np.array([0.42])])
    labels = list([np.array([0.9]), np.array([0.23])])
    preds = list()
    
    #preds.append(np.array([test_nn.forward_propagation(instances[0], test=True)])) # a's and z's are printed within the function when the test flag is set
    #print(f"Predicted output for instance 1: {preds[0]}")
    #print(f"Expected output for instance 1: {labels[0]}")
    #cost_1 = neural_net.indiv_cost_func(preds[0], labels[0])
    #print(f"Cost, J, associated with instance 1: {cost_1}")

    #preds.append(np.array([test_nn.forward_propagation(instances[1], test=True)])) # a's and z's are printed within the function when the test flag is set
    #print(f"Predicted output for instance 1: {preds[1]}")
    #print(f"Expected output for instance 1: {labels[1]}")
    #cost_2 = neural_net.indiv_cost_func(preds[1], labels[1])
    #print(f"Cost, J, associated with instance 2: {cost_2}")

    #total_cost = test_nn.reg_cost_func(preds, labels)
    #print(f"Final (regularized) cost, J, based on the complete training set: {total_cost}")

    test_nn.backward_propagation(instances, labels, test=True)

    return

    #
    # Example 2:
    #
    #   Forward Propagation:
    print("Example 2:")
    test_nn = neural_net(2, 2, [4, 3], 0.25)
    starting_weights = list([np.array([[0.42, 0.15, 0.4], [0.72, 0.1, 0.54], [0.01, 0.19, 0.42], [0.3, 0.35, 0.68]]), \
       np.array([[0.21, 0.67, 0.14, 0.96, 0.87], [0.87, 0.42, 0.2, 0.32, 0.89], [0.03, 0.56, 0.8, 0.69, 0.09]]), \
       np.array([[0.04, 0.87, 0.42, 0.53], [0.17, 0.1, 0.95, 0.69]])])
    test_nn.set_weights(starting_weights)
    print("Neural net layers:")
    test_nn.print_layers()

    instances = list([np.array([0.32, 0.68]), np.array([0.83, 0.02])])
    labels = list([np.array([0.75, 0.98]), np.array([0.75, 0.28])])
    preds = list()

    preds.append(test_nn.forward_propagation(instances[0], test=True)) # a's and z's are printed within the function when the test flag is set
    print(f"Predicted output for instance 1: {preds[0]}")
    print(f"Expected output for instance 1: {labels[0]}")
    cost_1 = neural_net.indiv_cost_func(preds[0], labels[0])
    print(f"Cost, J, associated with instance 1: {cost_1}")

    preds.append(test_nn.forward_propagation(instances[1], test=True)) # a's and z's are printed within the function when the test flag is set
    print(f"Predicted output for instance 1: {preds[1]}")
    print(f"Expected output for instance 1: {labels[1]}")
    cost_2 = neural_net.indiv_cost_func(preds[1], labels[1])
    print(f"Cost, J, associated with instance 2: {cost_2}")

    total_cost = test_nn.reg_cost_func(preds, labels)
    print(f"Final (regularized) cost, J, based on the complete training set: {total_cost}")


def main():
    #starting_weights = list([np.array([[0.42, 0.15, 0.4], [0.72, 0.1, 0.54], [0.01, 0.19, 0.42], [0.3, 0.35, 0.68]]), \
    #   np.array([[0.21, 0.67, 0.14, 0.96, 0.87], [0.87, 0.42, 0.2, 0.32, 0.89], [0.03, 0.56, 0.8, 0.69, 0.09]]), \
    #   np.array([[0.04, 0.87, 0.42, 0.53], [0.17, 0.1, 0.95, 0.69]])])

    #print(f"{np.shape(starting_weights[0])=}")
    #print(f"{np.shape(starting_weights[0])[0]=}")
    test = list()
    test.append(5)
    test.append(5)
    print(f"{test[-1]=}")
    tmp = len(test) -1 
    print(f"test[{tmp}]={test[tmp]}")

    print(f"{5 + np.nan}")
    test_arr = np.array([1,2,3])
    print(f"{np.shape(test_arr)}")
    print(f"{len(np.shape(test_arr))}")


if __name__ == "__main__":
    #main()
    test_examples()