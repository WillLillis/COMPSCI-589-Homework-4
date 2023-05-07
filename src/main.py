from cmath import cos
from string import printable
import numpy as np
from neural_nets import neural_net
from misc import k_folds_gen

def test_examples():
    #
    # Example 1:
    #
    #   Forward Propagation:
    print("Example 1:")
    test_nn = neural_net(1, 1, [2])
    starting_weights = list([np.array([[0.4, 0.1], [0.3, 0.2]]), np.array([np.array([0.7, 0.5, 0.6])])])
    test_nn.set_weights(starting_weights)
    print("Neural net layers:")
    test_nn.print_layers(info=True)

    instances = list([np.array([0.13]), np.array([0.42])])
    labels = list([np.array([np.array([0.9])]), np.array([np.array([0.23])])])
    preds = list()
    costs = list()
    print(f"Training set:")
    for i in range(len(instances)):
        print(f"Training instance {i + 1}")
        print(f"x: {instances[i]}")
        print(f"y: {labels[i]}")

    print("Computing the error/cost, J, of the network")
    print("Processing training instance 1:")
    preds.append(np.array([test_nn.forward_propagation(instances[0], test=True)])) # a's and z's are printed within the function when the test flag is set
    print(f"Predicted output for instance 1: {preds[0]}")
    print(f"Expected output for instance 1: {labels[0]}")
    costs.append(neural_net.indiv_cost_func(preds[0], labels[0]))
    print(f"Cost, J, associated with instance 1: {costs[0]}")

    print("Processing training instance 2:")
    preds.append(np.array([test_nn.forward_propagation(instances[1], test=True)])) # a's and z's are printed within the function when the test flag is set
    print(f"Predicted output for instance 2: {preds[1]}")
    print(f"Expected output for instance 2: {labels[1]}")
    costs.append(neural_net.indiv_cost_func(preds[1], labels[1]))
    print(f"Cost, J, associated with instance 2: {costs[1]}")

    total_cost = test_nn.reg_cost_func(preds, labels)
    print(f"Final (regularized) cost, J, based on the complete training set: {total_cost}")

    #   Backward Propagation:
    print("Running backpropagation:")
    test_nn.backward_propagation(instances, labels, test=True) # intermediate values are printed within the function when the test flag is set

    #
    # Example 2:
    #
    #   Forward Propagation:
    print("\n\n\nExample 2:")
    test_nn = neural_net(2, 2, [4, 3], 0.25)
    starting_weights = list([np.array([[0.42, 0.15, 0.4], [0.72, 0.1, 0.54], [0.01, 0.19, 0.42], [0.3, 0.35, 0.68]]), \
       np.array([[0.21, 0.67, 0.14, 0.96, 0.87], [0.87, 0.42, 0.2, 0.32, 0.89], [0.03, 0.56, 0.8, 0.69, 0.09]]), \
       np.array([[0.04, 0.87, 0.42, 0.53], [0.17, 0.1, 0.95, 0.69]])])
    test_nn.set_weights(starting_weights)
    print("Neural net layers:")
    test_nn.print_layers(info=True)

    instances = list([np.array([0.32, 0.68]), np.array([0.83, 0.02])])
    labels = list([np.array([0.75, 0.98]), np.array([0.75, 0.28])])
    preds = list()
    costs = list()
    print(f"Training set:")
    for i in range(len(instances)):
        print(f"Training instance {i + 1}")
        print(f"x: {instances[i]}")
        print(f"y: {labels[i]}")

    preds.append(test_nn.forward_propagation(instances[0], test=True)) # a's and z's are printed within the function when the test flag is set
    print(f"Predicted output for instance 1: {preds[0]}")
    print(f"Expected output for instance 1: {labels[0]}")
    costs.append(neural_net.indiv_cost_func(preds[0], labels[0]))
    print(f"Cost, J, associated with instance 1: {costs[0]}")

    preds.append(test_nn.forward_propagation(instances[1], test=True)) # a's and z's are printed within the function when the test flag is set
    print(f"Predicted output for instance 1: {preds[1]}")
    print(f"Expected output for instance 1: {labels[1]}")
    costs.append(neural_net.indiv_cost_func(preds[1], labels[1]))
    print(f"Cost, J, associated with instance 2: {costs[1]}")

    total_cost = test_nn.reg_cost_func(preds, labels)
    print(f"Final (regularized) cost, J, based on the complete training set: {total_cost}")

    #   Backward Propagation:
    print("Running backpropagation:")
    test_nn.backward_propagation(instances, labels, test=True) # intermediate values are printed within the function when the test flag is set

def test_congress(hidden_layers: list, num_iterations: int) -> None:
    k_folds_instances, k_folds_labels = k_folds_gen(1, "hw3_house_votes_84.csv")
    test_nn = neural_net(16, 2, hidden_layers)

    # process k_folds stuff here...
    k_folds_instances = k_folds_instances[0]
    k_folds_labels = k_folds_labels[0]

    for _ in range(num_iterations):
        test_nn.backward_propagation(k_folds_instances, k_folds_labels)

    num_instances = 0
    num_correct = 0
    for index in range(len(k_folds_instances)):
        num_instances += 1
        pred = test_nn.forward_propagation(k_folds_instances[index])
        pred = np.argmax(pred) + 1
        label = np.argmax(k_folds_labels[index][0]) + 1
        if pred == label:
            num_correct += 1

    print(f"Congressional Dataset Accuracy: {num_correct / num_instances}")

def test_wine(hidden_layers: list, num_iterations: int) -> None:
    k_folds_instances, k_folds_labels = k_folds_gen(1, "hw3_wine.csv")
    test_nn = neural_net(13, 3, hidden_layers)

    k_folds_instances = k_folds_instances[0]
    k_folds_labels = k_folds_labels[0]

    for _ in range(num_iterations):
        test_nn.backward_propagation(k_folds_instances, k_folds_labels)

    num_instances = 0
    num_correct = 0
    for index in range(len(k_folds_instances)):
        num_instances += 1
        pred = test_nn.forward_propagation(k_folds_instances[index])
        pred = np.argmax(pred) + 1
        label = np.argmax(k_folds_labels[index][0]) + 1
        if pred == label:
            num_correct += 1

    print(f"Wine Dataset Accuracy: {num_correct / num_instances}")




def main():
    #k_folds_instances, k_folds_labels = k_folds_gen(10, "hw3_house_votes_84.csv", False)
    #k_folds_instances, k_folds_labels = k_folds_gen(1, "hw3_house_votes_84.csv", False)
    k_folds_instances, k_folds_labels = k_folds_gen(1, "hw3_wine.csv")

    k_folds_instances = k_folds_instances[0]
    k_folds_labels = k_folds_labels[0]

    #test_nn = neural_net(16, 2, [3, 2])
    test_nn = neural_net(13, 3, [10, 10, 10])

    #starting_weights = list([np.array([[-1.65219515,  1.92639498, -0.3204609 , -0.73464556,  0.10543864,\
    #     0.64566062, -0.20451444,  0.3466626 , -0.67101306, -0.19561037,\
    #     0.10189969,  1.82879834,  0.11216628, -0.57861481, -0.89392526,\
    #     0.92091756,  0.22336052]]), np.array([[ 0.59783266,  0.83475445], [-0.82773738, -1.55786769]])])
    #
    #if test_nn.set_weights(starting_weights) == False:
    #    print("Sadness")
    #    return

    #print(f"{k_folds_instances=}")
    #return

    num_instances = 0
    num_correct = 0
    for index in range(len(k_folds_instances)):
        num_instances += 1
        pred = test_nn.forward_propagation(k_folds_instances[index])
        #print(f"{index}: {pred=}, {k_folds_labels[index]=}")
        #pred = np.argmax(pred)
        #print(f"Vals: {pred=}")
        pred = np.argmax(pred) + 1
        #print(f"{pred=}")
        #label = np.argmax(k_folds_labels[index][0])
        label = np.argmax(k_folds_labels[index][0]) + 1
        #print(f"{index}: {pred=}, {label=}")
        if pred == label:
            num_correct += 1
        #pred = 0 if pred[0] >= pred[1] else 1
        #label = 0 if k_folds_labels[index][0][0] == 1 else 1
        ##print(f"\t{pred=}")
        ##print(f"\t{label=}")
        #if pred == label:
        #    num_correct += 1

    print(f"(Before Training) Accuracy: {num_correct / num_instances}")

    ##test_nn.print_layers(dims=True)
    for _ in range(100):
        test_nn.backward_propagation(k_folds_instances, k_folds_labels)
    ##test_nn.print_layers(dims=True)

    num_instances = 0
    num_correct = 0
    for index in range(len(k_folds_instances)):
        num_instances += 1
        pred = test_nn.forward_propagation(k_folds_instances[index])
        #print(f"{index}: {pred=}, {k_folds_labels[index]=}")
        #pred = np.argmax(pred)
        #print(f"Vals: {pred=}")
        pred = np.argmax(pred) + 1
        #print(f"{pred=}")
        #label = np.argmax(k_folds_labels[index][0])
        label = np.argmax(k_folds_labels[index][0]) + 1
        #print(f"{index}: {pred=}, {label=}")
        if pred == label:
            num_correct += 1
    #    #pred = 0 if pred[0] >= pred[1] else 1
    #    #label = 0 if k_folds_labels[index][0][0] == 1 else 1
    #    ##print(f"\t{pred=}")
    #    ##print(f"\t{label=}")
    #    #if pred == label:
    #    #    num_correct += 1

    print(f"(After Training) Accuracy: {num_correct / num_instances}")





if __name__ == "__main__":
    #main()
    #test_examples()
    test_congress(list([3,2]), 500)
    test_congress(list([10,10,10]), 500)