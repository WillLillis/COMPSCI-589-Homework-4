import numpy as np
import math

# - going to assume a fully connected NN for sake of simplicity
# - not going to explicitly store the bias neurons, and instead will add them
# in as needed in the forward and backward propagation functions
class neural_net:
    # - input_nodes specifies the number of input nodes
    # - hidden_layers is a list, where each entry is an integer specifying the number
    # of nodes in that layer
    def __init__(self, input_nodes: int, output_nodes: int, hidden_layers: list, activ_func="sigmoid"):
        # sanity checks 
        if input_nodes <= 0:
            raise(f"Invalid parameter: {input_nodes=}")
        if output_nodes <= 0:
            raise(f"Invalid parameter: {output_nodes=}")
        for i in range(len(hidden_layers)):
            if hidden_layers[i] <= 0:
                raise(f"Invalid parameter: {hidden_layers[i]=}")

        self.activation_function = activ_func

        # now that we've done some parameter checks...
        self.weights = list() # list to hold the numpy arrays (make this a numpy array too?)
        # - first create the matrix to hold the weights between 
        # the input layer and first input layer
        weight_tmp = np.array(np.random.normal(0, 1, input_nodes + 1)) # + 1 for bias weights
        for _ in range(hidden_layers[0] - 1):
            weight_tmp = np.vstack((weight_tmp, np.random.normal(0, 1, input_nodes + 1))) # + 1 for bias weights
        self.weights.append(weight_tmp)

        # now for the hidden layers connecting to one another
        for i in range(len(hidden_layers) - 1):
             weight_tmp = np.array(np.random.normal(0, 1, hidden_layers[i] + 1)) # + 1 for bias weights
             for _ in range(hidden_layers[i + 1] - 1):
                weight_tmp = np.vstack((weight_tmp, np.random.normal(0, 1, hidden_layers[i] + 1))) # + 1 for bias weights
             self.weights.append(weight_tmp)

        # and finally the last hidden layer connecting to the output layer
        weight_tmp = np.array(np.random.normal(0, 1, hidden_layers[-1] + 1)) # + 1 for bias weights
        for _ in range(output_nodes - 1):
            weight_tmp = np.vstack((weight_tmp, np.random.normal(0, 1, hidden_layers[-1] + 1))) # + 1 for bias weights
        self.weights.append(weight_tmp)

    # - propagate instance through the network, get np array of output nodes returned
    def forward_propagation(self, instance: np.array) -> np.array:
        activation = np.insert(instance, 0, 1) # add the bias term to the input
        
        # for hidden layers 0...n-1
        for i in range(len(self.weights) - 1):
            activation = np.matmul(self.weights[i], activation) # multiply vector with hidden layer weights
            activation = self.activation_func(activation) # apply activation function g element-wise
            activation = np.insert(activation, 0, 1) # add bias term back in
        
        # apply last hidden layer, get output values
        activation = np.matmul(self.weights[-1], activation) 
        activation = self.activation_func(activation)
        return activation

    # - TODO: implement
    # - allow list of instances?
    def backward_propagation(self, instance: np.array):
        # propagate instance through network
        # compute error for all output neurons
        # for all neurons in hidden layers
            # compute delta values, ignoring bias neurons (first column?)
        # for each network layer
            # compute gradient
        # for each network layer
            # compute final regularized gradient
        pass
        

    def activation_func(self, input: np.array) -> np.array:
        if self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-input))
        else:
            raise(f"Invalid activation function parameter passed! {self.activation_function=}")


    def print_layers(self):
        for layer in self.weights:
            print(np.shape(layer))
            print(layer)


def main():
    pass


if __name__ == "__main__":
    main()