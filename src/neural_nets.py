from copy import deepcopy
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

    # same as forward_propagation but also returns the activations for each hidden layer
    # going to keep the two separate for performance reasons
    def forward_propagation_ret_act(self, instance: np.array) -> np.array:
        ret_activ = list()
        activation = np.insert(instance, 0, 1) # add the bias term to the input

        # for hidden layers 0...n-1
        for i in range(len(self.weights) - 1):
            activation = np.matmul(self.weights[i], activation) # multiply vector with hidden layer weights
            activation = self.activation_func(activation) # apply activation function g element-wise
            activation = np.insert(activation, 0, 1) # add bias term back in
            ret_activ.append(activation)
        
        # apply last hidden layer, get output values
        activation = np.matmul(self.weights[-1], activation) 
        activation = self.activation_func(activation)
        return activation, ret_activ

    # - TODO: implement
    # - allow list of instances?
    def backward_propagation(self, instances: np.array, labels: np.array, lambda_: float, alpha: float):
        # initialize D to all 0's, same dimension as the weights
        grads = deepcopy(self.weights) 
        for layer in grads:
            layer *= 0

        for instance, label in instances, labels:
            preds, activations = self.forward_propagation_ret_act(instance)
            deltas = list()
            regularizers = list()
            for _ in range(len(self.weights)):
                deltas.append([])
                regularizers.append([])
            deltas[-1] = preds - label 
            # "For each network layer, k = L - 1 ... 2"
            for k in range(len(self.weights) - 1, 0, -1): # confusing indices...
                tmp = np.matmul(np.transpose(self.weights[k]), deltas[k + 1]) # = weights^T x delta from previous layer ...
                tmp = np.multiply(tmp, activations[k]) # (element-wise) ... *  activation of current layer 
                tmp = np.multiply(tmp, 1 - activations[k]) # (element-wise) ... * (1 - activation of current layer)
                tmp = np.delete(tmp, 0) # remove first element (bias neuron)
                deltas[k] = tmp
            
            # "For each network layer, k = L - 1 ... 1"
            for k in range(len(self.weights) - 1, -1, -1): # confusing indices...
                grads[k] += np.matmul(deltas[k + 1], np.transpose(activations[k + 1])) # accumulates, in D(l=k), the gradients computed based on the current training instance
        # "For each network layer, k= L - 1 ... 1"
        for k in range(len(self.weights) - 1, -1, -1):
            regularizers[k] = lambda_ * self.weights[k]
            grads[k] = (1 / len(instances)) * (grads[k] + regularizers[k])
        # "At this point, D^(l=1) contains the gradients of the weights θ(l=1); (…); and D(l=L-1) contains the gradients of the weights θ(l=L-1)"
        # "For each network layer, k = L - 1 ... 1"
        for k in range(len(self.weights) - 1, -1, -1):
            self.weights[k] -= alpha * grads[k]        

    def activation_func(self, input_arr: np.array) -> np.array:
        if self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-input_arr))
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