from cmath import nan
from copy import deepcopy
import numpy as np

# - going to assume a fully connected NN for sake of simplicity
class neural_net:
    # - input_nodes : specifies the number of input nodes
    # - hidden_layers : a list, where each entry is an integer specifying the number
    # of nodes in that layer
    # - lambda_ : regularization term
    # - activ_func : which activation function to use
    def __init__(self, input_nodes: int, output_nodes: int, hidden_layers: list, lambda_: float, activ_func="sigmoid"):
        # sanity checks 
        if input_nodes <= 0:
            raise(f"Invalid parameter: {input_nodes=}")
        if output_nodes <= 0:
            raise(f"Invalid parameter: {output_nodes=}")
        for i in range(len(hidden_layers)):
            if hidden_layers[i] <= 0:
                raise(f"Invalid parameter: {hidden_layers[i]=}")

        self.activation_function = activ_func
        self.lambda_ = lambda_

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
    def forward_propagation(self, instance: np.array, test=False) -> np.array:
        activation = np.insert(instance, 0, 1) # add the bias term to the input
        if test == True:
            print(f"a_1: {activation}")

        # for hidden layers 0...n-1
        for i in range(len(self.weights) - 1):
            activation = np.matmul(self.weights[i], activation) # multiply vector with hidden layer weights
            if test == True:
                print(f"z_{i + 2}: {activation}") # the example files use 1 based indexing, hence the + 2....
            activation = self.activation_func(activation) # apply activation function g element-wise
            activation = np.insert(activation, 0, 1) # add bias term back in
            if test == True:
                print(f"a_{i + 2}: {activation}")
        
        # apply last hidden layer, get output values
        activation = np.matmul(self.weights[-1], activation) 
        if test == True:
            print(f"z_{len(self.weights) + 1}: {activation}")
        activation = self.activation_func(activation)
        if test == True:
            print(f"a_{len(self.weights) + 1}: {activation}")
        if test == True:
            print(f"f(x): {activation}")
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

    # - TODO: test
    def backward_propagation(self, instances: np.array, labels: np.array, lambda_: float, alpha: float):
        # initialize D to all 0's, same dimension as the weights
        grads = deepcopy(self.weights) # just copying to get the dimensions right
        for layer in grads:
            layer *= 0

        for instance, label in instances, labels:
            preds, activations = self.forward_propagation_ret_act(instance)
            deltas = list()
            regularizers = list()
            # Need lists with full length because we're indexing backwards
                # - going to initialize with None so it'll be easier to tell 
                # when we don't populate an index and then try to use it
            for _ in range(len(self.weights)):
                deltas.append(None)
                regularizers.append(None)
            deltas[-1] = preds - label 
            # "For each network layer, k = L - 1...2"
            for k in range(len(self.weights) - 1, 0, -1): # confusing indices...
                tmp = np.matmul(np.transpose(self.weights[k]), deltas[k + 1]) # = weights^T x delta from previous layer ...
                tmp = np.multiply(tmp, activations[k]) # (element-wise) ... *  activation of current layer 
                tmp = np.multiply(tmp, 1 - activations[k]) # (element-wise) ... * (1 - activation of current layer)
                tmp = np.delete(tmp, 0) # remove first element (bias neuron)
                deltas[k] = tmp
            
            # "For each network layer, k = L - 1...1"
            for k in range(len(self.weights) - 1, -1, -1): # confusing indices...
                grads[k] += np.matmul(deltas[k + 1], np.transpose(activations[k + 1])) # accumulates, in D(l=k), the gradients computed based on the current training instance
        # "For each network layer, k= L - 1...1"
        for k in range(len(self.weights) - 1, -1, -1):
            regularizers[k] = lambda_ * self.weights[k]
            grads[k] = (1 / len(instances)) * (grads[k] + regularizers[k])
        # "At this point, D^(l=1) contains the gradients of the weights θ(l=1); (…); and D(l=L-1) contains the gradients of the weights θ(l=L-1)"
        # "For each network layer, k = L - 1...1"
        for k in range(len(self.weights) - 1, -1, -1): # confusing indices...
            self.weights[k] -= alpha * grads[k]

    def activation_func(self, input_arr: np.array) -> np.array:
        if self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-input_arr))
        elif self.activation_function == "ReLU":
            return [max(0, x) for x in input_arr]
        else:
            raise(f"Invalid activation function parameter passed! {self.activation_function=}")
    
    # computes cost for INDIVIDUAL prediction and expected output
    @staticmethod
    def indiv_cost_func(pred: np.array, label: np.array) -> float:
        # check dimensions
        if np.shape(pred) != np.shape(label):
            print(f"Error: Mismatching dimensions betweent the prediction and labels arrays! {np.shape(pred)}, {np.shape(label)}")
            raise("Dimension error!")
        accum = 0
        for i in range(len(pred)):
            #print(f"{pred[i]=}, {label[i]=}")
            #print(f"+= {-label[i]} ln({pred[i]}) - (1 - {label[i]})ln(1 - {pred[i]})")
            accum += (-label[i] * (np.log(pred[i]))) - (1 - label[i]) * (np.log(1 - pred[i]))
        return accum


    # figure out wtf is going on
    # preds and labels are lists containing np.array's
    def reg_cost_func(self, preds: list, labels: list) -> float:
        # check lengths of lists
        if len(preds) != len(labels):
            print(f"ERROR! Mismatching lengths for prediction and label lists. ({len(preds)=}, {len(labels)=})")
            return nan

        accum = 0
        # iterate through preds, and labels, get indiviudal costs into accum
        for i in range(len(preds)):
            accum += neural_net.indiv_cost_func(preds[i], labels[i])

        accum /= len(preds)

        if self.lambda_ == 0:
            return accum
        # add up square of all weights except bias weights
        reg = 0
        for layer in self.weights:
            # couldn't find a good way to sum all the columns up except the first
            # so I guess we're doing this explicitly...
            for i in range(np.shape(layer)[0]):
                if len(np.shape(layer)) == 1:
                    reg += layer[i]**2
                elif len(np.shape(layer)) == 2:
                    for j in range(1, np.shape(layer)[1]):
                        reg += layer[i][j]**2
                else:
                    print("ISSUE: 3D weight matrix???")
                    return nan
        # S = (lambda/ 2n) * S
        reg = (self.lambda_ / (2 * len(preds))) * reg
            
        return accum + reg
    
    # - for debugging purposes, allows you to manually set the weights of the NN
    # by passing in a list of np.array's 
    def set_weights(self, weights_in: list()) -> bool:
        # first make sure the dimensions line up
        for i in range(min(len(self.weights), len(weights_in))):
            if np.shape(self.weights[i]) != np.shape(weights_in[i]):
                print(f"Error! Dimensions of the new weights do not match that of the existing NN!\n{np.shape(self.weights[i])=}, {np.shape(weights_in[i])=}")
                return False
        # then perform the assignment
        self.weights = deepcopy(weights_in)
        return True

    def print_layers(self, dims=False):
        for layer in self.weights:
            if dims == True:
                print(np.shape(layer))
            print(layer)

def main():
    pass


if __name__ == "__main__":
    main()