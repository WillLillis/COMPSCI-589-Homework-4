from copy import deepcopy
import numpy as np
from misc import print_np_tabbed

class neural_net:
    # - input_nodes : specifies the number of input nodes
    # - hidden_layers : a list, where each entry is an integer specifying the number
    # of nodes in that layer
    # - lambda_ : regularization term
    # - alpha : learning rate
    # - activ_func : which activation function to use
    def __init__(self, input_nodes: int, output_nodes: int, hidden_layers: list, lambda_=0.0, alpha=1.0, activ_func="sigmoid"):
        # sanity checks 
        if input_nodes <= 0:
            print(f"Invalid parameter: {input_nodes=}")
            raise("Inavlid parameter")
        if output_nodes <= 0:
            print(f"Invalid parameter: {output_nodes=}")
            raise("Inavlid parameter")
        for i in range(len(hidden_layers)):
            if hidden_layers[i] <= 0:
                print(f"Invalid parameter: {hidden_layers[i]=}")
                raise("Inavlid parameter")

        self.activation_function = activ_func
        self.lambda_ = lambda_
        self.alpha = alpha

        # now that we've done some parameter checks...
        self.weights = list() # list to hold the numpy arrays (make this a numpy array too?)
        # - first create the matrix to hold the weights between 
        # the input layer and first input layer
        weight_tmp = np.array(np.random.normal(0, 1, input_nodes + 1)) # + 1 for bias weights
        for _ in range(hidden_layers[0] - 1):
            weight_tmp = np.vstack((weight_tmp, np.random.normal(0, 1, input_nodes + 1))) # + 1 for bias weights
        if len(np.shape(weight_tmp)) == 1:
            self.weights.append(np.array([weight_tmp]))
        elif len(np.shape(weight_tmp)) == 2:
            self.weights.append(weight_tmp)
        else:
            print("ERROR: INVALID DIMENSIONS!!!")

        # now for the hidden layers connecting to one another
        for i in range(len(hidden_layers) - 1):
             weight_tmp = np.array(np.random.normal(0, 1, hidden_layers[i] + 1)) # + 1 for bias weights
             for _ in range(hidden_layers[i + 1] - 1):
                weight_tmp = np.vstack((weight_tmp, np.random.normal(0, 1, hidden_layers[i] + 1))) # + 1 for bias weights
             if len(np.shape(weight_tmp)) == 1:
                self.weights.append(np.array([weight_tmp]))
             elif len(np.shape(weight_tmp)) == 2:
                self.weights.append(weight_tmp)
             else:
                print("ERROR: INVALID DIMENSIONS!!!")

        # and finally the last hidden layer connecting to the output layer
        weight_tmp = np.array(np.random.normal(0, 1, hidden_layers[-1] + 1)) # + 1 for bias weights
        for _ in range(output_nodes - 1):
            weight_tmp = np.vstack((weight_tmp, np.random.normal(0, 1, hidden_layers[-1] + 1))) # + 1 for bias weights
        if len(np.shape(weight_tmp)) == 1:
            self.weights.append(np.array([weight_tmp]))
        elif len(np.shape(weight_tmp)) == 2:
            self.weights.append(weight_tmp)
        else:
            print("ERROR: INVALID DIMENSIONS!!!")

    # - propagate instance through the network, get np array of output nodes returned
    def forward_propagation(self, instance: np.array, test=False, test_indent=1) -> np.array:
        if test == True:
            print('\t' * test_indent, f"Forward propagating the input {instance}")

        activation = np.insert(instance, 0, 1) # add the bias term to the input
        if test == True:
            print('\t' * test_indent, f"a_1: {activation}")

        # for hidden layers 0...n-1
        for i in range(len(self.weights) - 1):
            activation = np.matmul(self.weights[i], activation) # multiply vector with hidden layer weights
            if test == True:
                print('\t' * test_indent, f"z_{i + 2}: {activation}") # the example files use 1 based indexing, hence the + 2....
            activation = self.activation_func(activation) # apply activation function g element-wise
            activation = np.insert(activation, 0, 1) # add bias term back in
            if test == True:
                print('\t' * test_indent, f"a_{i + 2}: {activation}")
        
        # apply last hidden layer, get output values
        activation = np.matmul(self.weights[-1], activation) 
        if test == True:
            print('\t' * test_indent, f"z_{len(self.weights) + 1}: {activation}")
        activation = self.activation_func(activation)
        if test == True:
            print('\t' * test_indent, f"a_{len(self.weights) + 1}: {activation}")
        if test == True:
            print('\t' * test_indent, f"f(x): {activation}")
        return activation

    # same as forward_propagation but also returns the activations for each hidden layer
    # going to keep the two separate for performance reasons
    def forward_propagation_ret_act(self, instance: np.array) -> np.array:
        ret_activ = list()
        activation = np.insert(instance, 0, 1) # add the bias term to the input
        ret_activ.append(np.array([activation]).T) # Transpose to match expected shape in backprop alg

        # for hidden layers 0...n-1
        for i in range(len(self.weights) - 1):
            activation = np.matmul(self.weights[i], activation) # multiply vector with hidden layer weights
            activation = self.activation_func(activation) # apply activation function g element-wise
            activation = np.insert(activation, 0, 1) # add bias term back in
            ret_activ.append(np.array([activation]).T) # Transpose to match expected shape in backprop alg
        
        # apply last hidden layer, get output values
        activation = np.matmul(self.weights[-1], activation) 
        activation = self.activation_func(activation)
        ret_activ.append(np.array([activation]).T) # Transpose to match expected shape in backprop alg
        return np.array([activation]), ret_activ

    def backward_propagation(self, instances: list, labels: list, test=False, test_indent=1) -> None:
        # check to make sure instances and labels are the same length!
        if len(instances) != len(labels):
            print(f"Error: Mismatching dimensions betweent the prediction and labels arrays! {len(instances)=}, {len(labels)=}")
        # initialize D to all 0's, same dimension as the weights
        grads = deepcopy(self.weights) # just copying to get the dimensions right

        for index in range(len(grads)):
            grads[index][:] = 0 # zero out all entries in the matrix

        regularizers = deepcopy(grads) # same dimensions as weights, also needs to be initialized to zero

        for index in range(len(instances)):
            if test == True:
                print('\t' * test_indent, f"Computing gradients based on training instance {index + 1}")
            preds, activations = self.forward_propagation_ret_act(instances[index])
            deltas = list()
            # Need lists with full length because we're indexing backwards
                # - going to initialize with None so it'll be easier to tell 
                # when we don't populate an index and then try to use it
            for _ in range(len(self.weights)):
                deltas.append(None)
                regularizers.append(None)

            deltas[-1] = preds - labels[index]
            if len(np.shape(deltas[-1])) == 2: # more dimensionality fixes for numpy nonsense
                deltas[-1] = deltas[-1].T
            
            if test == True:
                #print('\t' * test_indent, f"\tdelta{len(deltas) + 1}: {deltas[-1]}")
                print('\t' * (test_indent + 1), f"delta{len(deltas) + 1}:")
                print_np_tabbed(deltas[-1], test_indent + 1)

            # "For each network layer, k = L - 1...2"
            for k in range(len(self.weights) - 1, 0, -1): # confusing indices...
                tmp = np.matmul(self.weights[k].T, deltas[k])
                tmp *= activations[k]
                tmp *= (1 - activations[k])
                tmp = np.delete(tmp, 0)
                if len(np.shape(tmp)) == 1:
                    deltas[k - 1] = np.array([tmp]).T
                elif len(np.shape(tmp)) == 2:
                    deltas[k - 1] = tmp
                else:
                    print("Lol time to be sad")
                    raise("Dimension error!")
                if test == True:
                    print('\t' * (test_indent + 1), f"delta{k+1}:")
                    print_np_tabbed(deltas[k - 1], test_indent + 1)
                
            # "For each network layer, k = L - 1...1"
            for k in range(len(self.weights) - 1, -1, -1): # confusing indices...
                grads[k] += np.matmul(deltas[k], activations[k].T) # accumulates, in D(l=k), the gradients computed based on the current training instance
                if test == True:
                    print('\t' * (test_indent + 1), f"Gradients of Theta{k + 1} based on training instance {index + 1}:")
                    print_np_tabbed(np.matmul(deltas[k], activations[k].T), test_indent + 1)
            
        # "For each network layer, k= L - 1...1"
        if test == True:
            print('\t' * test_indent, "The entire training set has been processed. Computing the average (regularized) gradients:")
        for k in range(len(self.weights) - 1, -1, -1):
            regularizers[k] = self.lambda_ * self.weights[k]
            regularizers[k][:, 0] = 0 # Zero out first column
            grads[k] = (1 / len(instances)) * (grads[k] + regularizers[k])
            if(test==True):
                print('\t' * test_indent, f"Final regularized gradients of Theta{k+1}:")
                print_np_tabbed(grads[k], test_indent)
        # "At this point, D^(l=1) contains the gradients of the weights θ(l=1); (…); and D(l=L-1) contains the gradients of the weights θ(l=L-1)"
        # "For each network layer, k = L - 1...1"
        for k in range(len(self.weights) - 1, -1, -1): # confusing indices...
            self.weights[k] -= self.alpha * grads[k]

    def activation_func(self, input_arr: np.array) -> np.array:
        if self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-input_arr))
        elif self.activation_function == "ReLU":
            return [max(0, x) for x in input_arr]
        else:
            print(f"Invalid activation function parameter passed! {self.activation_function=}")
            raise("Invalid parameter")
    
    # computes cost for INDIVIDUAL prediction and expected output instances
    @staticmethod
    def indiv_cost_func(pred: np.array, label: np.array) -> float:
        # check dimensions
        if np.shape(pred) != np.shape(label):
            print(f"Error: Mismatching dimensions betweent the prediction and labels arrays! {np.shape(pred)}, {np.shape(label)}")
            raise("Dimension error!")
        accum = 0

        # TODO: take this out????
        if len(np.shape(pred)) == 2:
            pred = pred[0]
            label = label[0]

        for i in range(len(pred)):
            #print(f"\t{pred[i]=}, {label[i]=}")
            accum += (-label[i] * (np.log(pred[i]))) - (1 - label[i]) * (np.log(1 - pred[i]))
        return accum


    # preds and labels are lists containing np.array's
    def reg_cost_func(self, preds: list, labels: list) -> float:
        # check lengths of lists
        if len(preds) != len(labels):
            print(f"ERROR! Mismatching lengths for prediction and label lists. ({len(preds)=}, {len(labels)=})")
            return np.nan
        elif len(preds) == 0:
            return 0

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
                for j in range(1, np.shape(layer)[1]):
                        reg += layer[i][j]**2
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

    def print_layers(self, info=False, num_tabs=1)  -> None:
        for i in range(len(self.weights)):
            if info == True:
                print(f"Layer {i+1}: {np.shape(self.weights[i])}")
            #print(f"{self.weights[i]}\n")
            print_np_tabbed(self.weights[i], num_tabs)

def main():
    pass


if __name__ == "__main__":
    main()