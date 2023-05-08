import csv
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

def test_congress(num_folds: int, hidden_layers: list, num_iterations: int) -> None:
    k_folds_instances, k_folds_labels = k_folds_gen(num_folds, "hw3_house_votes_84.csv")

    accuracies = []
    precisions = []
    recalls = []
    F1s = []
    for k in range(num_folds):
        test_nn = neural_net(16, 2, hidden_layers, lambda_=0.0001, alpha=0.1)
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        data = []
        labels = []
        test_fold = k_folds_instances[k]
        test_labels = k_folds_labels[k]

        for index in range(num_folds):
            if index != k:
                data += k_folds_instances[index]
                labels += k_folds_labels[index]


        # Trying out mini-batch updates here...
        data_chunks = [data[x:x+20] for x in range(0, len(data), 20)] # https://stackoverflow.com/questions/9671224/split-a-python-list-into-other-sublists-i-e-smaller-lists
        label_chunks = [labels[x:x+20] for x in range(0, len(labels), 20)]

        for index in range(len(data_chunks)):
            for _ in range(num_iterations):
                test_nn.backward_propagation(data_chunks[index], label_chunks[index])

        tested = False
        for index in range(len(test_fold)):
            tested = True
            pred = test_nn.forward_propagation(test_fold[index])
            pred = np.argmax(pred)
            label = np.argmax(test_labels[index][0])
            if label == 0: # negative class instance
                if pred == 0:
                    TN += 1
                elif pred == 1:
                    FP +=1 
                else:
                    print("uh oh")
            elif label == 1: # positive class instance
                if pred == 0:
                    FN += 1
                elif pred == 1:
                    TP += 1
                else:
                    print("uh ohh")
            else:
                print("uh ohhh")
        if tested == True: # only run these numbers of the fold was non-empty
            if TP == 0 and TN == 0 and FP == 0 and FN == 0:
                print(f"ERROR! {TP=} and {TN=} and {FP=} and {FN=}")
                return
            accuracies.append((TP + TN) / (TP + TN + FP + FN))
            if TP == 0 and FP == 0:
                print(f"ERROR! {TP=} and {FP=}")
                return
            precisions.append(TP / (TP + FP))
            if TP == 0 and FN == 0:
                print(f"ERROR! {TP=} and {FN=}")
                return
            recalls.append(TP / (TP + FN))
            F1s.append((2.0 * precisions[-1] * recalls[-1]) / (precisions[-1] + recalls[-1]))

    print(f"Congressional Dataset Results ({num_folds} folds, {hidden_layers=}, {num_iterations} backpropagations):")
    print(f"\tAvg Accuracy: {sum(accuracies) / len(accuracies)}")
    print(f"\tAvg Precision: {sum(precisions) / len(precisions)}")
    print(f"\tAvg Recall: {sum(recalls) / len(recalls)}")
    print(f"\tAvg F1 Score: {sum(F1s) / len(F1s)}")

def test_wine(num_folds: int, hidden_layers: list, num_iterations: int) -> None:
    k_folds_instances, k_folds_labels = k_folds_gen(num_folds, "hw3_wine.csv")

    accuracies = []
    precisions = []
    recalls = []
    F1s = []
    for k in range(num_folds):
        test_nn = neural_net(13, 3, hidden_layers, lambda_=0.0001, alpha=0.1)
        data = []
        labels = []
        test_fold = k_folds_instances[k]
        test_labels = k_folds_labels[k]

        for index in range(num_folds):
            if index != k:
                data += k_folds_instances[index]
                labels += k_folds_labels[index]

        # Trying out mini-batch updates here...
        data_chunks = [data[x:x+20] for x in range(0, len(data), 20)] # https://stackoverflow.com/questions/9671224/split-a-python-list-into-other-sublists-i-e-smaller-lists
        label_chunks = [labels[x:x+20] for x in range(0, len(labels), 20)]

        for index in range(len(data_chunks)):
            for _ in range(num_iterations):
                test_nn.backward_propagation(data_chunks[index], label_chunks[index])

        accuracy1 = 0
        accuracy2 = 0
        accuracy3 = 0
        precision1 = 0
        precision2 = 0
        precision3 = 0
        recall1 = 0
        recall2 = 0
        recall3 = 0
        F1_1 = 0
        F1_2 = 0
        F1_3 = 0

        TP_1 = 0
        TP_2 = 0
        TP_3 = 0
        TN_1 = 0
        TN_2 = 0
        TN_3 = 0
        FP_1 = 0
        FP_2 = 0
        FP_3 = 0
        FN_1 = 0
        FN_2 = 0
        FN_3 = 0
        tested = False
        for index in range(len(test_fold)):
            tested = True
            pred = test_nn.forward_propagation(test_fold[index])
            pred = np.argmax(pred) + 1
            label = np.argmax(test_labels[index][0]) + 1
            if label == 1: # first class
                if pred == 1:
                    TP_1 += 1
                    TN_2 += 1
                    TN_3 += 1
                elif pred == 2:
                    FN_1 += 1
                    FP_2 += 1
                    TN_3 += 1
                elif pred == 3:
                    FN_1 += 1
                    TN_2 += 1
                    FP_3 += 1
                else:
                    print("uh oh")
            elif label == 2: # second class
                if pred == 1:
                    FP_1 += 1
                    FN_2 += 1
                    TN_3 += 1
                elif pred == 2:
                    TN_1 += 1
                    TP_2 += 1
                    TN_3 += 1
                elif pred == 3:
                    TN_1 += 1
                    FN_2 += 1
                    FP_3 += 1
                else:
                    print("uh ohh")
            elif label == 3: # third class
                if pred == 1:
                    FP_1 += 1
                    TN_2 += 1
                    FN_3 += 1
                elif pred == 2:
                    TN_1 += 1
                    FP_2 += 1
                    FN_3 += 1
                elif pred == 3:
                    TN_1 += 1
                    TN_2 += 1
                    TP_3 += 1
                else:
                    print("uh ohhh")
        if tested == True: # only run these numbers of the fold was non-empty
            if TP_1 == 0 and TN_1 == 0 and FP_1 == 0 and FN_1 == 0:
                print(f"ERROR: {TP_1=} and {TN_1=} and {FP_1=} and {FN_1=}")
                return
            accuracy1 = (TP_1 + TN_1) / (TP_1 + TN_1 + FP_1 + FN_1)
            if TP_2 == 0 and TN_2 == 0 and FP_2 == 0 and FN_2 == 0:
                print(f"ERROR: {TP_2=} and {TN_2=} and {FP_2=} and {FN_2=}")
                return
            accuracy2 = (TP_2 + TN_2) / (TP_2 + TN_2 + FP_2 + FN_2)
            if TP_3 == 0 and TN_3 == 0 and FP_3 == 0 and FN_3 == 0:
                print(f"ERROR: {TP_3=} and {TN_3=} and {FP_3=} and {FN_3=}")
                return
            accuracy3 = (TP_3 + TN_3) / (TP_3 + TN_3 + FP_3 + FN_3)
            if TP_1 == 0 and FP_1 == 0:
                print(f"ERROR: {TP_1=} and {FP_1=}")
                return
            precision1 = (TP_1) / (TP_1 + FP_1)
            if TP_2 == 0 and FP_2 == 0:
                print(f"ERROR: {TP_2=} and {FP_2=}")
                return
            precision2 = (TP_2) / (TP_2 + FP_2)
            if TP_3 == 0 and FP_3 == 0:
                print(f"ERROR: {TP_3=} and {FP_3=}")
                return
            precision3 = (TP_3) / (TP_3 + FP_3)
            if TP_1 == 0 and FN_1 == 0:
                print(f"ERROR: {TP_1=} and {FN_1=}")
                return
            recall1 = (TP_1) / (TP_1 + FN_1)
            if TP_2 == 0 and FN_2 == 0:
                print(f"ERROR: {TP_2=} and {FN_2=}")
                return
            recall2 = (TP_2) / (TP_2 + FN_2)
            if TP_3 == 0 and FN_3 == 0:
                print(f"ERROR: {TP_3=} and {FN_3=}")
                return
            recall3 = (TP_3) / (TP_3 + FN_3)
            if precision1 == 0 and recall1 == 0:
                print(f"ERROR: {precision1=} and {recall1=}")
                return
            F1_1 = (2.0 * precision1 * recall1) / (precision1 + recall1)
            if precision2 == 0 and recall2 == 0:
                print(f"ERROR: {precision2=} and {recall2=}")
                return
            F1_2 = (2.0 * precision2 * recall2) / (precision2 + recall2)
            if precision3 == 0 and recall3 == 0:
                print(f"ERROR: {precision3=} and {recall3=}")
                return
            F1_3 = (2.0 * precision3 * recall3) / (precision3 + recall3)
            accuracies.append((accuracy1 + accuracy2 + accuracy3) / 3.0)
            precisions.append((precision1 + precision2 + precision3) / 3.0)
            recalls.append((recall1 + recall2 + recall3) / 3.0)
            F1s.append((F1_1 + F1_2 + F1_3) / 3.0)
    print(f"Wine Dataset Results ({num_folds} folds, {hidden_layers=}, {num_iterations} backpropagations):")
    print(f"\tAvg Accuracy: {sum(accuracies) / len(accuracies)}")
    print(f"\tAvg Precision: {sum(precisions) / len(precisions)}")
    print(f"\tAvg Recall: {sum(recalls) / len(recalls)}")
    print(f"\tAvg F1 Score: {sum(F1s) / len(F1s)}")

def test_cancer(num_folds: int, hidden_layers: list, num_iterations: int) -> None:
    k_folds_instances, k_folds_labels = k_folds_gen(num_folds, "hw3_cancer.csv")
    
    accuracies = []
    precisions = []
    recalls = []
    F1s = []
    for k in range(num_folds):
        test_nn = neural_net(9, 2, hidden_layers, lambda_=0.0001, alpha=0.1)
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        data = []
        labels = []
        test_fold = k_folds_instances[k]
        test_labels = k_folds_labels[k]

        for index in range(num_folds):
            if index != k:
                data += k_folds_instances[index]
                labels += k_folds_labels[index]

        # Trying out mini-batch updates here...
        data_chunks = [data[x:x+50] for x in range(0, len(data), 50)] # https://stackoverflow.com/questions/9671224/split-a-python-list-into-other-sublists-i-e-smaller-lists
        label_chunks = [labels[x:x+50] for x in range(0, len(labels), 50)]

        for index in range(len(data_chunks)):
            for _ in range(num_iterations):
                test_nn.backward_propagation(data_chunks[index], label_chunks[index])

        tested = False
        for index in range(len(test_fold)):
            tested = True
            pred = test_nn.forward_propagation(test_fold[index])
            pred = np.argmax(pred)
            label = np.argmax(test_labels[index][0])
            if label == 0: # negative class instance
                if pred == 0:
                    TN += 1
                elif pred == 1:
                    FP +=1 
                else:
                    print("uh oh")
            elif label == 1: # positive class instance
                if pred == 0:
                    FN += 1
                elif pred == 1:
                    TP += 1
                else:
                    print("uh ohh")
            else:
                print("uh ohhh")
        if tested == True: # only run these numbers of the fold was non-empty
            if TP == 0 and TN == 0 and FP == 0 and FN == 0:
                print(f"ERROR! {TP=} and {TN=} and {FP=} and {FN=}")
                return
            accuracies.append((TP + TN) / (TP + TN + FP + FN))
            if TP == 0 and FP == 0:
                print(f"ERROR! {TP=} and {FP=}")
                return
            precisions.append(TP / (TP + FP))
            if TP == 0 and FN == 0:
                print(f"ERROR! {TP=} and {FN=}")
                return
            recalls.append(TP / (TP + FN))
            F1s.append((2.0 * precisions[-1] * recalls[-1]) / (precisions[-1] + recalls[-1]))

    print(f"Cancer Dataset Results ({num_folds} folds, {hidden_layers=}, {num_iterations} backpropagations):")
    print(f"\tAvg Accuracy: {sum(accuracies) / len(accuracies)}")
    print(f"\tAvg Precision: {sum(precisions) / len(precisions)}")
    print(f"\tAvg Recall: {sum(recalls) / len(recalls)}")
    print(f"\tAvg F1 Score: {sum(F1s) / len(F1s)}")

def test_contraceptive(num_folds: int, hidden_layers: list, num_iterations: int) -> None:
    k_folds_instances, k_folds_labels = k_folds_gen(num_folds, "cmc.csv")
    
    accuracies = []
    precisions = []
    recalls = []
    F1s = []
    for k in range(num_folds):
        test_nn = neural_net(9, 3, hidden_layers)
        data = []
        labels = []
        test_fold = k_folds_instances[k]
        test_labels = k_folds_labels[k]

        for index in range(num_folds):
            if index != k:
                data += k_folds_instances[index]
                labels += k_folds_labels[index]

        # Trying out mini-batch updates here...
        data_chunks = [data[x:x+100] for x in range(0, len(data), 100)] # https://stackoverflow.com/questions/9671224/split-a-python-list-into-other-sublists-i-e-smaller-lists
        label_chunks = [labels[x:x+100] for x in range(0, len(labels), 100)]
        
        for index in range(len(data_chunks)):
            for _ in range(num_iterations):
                test_nn.backward_propagation(data_chunks[index], label_chunks[index])

        accuracy1 = 0
        accuracy2 = 0
        accuracy3 = 0
        precision1 = 0
        precision2 = 0
        precision3 = 0
        recall1 = 0
        recall2 = 0
        recall3 = 0
        F1_1 = 0
        F1_2 = 0
        F1_3 = 0

        TP_1 = 0
        TP_2 = 0
        TP_3 = 0
        TN_1 = 0
        TN_2 = 0
        TN_3 = 0
        FP_1 = 0
        FP_2 = 0
        FP_3 = 0
        FN_1 = 0
        FN_2 = 0
        FN_3 = 0
        tested = False
        for index in range(len(test_fold)):
            tested = True
            pred = test_nn.forward_propagation(test_fold[index])
            pred = np.argmax(pred) + 1
            label = np.argmax(test_labels[index][0]) + 1
            if label == 1: # first class
                if pred == 1:
                    TP_1 += 1
                    TN_2 += 1
                    TN_3 += 1
                elif pred == 2:
                    FN_1 += 1
                    FP_2 += 1
                    TN_3 += 1
                elif pred == 3:
                    FN_1 += 1
                    TN_2 += 1
                    FP_3 += 1
                else:
                    print("uh oh")
            elif label == 2: # second class
                if pred == 1:
                    FP_1 += 1
                    FN_2 += 1
                    TN_3 += 1
                elif pred == 2:
                    TN_1 += 1
                    TP_2 += 1
                    TN_3 += 1
                elif pred == 3:
                    TN_1 += 1
                    FN_2 += 1
                    FP_3 += 1
                else:
                    print("uh ohh")
            elif label == 3: # third class
                if pred == 1:
                    FP_1 += 1
                    TN_2 += 1
                    FN_3 += 1
                elif pred == 2:
                    TN_1 += 1
                    FP_2 += 1
                    FN_3 += 1
                elif pred == 3:
                    TN_1 += 1
                    TN_2 += 1
                    TP_3 += 1
                else:
                    print("uh ohhh")
        if tested == True: # only run these numbers of the fold was non-empty
            if TP_1 == 0 and TN_1 == 0 and FP_1 == 0 and FN_1 == 0:
                print(f"ERROR: {TP_1=} and {TN_1=} and {FP_1=} and {FN_1=}")
                return
            accuracy1 = (TP_1 + TN_1) / (TP_1 + TN_1 + FP_1 + FN_1)
            if TP_2 == 0 and TN_2 == 0 and FP_2 == 0 and FN_2 == 0:
                print(f"ERROR: {TP_2=} and {TN_2=} and {FP_2=} and {FN_2=}")
                return
            accuracy2 = (TP_2 + TN_2) / (TP_2 + TN_2 + FP_2 + FN_2)
            if TP_3 == 0 and TN_3 == 0 and FP_3 == 0 and FN_3 == 0:
                print(f"ERROR: {TP_3=} and {TN_3=} and {FP_3=} and {FN_3=}")
                return
            accuracy3 = (TP_3 + TN_3) / (TP_3 + TN_3 + FP_3 + FN_3)
            if TP_1 == 0 and FP_1 == 0:
                print(f"ERROR: {TP_1=} and {FP_1=}")
                return
            precision1 = (TP_1) / (TP_1 + FP_1)
            if TP_2 == 0 and FP_2 == 0:
                print(f"ERROR: {TP_2=} and {FP_2=}")
                return
            precision2 = (TP_2) / (TP_2 + FP_2)
            if TP_3 == 0 and FP_3 == 0:
                print(f"ERROR: {TP_3=} and {FP_3=}")
                return
            precision3 = (TP_3) / (TP_3 + FP_3)
            if TP_1 == 0 and FN_1 == 0:
                print(f"ERROR: {TP_1=} and {FN_1=}")
                return
            recall1 = (TP_1) / (TP_1 + FN_1)
            if TP_2 == 0 and FN_2 == 0:
                print(f"ERROR: {TP_2=} and {FN_2=}")
                return
            recall2 = (TP_2) / (TP_2 + FN_2)
            if TP_3 == 0 and FN_3 == 0:
                print(f"ERROR: {TP_3=} and {FN_3=}")
                return
            recall3 = (TP_3) / (TP_3 + FN_3)
            if precision1 == 0 and recall1 == 0:
                print(f"ERROR: {precision1=} and {recall1=}")
                return
            F1_1 = (2.0 * precision1 * recall1) / (precision1 + recall1)
            if precision2 == 0 and recall2 == 0:
                print(f"ERROR: {precision2=} and {recall2=}")
                return
            F1_2 = (2.0 * precision2 * recall2) / (precision2 + recall2)
            if precision3 == 0 and recall3 == 0:
                print(f"ERROR: {precision3=} and {recall3=}")
                return
            F1_3 = (2.0 * precision3 * recall3) / (precision3 + recall3)
            accuracies.append((accuracy1 + accuracy2 + accuracy3) / 3.0)
            precisions.append((precision1 + precision2 + precision3) / 3.0)
            recalls.append((recall1 + recall2 + recall3) / 3.0)
            F1s.append((F1_1 + F1_2 + F1_3) / 3.0)
    print(f"Contraceptive Choice Dataset Results ({num_folds} folds, {hidden_layers=}, {num_iterations} backpropagations):")
    print(f"\tAvg Accuracy: {sum(accuracies) / len(accuracies)}")
    print(f"\tAvg Precision: {sum(precisions) / len(precisions)}")
    print(f"\tAvg Recall: {sum(recalls) / len(recalls)}")
    print(f"\tAvg F1 Score: {sum(F1s) / len(F1s)}")

def J_data():
    # Congressional Dataset
    #test_nn = neural_net(16, 2, [10,10,10], lambda_=0.001, alpha=0.01)
    #k_folds_instances, k_folds_labels = k_folds_gen(1, "hw3_house_votes_84.csv")\
    # Wine Dataset
    #test_nn = neural_net(13, 3, [10,10,10], lambda_=0.001, alpha=0.01)
    #k_folds_instances, k_folds_labels = k_folds_gen(1, "hw3_wine.csv")
    # Cancer Dataset
    test_nn = neural_net(9, 2, [10,10], lambda_=0.001, alpha=0.01)
    k_folds_instances, k_folds_labels = k_folds_gen(1, "hw3_cancer.csv")

    k_folds_instances = k_folds_instances[0]
    k_folds_labels = k_folds_labels[0]

    # create test set
    test = []
    test_labels = []
    num_instances = len(k_folds_instances)
    for _ in range(int(0.3 * num_instances)):
        test.append(k_folds_instances.pop(-1))
        test_labels.append(k_folds_labels.pop(-1))

    with open(f"results.csv", 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for a in range(25):
            print(a)
            for index_i in range(len(k_folds_instances)):
                test_nn.backward_propagation(k_folds_instances[index_i], k_folds_labels[index_i])
                if (index_i + 1) % 5 == 0:
                    preds = list()
                    for index in range(len(test)):
                        preds.append(np.array([test_nn.forward_propagation(test[index])]))
                    csv_writer.writerow([(index_i + 1) + (a * len(k_folds_instances)), test_nn.reg_cost_func(preds, test_labels)])
                    #print(f"{(index_i + 1) * (a + 1)}, {test_nn.reg_cost_func(preds, test_labels)}")
def main() 
    #test_examples()
    #test_congress(5, list([10, 10, 10, 10, 10]), 200)
    #test_wine(5, list([10,10,10,10]), 200)
    #test_cancer(5, list([20, 20, 20]), 200)
    #test_contraceptive(5, list([10,10,10]), 200)
    J_data()               

if __name__ == "__main__":
    main()