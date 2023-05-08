# in an effort to keep this project more manageable, I'll reorganize and throw
# some of the utility functions in here...
import csv
import random
import os
import numpy as np


def print_np_tabbed(array_in: np.array, num_tabs: int) -> None:
    for row in array_in:
        print('\t' * num_tabs, row)
# going to have to make some changes so the data is compatible with 
# my NN model, but nothing crazy

def k_folds_gen(k: int, file_name: str) -> None:
    # find class proportiions in data set
    # make k folds
    # populate each fold according to class proportions (randomly)
    # for each fold i...
        # pass training data in to train random forest
        # evaluate on fold i
    with open(os.path.join(os.path.dirname(__file__), os.pardir, os.path.join('data',file_name)), encoding="utf-8") as raw_data_file: 
    #with open(file_name, encoding="utf-8") as raw_data_file:
        # the data files all follow different labeling conventions and/or use different delimiters...
        # could make this more general, but here we'll more or less hardcode in the correct procedure for
        # the cancer, house votes, and wine datasets
        if 'hw3_cancer.csv' in file_name:
            data_reader = csv.reader(raw_data_file, delimiter='\t')
            data_set = list(data_reader)
            for i in range(1, len(data_set)):
                for j in range(len(data_set[i])):
                    data_set[i][j] = int(float(data_set[i][j]))
            for col in range(1, len(data_set[0])): # for all attributes
                tmp_max = float('-inf')
                tmp_min = float('inf')
                for i in range(1, len(data_set)):
                    tmp_max = max(tmp_max, data_set[i][col])
                    tmp_min = min(tmp_min, data_set[i][col])
                for j in range(1, len(data_set)):
                    data_set[j][col] = (data_set[j][col] - tmp_min) / (tmp_max - tmp_min)
        elif 'hw3_house_votes_84.csv' in file_name:
            data_reader = csv.reader(raw_data_file)
            data_set = list(data_reader)
            for i in range(1, len(data_set)):
                for j in range(len(data_set[i])):
                    data_set[i][j] = int(data_set[i][j])
        elif 'hw3_wine.csv' in file_name:
            data_reader = csv.reader(raw_data_file, delimiter='\t')
            data_set = list(data_reader)
            for i in range(1, len(data_set)):
                data_set[i][0] = int(data_set[i][0])
                for j in range(1, len(data_set[i])):
                    data_set[i][j] = float(data_set[i][j])
            # normalize!!!
            for col in range(1, len(data_set[0])): # for all attributes
                tmp_max = float('-inf')
                tmp_min = float('inf')
                for i in range(1, len(data_set)):
                    tmp_max = max(tmp_max, data_set[i][col])
                    tmp_min = min(tmp_min, data_set[i][col])
                for j in range(1, len(data_set)):
                    data_set[j][col] = (data_set[j][col] - tmp_min) / (tmp_max - tmp_min)
            # for the sake of simplicity, at this point I'm going to move
            # the wine classes to the last column so it matches the other data sets
            for entry in data_set:
                tmp = entry.pop(0)
                entry.append(tmp)
        elif 'cmc.csv' in file_name:
            data_reader = csv.reader(raw_data_file, delimiter=',')
            data_set = list(data_reader)
            for i in range(len(data_set)):
                for j in range(len(data_set[i])):
                    data_set[i][j] = int(data_set[i][j])
            #for col in range(len(data_set[0]) - 1): # for all attributes
            #    tmp_max = float('-inf')
            #    tmp_min = float('inf')
            #    for i in range(len(data_set)):
            #        tmp_max = max(tmp_max, data_set[i][col])
            #        tmp_min = min(tmp_min, data_set[i][col])
            #    for j in range(len(data_set)):
            #        data_set[j][col] = (data_set[j][col] - tmp_min) / (tmp_max - tmp_min)
            # normalize!!!
            col = 0 # need to normalize the age attribute...
            tmp_max = float('-inf')
            tmp_min = float('inf')
            for i in range(len(data_set)):
                tmp_max = max(tmp_max, data_set[i][col])
                tmp_min = min(tmp_min, data_set[i][col])
            for j in range(len(data_set)):
                data_set[j][col] = (data_set[j][col] - tmp_min) / (tmp_max - tmp_min)
            col = 3 # ...and the number of children attribute
            tmp_max = float('-inf')
            tmp_min = float('inf')
            for i in range(len(data_set)):
                tmp_max = max(tmp_max, data_set[i][col])
                tmp_min = min(tmp_min, data_set[i][col])
            for j in range(len(data_set)):
                data_set[j][col] = (data_set[j][col] - tmp_min) / (tmp_max - tmp_min)
        else:
            print(f"Bad file name passed as parameter! ({file_name})")
            return None

    class_partitioned = {}
    for i in range(1, len(data_set)):
        if data_set[i][-1] in class_partitioned:
            class_partitioned[data_set[i][-1]].append(data_set[i])
        else:
            class_partitioned[data_set[i][-1]] = list()
            class_partitioned[data_set[i][-1]].append(data_set[i])
    
    class_proportions = {}
    for item in class_partitioned:
        class_proportions[item] = len(class_partitioned[item]) / (len(data_set) - 1)

    # create list of lists to hold our k folds
    k_folds_instances = []
    k_folds_labels = []
    for _ in range(k):
        k_folds_instances.append([])
        k_folds_labels.append([])

    entries_per_fold = int((len(data_set) - 1) / k)
    while k * entries_per_fold > (len(data_set) - 1):
        entries_per_fold -= 1

    if len(class_proportions) == 2:
        for index in range(k):
            for _ in range(entries_per_fold):
                if random.uniform(0,1) <= class_proportions[0]:
                    if len(class_partitioned[0]) == 0:
                        break
                    tmp = random.randrange(len(class_partitioned[0]))
                    new_entry = class_partitioned[0].pop(tmp)
                    label = new_entry[-1]
                    new_entry = np.array([new_entry])
                    new_label = np.array([np.array([1, 0])]) if label == 0 else np.array([np.array([0, 1])])
                    new_entry = np.delete(new_entry, -1) # remove the class label 
                    new_entry = np.array([new_entry])
                    #print(f"{new_label=}, {new_entry=}")
                    #print(f"{np.shape(new_label)=}, {np.shape(new_entry)=}")
                    k_folds_instances[index].append(new_entry)
                    k_folds_labels[index].append(new_label)
                else:
                    if len(class_partitioned[1]) == 0:
                        break
                    tmp = random.randrange(len(class_partitioned[1]))
                    new_entry = class_partitioned[1].pop(tmp)
                    label = new_entry[-1]
                    new_entry = np.array([new_entry])
                    new_label = np.array([np.array([1, 0])]) if label == 0 else np.array([np.array([0, 1])])
                    new_entry = np.delete(new_entry, -1) # remove the class label 
                    new_entry = np.array([new_entry])
                    k_folds_instances[index].append(new_entry)
                    k_folds_labels[index].append(new_label)
    elif len(class_proportions) == 3:
        for index in range(k):
            for _ in range(entries_per_fold):
                u = random.uniform(0,1)
                if u <= class_proportions[1]:
                    if len(class_partitioned[1]) == 0:
                        break
                    tmp = random.randrange(len(class_partitioned[1]))
                    new_entry = class_partitioned[1].pop(tmp)
                    label = new_entry[-1]
                    new_entry = np.array([new_entry])
                    new_label = np.array([0, 0, 0])
                    new_label[label - 1] = 1
                    new_label = np.array([new_label])
                    new_entry = np.delete(new_entry, -1) # remove the class label 
                    new_entry = np.array([new_entry])
                    k_folds_instances[index].append(new_entry)
                    k_folds_labels[index].append(new_label)
                elif (u > class_proportions[1]) and (u <= (class_proportions[1] + class_proportions[2])):
                    if len(class_partitioned[2]) == 0:
                        break
                    tmp = random.randrange(len(class_partitioned[2]))
                    new_entry = class_partitioned[2].pop(tmp)
                    label = new_entry[-1]
                    new_entry = np.array([new_entry])
                    new_label = np.array([0, 0, 0])
                    new_label[label - 1] = 1
                    new_label = np.array([new_label])
                    new_entry = np.delete(new_entry, -1) # remove the class label 
                    new_entry = np.array([new_entry])
                    k_folds_instances[index].append(new_entry)
                    k_folds_labels[index].append(new_label)
                else:
                    if len(class_partitioned[3]) == 0:
                        break
                    tmp = random.randrange(len(class_partitioned[3]))
                    new_entry = class_partitioned[3].pop(tmp)
                    label = new_entry[-1]
                    new_entry = np.array([new_entry])
                    new_label = np.array([0, 0, 0])
                    new_label[label - 1] = 1
                    new_label = np.array([new_label])
                    new_entry = np.delete(new_entry, -1) # remove the class label 
                    new_entry = np.array([new_entry])
                    k_folds_instances[index].append(new_entry)
                    k_folds_labels[index].append(new_label)
    else:
        print("ERROR!!!!!!!")

    return k_folds_instances, k_folds_labels

    # populate the folds according to the original data set's class proportions
    # ok to do this in a randomized fashion?