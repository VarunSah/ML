'''
Created on Sep 26, 2016

@author: varun
'''

import sys
import random
import math
import numpy
import scipy
import pylab
import matplotlib
from scipy.io import arff
from matplotlib import pyplot

def incorrectUsageMessage():
    print "Incorrect usage. Correct format : python dt-learn <train file path> <test file path> <leaf threshold>"


def get_entropy(data, class_label):    
    entropy = 0.0
    unique_counts_map = {}
    
    for row in data:
        if unique_counts_map.has_key(row[class_label]):
            unique_counts_map[row[class_label]] += 1.0
        else:
            unique_counts_map[row[class_label]] = 1.0
    
    probabilities = (numpy.array(unique_counts_map.values())).astype('float')/len(data)
    
    for p in probabilities:
        if p != 0.0:
            entropy += (-p) * numpy.log2(p)           
    return entropy 


def get_nominal_information_gain(data, class_label, split_feature):
    split_entropy=0.0
    unique_counts_map = {}
    
    for row in data:
        if (unique_counts_map.has_key(row[split_feature])):
            unique_counts_map[row[split_feature]] += 1.0
        else:
            unique_counts_map[row[split_feature]] = 1.0

    probabilities = (numpy.array(unique_counts_map.values())).astype('float')/float(sum(unique_counts_map.values()))    
    probability_key_map = zip(probabilities, unique_counts_map.keys())

    for p, key in probability_key_map:
        sub_data = [row for row in data if row[split_feature] == key]
        if p != 0.0:
            split_entropy += p * get_entropy(sub_data, class_label)
    
    information_gain = get_entropy(data, class_label) - split_entropy
    return(information_gain, None)


def get_numeric_information_gain(data, class_label, feature):
    split_entropy = 0.0
    candidate_thresholds = get_candidate_thresholds(data, class_label, feature)
    candidate_information_gain = []
    
    for threshold in candidate_thresholds:
        split_entropy = 0.0
        less_or_equal_data = [x for x in data if x[feature] <= threshold]
        greater_data = [x for x in data if x[feature] > threshold]
               
        le_proportion = float(len(less_or_equal_data))/float(len(data))
        g_proportion = float(len(greater_data))/float(len(data))
        split_entropy += le_proportion * get_entropy(less_or_equal_data, class_label)
        split_entropy += g_proportion * get_entropy(greater_data, class_label)
        
        candidate_information_gain.append(split_entropy)
        
    if (len(candidate_thresholds) != 0):
        initial_entropy = get_entropy(data, class_label)
        candidate_information_gain_list = [(initial_entropy - entropy) for entropy in candidate_information_gain]
        chosen_information_gain = max(candidate_information_gain_list)    
        chosen_threshold = candidate_thresholds[candidate_information_gain_list.index(chosen_information_gain)]
    else:
        chosen_information_gain = None
        chosen_threshold = None
     
    return (chosen_information_gain, chosen_threshold)


def get_candidate_thresholds(data, class_label, split_feature):
    candidate_thresholds = []
    
    split_feature_unique_values = get_set_of_values(data, split_feature)
    split_feature_unique_values.sort()

    feature_value_class_purity = dict((value,-1) for value in split_feature_unique_values)
    
    for row in data:
        current_value = row[split_feature]
        current_label = row[class_label]

        if(feature_value_class_purity[current_value] == -1):
            if(current_label == unique_class_values[0]):
                feature_value_class_purity[current_value] = 0
            elif (current_label == unique_class_values[1]):
                feature_value_class_purity[current_value] = 1
        elif(feature_value_class_purity[current_value] == 0):
            if(current_label == unique_class_values[1]):
                feature_value_class_purity[current_value] = 2
        elif(feature_value_class_purity[current_value] == 0):
            if(current_label == unique_class_values[1]):
                feature_value_class_purity[current_value] = 2
        
    for index in range(0,len(split_feature_unique_values)-1):
        current_value = split_feature_unique_values[index]
        next_value = split_feature_unique_values[index+1]
        
        current_purity = feature_value_class_purity[current_value]
        next_purity = feature_value_class_purity[next_value]
 
        if((current_purity == 2) or (next_purity == 2)):
            candidate_thresholds.append(float(current_value + next_value)/2.0)
        elif (abs(current_purity - next_purity) == 1 ):
            candidate_thresholds.append(float(current_value + next_value)/2.0)
 
    return candidate_thresholds
  
    
def select_split_feature(data, class_label, features):
    max_information_gain = 0.0
    chosen_split_feature = None
    chosen_threshold = None
    
    for feature in features:
        if(feature_type_map[feature] == 'nominal'):
            current_information_gain, current_threshold = get_nominal_information_gain(data, class_label, feature)
        else:
            current_information_gain, current_threshold = get_numeric_information_gain(data, class_label, feature)
        if (feature != class_label):
            if (current_information_gain > max_information_gain):
                max_information_gain = current_information_gain
                chosen_split_feature = feature
                chosen_threshold = current_threshold
    return chosen_split_feature, chosen_threshold


def get_data_subset(data, features, current_feature, current_feature_output, is_nominal, threshold):
    
    current_feature_index = features.tolist().index(current_feature)
    data_subset = data[:]

    if(is_nominal == True):
        indices = []
        index =0
        for row in data_subset:
            if(row[current_feature_index] != current_feature_output):
                indices.append(index)
            index += 1
        data_subset = numpy.delete(data_subset, indices, 0)
        remaining_features = [x for x in data_subset.dtype.names if x not in [current_feature]]
        data_subset = data_subset[remaining_features]                
    else:
        indices = []
        index =0        
        for row in data:
            #only include rows which are <= threshold or > threshold as appropriate
            if(((current_feature_output == "<=") and (float(row[current_feature_index]) <= threshold)) or ((current_feature_output == ">") and (float(row[current_feature_index]) > threshold))):
                index += 1
                continue
            else:
                indices.append(index)
                index +=1
        data_subset = numpy.delete(data_subset, indices, 0)  
    return data_subset


def build_decision_tree(data, class_label, features, m,recursion_level, default_label)  :  
    #Global variable to maintain output across recursive calls
    global output_string
    recursion_level += 1

    data = data[:]
    class_label_values = []
    
    #In given data, get all class label values
    for row in data:
        class_label_values.append(row[-1])
    
    #Given that we can assume that the class attribute is binary
   
    first_class_value_count = class_label_values.count(unique_class_range[0])
    second_class_value_count = len(class_label_values) - first_class_value_count
    
    class_distribution_string = " ["+str(first_class_value_count)+" "+str(second_class_value_count)+"]"
    
    if first_class_value_count > second_class_value_count:
        majority_class = unique_class_range[0]
    elif second_class_value_count > first_class_value_count:
        majority_class = unique_class_range[1]
    else:
        majority_class = default_label
    
    #Remove last element of output string
    output_string = output_string[:-1]
    
    if recursion_level != 0:
        output_string += class_distribution_string + "\n"
       
    if (len(data) == 0):
        output_string = output_string[:-1]
        output_string += ": "+unique_class_values[0]+"\n"
        return unique_class_values[0]
    elif len(class_label_values) == class_label_values.count(class_label_values[0]): 
         #(i) all of the training instances reaching the node belong to the same class
        output_string = output_string[:-1]
        output_string += ": "+class_label_values[0]+"\n"
        return class_label_values[0]
    elif (len(features) == 0) or (len(data) < m):
        #(ii) there are fewer than m training instances reaching the node, where m is provided as input to the program
        if(first_class_value_count > second_class_value_count):
            majority_class = unique_class_range[0]
            output_string = output_string[:-1]
            output_string += ": "+ unique_class_range[0]+"\n"
            return unique_class_range[0]
        elif(first_class_value_count < second_class_value_count):
            majority_class = unique_class_range[1]
            output_string = output_string[:-1]
            output_string += ": "+unique_class_range[1]+"\n"
            return unique_class_range[1]
        else:            
            output_string = output_string[:-1]
            output_string += ": "+  majority_class + "\n"
            return majority_class
    else:        
        split_feature,  split_threshold = select_split_feature(data, class_label,  features)

        if(split_feature is None): 
            #(iii) no feature has positive information gain
            if(first_class_value_count >= second_class_value_count):
                output_string = output_string[:-1]
                output_string += ": "+ unique_class_values[0]+"\n"
                return unique_class_values[0]
            elif(first_class_value_count < second_class_value_count):
                output_string = output_string[:-1]
                output_string += ": "+unique_class_values[1]+"\n"
                return unique_class_values[1]
        else:
            decision_tree = {split_feature:{}}           
            if split_threshold is None:
                #Split Feature is nominal 
                split_criteria_list = feature_range_map[split_feature]
                for split_criteria in split_criteria_list:
                    for tab_count in range(0,recursion_level):
                        output_string += "|\t"
                    output_string += split_feature+" = "+split_criteria+'\n'
                    data_subset = get_data_subset(data, features, split_feature, split_criteria,True, 0)
                    features_subset = features[:]
                    features_subset = numpy.delete(features_subset, features_subset.tolist().index(split_feature))
                    sub_decision_tree = build_decision_tree(data_subset, class_label, features_subset, m,recursion_level, majority_class)
                    
                    if sub_decision_tree is None:
                        output_string = output_string[:-1]
                        output_string += ": "+  majority_class + "\n"
                        sub_decision_tree = majority_class
                        
                    decision_tree[split_feature][split_criteria] = sub_decision_tree                
            else:
                #Split Feature is numeric
                split_criteria_list = ["<=", ">"]
                for split_criteria in split_criteria_list:
                    for tab_count in range(0,recursion_level):
                        output_string += "|\t"
                    output_string += split_feature+" "+split_criteria+" "+str("%.6f"%split_threshold)+'\n'
                    data_subset = get_data_subset(data, features,split_feature,split_criteria, False, split_threshold)
                    features_subset = features[:]
                    sub_decision_tree = build_decision_tree(data_subset, class_label, features_subset, m,recursion_level, majority_class)
                    
                    if sub_decision_tree is None:
                        output_string = output_string[:-1]
                        output_string += ": "+  majority_class + "\n"
                        sub_decision_tree = majority_class
                    
                    split_criteria = split_criteria+" "+str(split_threshold)
                    decision_tree[split_feature][split_criteria] = sub_decision_tree
    return decision_tree


def get_set_of_values (data, feature):
    feature_value_list = data[feature].tolist()
    unique_feature_values = sorted(set(feature_value_list), key=feature_value_list.index)      
    return unique_feature_values


def get_feature_ranges(training_data_file):
    file = open(training_data_file,"r")
    lines = file.read().split("\n")
    feature_ranges = {}
    for line in lines:
            if line.startswith("@attribute"):
                tokens =line.split(' ')
                if len(tokens) <4:
                    feature_ranges[eval(tokens[1])] = tokens[-1]   
                else:
                    feature_ranges[eval(tokens[1])] = []
                    for index in range(3,len(tokens)):
                        feature_ranges[eval(tokens[1])].append(tokens[index][:-1])
    return feature_ranges


def get_predictions(decision_tree, test_data):
    result = ""
    correct_predictions = 0
    incorrect_predictions =0 
    result += "<Predictions for the Test Set Instances>\n"
    for index in range(0,len(test_data)):
        test_row = test_data[index]
        if type(decision_tree) is dict:
            current_tree = decision_tree.copy()
            predicted_label = ""
            while (type(current_tree) is dict):
                current_feature = current_tree.keys()[0]
                current_feature_index = features.tolist().index(current_feature)
                current_feature_value = test_row[current_feature_index]
                current_tree = current_tree[current_feature]
                next_level_keys = current_tree.keys()
                if '<=' in next_level_keys[0] or '>' in next_level_keys[0]:
                    # numeric feature
                    if(eval(str(current_feature_value)+next_level_keys[0])):
                        predicted_label = current_tree[next_level_keys[0]]
                        current_tree = current_tree[next_level_keys[0]]
                    else:
                        predicted_label = current_tree[next_level_keys[1]]
                        current_tree = current_tree[next_level_keys[1]] 
                elif (str(current_feature_value) in current_tree.keys()):
                    # nominal feature
                    predicted_label = current_tree[current_feature_value]
                    current_tree = current_tree[current_feature_value]
                else:
                    print "Invalid Input"
        else:
            predicted_label = decision_tree

        result += "%d: Actual: "%(index+1)+test_row[-1]+" Predicted: "+predicted_label + "\n"
        if predicted_label == test_row[-1]:
            correct_predictions += 1
        else:
            incorrect_predictions += 1
    
    result += "Number of correctly classified: "+str(correct_predictions)+" Total number of test instances: "+str(correct_predictions+incorrect_predictions)
    return (correct_predictions,incorrect_predictions, result)
       

def plot_accuracy_vs_training_set_size(data, test_data, percentages_list, iterations, m , dataset_name): 
    dataset_name = dataset_name.split('_')[0]
    total_data_count = len(data)
    total_test_count = len(test_data)
    table = numpy.zeros([len(percentages_list),4])
    for i in xrange(len(percentages_list)):
        correct_predictions_list = []
        if percentages_list[i] == 100.0:
            decision_tree = build_decision_tree(data, class_label, features, m,-1, default_label)
            correct_predictions, incorrect_predictions, result = get_predictions(decision_tree, test_data)
            correct_predictions_list.append(correct_predictions)           
        else: 
            for t in xrange(iterations):
                random.seed(t)
                indices = random.sample(range(total_data_count), int(math.ceil(percentages_list[i]*total_data_count/float(100.0))))

                data_subset = data[indices] 
                decision_tree = build_decision_tree(data_subset, class_label, features, m,-1, default_label)
                correct_predictions, incorrect_predictions, result = get_predictions(decision_tree, test_data)
                correct_predictions_list.append(correct_predictions)
        table[i,:] = numpy.array([percentages_list[i], min(correct_predictions_list)/float(total_test_count), numpy.mean(correct_predictions_list)/float(total_test_count), max(correct_predictions_list)/float(total_test_count)])
    
    pylab.figure("Figure1: Plot of Prediction Accuracy as a function of Training Data size (Part [2])")
    pylab.plot(table[:,0], table[:,1], 'rx')
    pylab.plot(table[:,0], table[:,1], 'r', label = 'minimum')
    pylab.hold(True)
    pylab.plot(table[:,0], table[:,2], 'bx')
    pylab.plot(table[:,0], table[:,2], 'b', label = 'mean')
    pylab.hold(True)
    pylab.plot(table[:,0], table[:,3], 'gx')
    pylab.plot(table[:,0], table[:,3], 'g', label = 'maximum')
    pylab.hold(True)
    
    deviations = []
    deviations.append(table[:,2] - table[:,1])
    deviations.append(table[:,3] - table[:,2])
    pyplot.errorbar(table[:,0], table[:,2], deviations, fmt='-o')
    
    pylab.title("Plot of Prediction Accuracy as a function of Training Data size")
    pylab.ylabel("Accuracy on Test Set")
    pylab.xlabel("Percentage of Training Set used")
    pylab.legend(loc = 'lower right')
    pylab.savefig("VarunSah_Homework1_Part2_" + dataset_name + ".jpg")

        
def plot_accuracy_vs_tree_size(data, test_data, m_list, dataset_name):
    dataset_name = dataset_name.split('_')[0]
    accuracy_list = []
    total_test_count = len(test_data)
    for m in m_list:
        decision_tree = build_decision_tree(data, class_label, features, m,-1, default_label)
        correct_predictions, incorrect_predictions, result = get_predictions(decision_tree, test_data)
        #print correct_predictions
        accuracy_list.append(correct_predictions/float(total_test_count))
    
    pylab.figure("Figure2: Plot of Prediction Accuracy as a function of Decision Tree size (Part [3])")
    pylab.plot(m_list, accuracy_list, 'bx')
    pylab.plot(m_list, accuracy_list, 'b')
    pylab.title("Plot of Prediction Accuracy as a function of Decision Tree size ")
    pylab.ylabel("Accuracy on Test Set")
    pylab.xlabel("m (inversely proportional to Decision Tree size)")
    pylab.savefig("VarunSah_Homework1_Part3_" + dataset_name + ".jpg") 

 
if __name__ == '__main__':
   
    # 0) take input and handle incorrect number of arguments 
    argv = sys.argv[1:]
    assert len(argv) == 3, incorrectUsageMessage()
    training_data_file = argv[0]
    test_data_file = argv[1]
    m = int(argv[2])
    
    # 1) load the training data set
    data, metadata = arff.loadarff(training_data_file)    
    
    features = numpy.array(metadata.names())
    class_label = features[-1]  
    features = features[:-1]
    
    feature_types = numpy.array(metadata.types())
    class_label_type = feature_types[-1]
    feature_types = feature_types[:-1]
    feature_type_map = dict(zip(features, feature_types))
#     feature_range_map = get_feature_ranges(training_data_file)
    feature_range_map = {}
    for name in metadata.names():
        feature_range_map[name] = metadata[name][1]
     
    unique_class_values = get_set_of_values(data, class_label)
    unique_class_range = feature_range_map[class_label]
    
    default_label = unique_class_range[0]
    # 2) generate a decision tree using training data set
    output_string=""
    decision_tree = build_decision_tree(data, class_label, features, m, -1, default_label)
    
    if type(decision_tree) is str:
        output_string = output_string[2:]
    print output_string[:-1]
       
    # 3) load the test data set
    test_data, metadata = arff.loadarff(test_data_file)
    
    # 4) make predictions for the test data set using the decision tree
    correct_predictions, incorrect_predictions, result_line = get_predictions(decision_tree, test_data)
    print result_line
    
    # 5) plot accuracy as a function of training set size 
    iterations = 10
    m_part2 = 4     # given m = 4 in part[2]
    percentages_list = [5.0, 10.0, 20.0, 50.0, 100.0]
#     plot_accuracy_vs_training_set_size(data, test_data, percentages_list, iterations, m_part2, training_data_file)
    
    # 6) plot accuracy as a function of decision_tree size (represented by value of m)
    m_list = [2.0, 5.0, 10.0, 20.0]   
#     plot_accuracy_vs_tree_size(data, test_data, m_list, training_data_file)
    
#     pylab.show()
