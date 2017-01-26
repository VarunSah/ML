'''
Created on Oct 13, 2016

@author: varun
'''
import sys
import numpy
from math import sqrt as square_root

from scipy.io import arff

def incorrectUsageMessage():
    print "Incorrect usage. Correct format : kNN <train-set-file> <test-set-file> k"

def getEuclideanDistance(training_instance, test_instance, number_of_features):
    squared_distance = 0
#     print test_instance
#     print training_instance
    for i in range(number_of_features):
        squared_distance += pow((test_instance[i] - training_instance[i]), 2)        
    distance = square_root(squared_distance)
    return distance

def getNeighbors(training_data, test_instance, k):
    distances = []
    number_of_features = len(test_instance)-1
    number_of_training_instances = len(training_data)
    for i in range(number_of_training_instances):
        training_instance= training_data[i]
        distance = getEuclideanDistance( training_instance, test_instance, number_of_features)
        distances.append((training_data[i], distance))
    distances.sort(key= lambda distances : distances[1])
    neighbors = distances
    neighbors.sort(key=lambda tup : (tup[1]))
    neighbors = neighbors[:k]
    if(task == 'regression'):
#         print neighbors
        return neighbors 
    neighbors = customSort(neighbors, output_arff_order)
#     print neighbors
    return neighbors

def customSort(neighbors, output_arff_order):
    neighbors.sort(key=lambda tup : (tup[1], output_arff_order.index(tup[0][-1]))) 
    return neighbors       

def determineClass(neighbors, k):
    current_neighbor_output = '';
    class_label_frequencies = []
    encountered_class_labels = []
    for i in range(k):
        current_neighbor_output = neighbors[i][0][-1]
        if(encountered_class_labels.count(current_neighbor_output) == 0):
            class_label_frequencies.append(list((current_neighbor_output, 1)))
            encountered_class_labels.append(current_neighbor_output)
        else:
            label_index = encountered_class_labels.index(current_neighbor_output)
            class_label_frequencies[label_index][1] += 1    
    predicted_class = getMostFrequentClass(class_label_frequencies)
    return predicted_class

def determineClassSimplified(neighbors, k):
    class_label_frequencies = []
    for class_label in output_arff_order:
        class_label_frequencies.append(list((class_label, 0)))
    for i in range(k):        
        current_neighbor_output = neighbors[i][0][-1]
        label_index = output_arff_order.index(current_neighbor_output)
        class_label_frequencies[label_index][1] += 1
        
    predictedClass = getMostFrequentClass(class_label_frequencies)
    return predictedClass

def getMostFrequentClass(class_label_frequencies):
    most_frequent_class = ''
    max_frequency = -1
#     print class_label_frequencies
    for i in range(len(class_label_frequencies)):
        if(class_label_frequencies[i][1] > max_frequency):
            max_frequency = class_label_frequencies[i][1]
            most_frequent_class = class_label_frequencies[i][0]
#     print most_frequent_class
    return most_frequent_class

def getResponseValue(neighbors, k):
    response = 0
    for i in range(k):
        response += neighbors[i][0][-1]
    response = response/k 
    return response 

if __name__ == '__main__':
   
    # 0) take input and handle incorrect number of arguments 
    argv = sys.argv[1:]
    assert len(argv) == 3, incorrectUsageMessage()
    training_data_file = argv[0]
    test_data_file = argv[1]
    k = int(argv[2])
    
    # 1) load the training data set
    training_data, metadata = arff.loadarff(training_data_file)        
   
    features = numpy.array(metadata.names())
    response_or_class = features[-1]  
    output_arff_order =[]
    
    if(response_or_class == 'response'):
        task = 'regression'
    elif (response_or_class == 'class'):
        task = 'classification'   
        output_arff_order =  metadata[response_or_class][1]     
    
    test_data, test_metadata = arff.loadarff(test_data_file)
    number_of_test_instances = len(test_data)
    test_predictions = []
    
    correct_classifications = 0
    
    regression_errors = []
    total_regression_error = 0
    accuracy = 0.0
    
    print 'k value : ' + str(k)
    
    if(task == 'classification'):        
#         output_arff_order =  metadata[response_or_class][1]
#         print output_arff_order        
        for m in range(number_of_test_instances):
#             print m+1
            test_predictions.append(determineClassSimplified(getNeighbors(training_data, test_data[m], k), k))
            predicted_class = test_predictions[m]
            actual_class = test_data[m][-1]
            print 'Predicted class : ' + predicted_class + '\tActual class : ' + actual_class
            if(predicted_class == actual_class):
                correct_classifications +=1
        accuracy = float(correct_classifications)/float(number_of_test_instances)
        print 'Number of correctly classified instances : ' + str(correct_classifications) +'\nTotal number of instances : ' + str(number_of_test_instances) + '\nAccuracy : '+ "{0:.16f}".format(accuracy)
    elif (task == 'regression'):
        for j in range(number_of_test_instances):
            test_predictions.append(getResponseValue(getNeighbors(training_data, test_data[j], k), k))
            predicted_response = test_predictions[j]
            actual_response = test_data[j][-1]
            print 'Predicted value : ' + "{0:.6f}".format(predicted_response) + '\tActual value : ' + "{0:.6f}".format(actual_response)
            regression_errors.append(abs(predicted_response - actual_response))
            total_regression_error += regression_errors[j]
        mean_error = total_regression_error/number_of_test_instances
        print 'Mean absolute error : ' + "{0:.16f}".format(mean_error) + '\nTotal number of instances : ' + str(number_of_test_instances)