'''
Created on Nov 04, 2016

@author: varun
'''
import sys
import numpy
import math
import random

import numpy.lib.recfunctions as nlr

from scipy.io import arff
import pylab

def incorrectUsageMessage():
    print "Incorrect usage. Correct format : nnet l h e <train-set-file> <test-set-file>"
     
class NeuralNetworkWithHiddenUnits:
    def __init__(self, input_units_count, hidden_units_count):
        self.hidden_input_weights = numpy.asarray( numpy.random.uniform( low= -0.01, high=0.01, size=hidden_units_count*input_units_count ) )
        self.hidden_input_weights = self.hidden_input_weights.reshape(hidden_units_count, input_units_count)
        
        self.hidden_output_weights = numpy.asarray( numpy.random.uniform( low= -0.01, high=0.01, size=hidden_units_count ) )

        self.hidden_input_biases =  numpy.random.uniform( low= -0.01, high=0.01, size=hidden_units_count )

        self.hidden_output_bias = numpy.random.uniform( low= -0.01, high=0.01, size=1 )

    def sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-x))

    def dot(self, x, y):
        dot_product = 0.0
        for index in range(0, len(x)):
            dot_product += float(x[index] * y[index]) 
        return dot_product
#         return numpy.dot(x, y.T)

    def get_activation_value(self, encoded_instance, weights, bias):
        return self.sigmoid(self.dot(encoded_instance, weights) + bias)

    def get_thresholded_value(self, sigmoid_output):
        if sigmoid_output > 0.5:
            return 1.0
        else:
            return 0.0

    def train(self, training_set, learning_rate, epoch_count, number_of_hidden_units):
        for current_epoch in range(0, epoch_count):
            actual_outputs = []
            network_activation_values = []
            network_predicted_outputs = []
            network_errors = []            
            correct_classifications = 0
            incorrect_classifications = 0
            epoch_cross_entropy_error = 0.0
            for training_instance in training_set:
                
                training_instance = training_instance.tolist()
                
                hidden_activation_values = []
                hidden_errors = []
                for i in range(0, number_of_hidden_units):
                    hidden_activation_value = self.get_activation_value(training_instance[:-1], self.hidden_input_weights[i], self.hidden_input_biases[i])
                    hidden_activation_values.append(hidden_activation_value)
                
                actual_output = training_instance[-1]
                actual_outputs.append(actual_output)
                
                network_activation_value = self.get_activation_value(hidden_activation_values, self.hidden_output_weights, self.hidden_output_bias)
                network_activation_values.append(network_activation_value)
                
                network_predicted_output =  self.get_thresholded_value(network_activation_value)
                network_predicted_outputs.append(network_predicted_output)
                
                network_error = actual_output - network_activation_value
                network_errors.append(network_error)
                
                if network_predicted_output == actual_output:
                    correct_classifications = correct_classifications + 1
                else :
                    incorrect_classifications = incorrect_classifications + 1
                                
                #backprop 
                for i in range(0, number_of_hidden_units):
                    hidden_error = (hidden_activation_values[i])*(1 - hidden_activation_values[i]) * network_error * self.hidden_output_weights[i]
                    hidden_errors.append(hidden_error)
                    
#                 print hidden_activation_values
                self.hidden_output_bias += learning_rate * network_error * 1.0
                for i in range(0, number_of_hidden_units):
                    self.hidden_output_weights[i] +=  learning_rate * network_error * hidden_activation_values[i]
                                      
                    self.hidden_input_biases[i] += learning_rate * hidden_errors[i] * 1.0
                    for j in range(0, len(self.hidden_input_weights[i])):
                        self.hidden_input_weights[i][j] +=  learning_rate * hidden_errors[i] * training_instance[j]
                        
            for i in range(len(actual_outputs)):
                epoch_cross_entropy_error += get_cross_entropy(actual_outputs[i], network_activation_values[i])   
                

            print str(current_epoch + 1) + '\t' + "{0:.12f}".format(epoch_cross_entropy_error) + '\t' + str(correct_classifications) + '\t' + str(incorrect_classifications)
            
    def test(self, test_set, class_range):
        correctly_classified = 0
        incorrectly_classified = 0
        activation_values = []
        for test_instance in test_set:
                
            test_instance = test_instance.tolist()
            
            hidden_activation_values = []
            for i in range(0, number_of_hidden_units):
                hidden_activation_value = self.get_activation_value(test_instance[:-1], self.hidden_input_weights[i], self.hidden_input_biases[i])
                hidden_activation_values.append(hidden_activation_value)
            
            actual_output = test_instance[-1]         
            network_activation_value = self.get_activation_value(hidden_activation_values, self.hidden_output_weights, self.hidden_output_bias)
            activation_values.append(network_activation_value)          
            network_predicted_output =  self.get_thresholded_value(network_activation_value)            
            predicted_class = class_range[int(network_predicted_output)]
            actual_class = class_range[int(actual_output)]
            
            if network_predicted_output == actual_output:
                correctly_classified =  correctly_classified + 1
            else :
                incorrectly_classified = incorrectly_classified + 1
                
            print "{0:.12f}".format(network_activation_value) + '\t' + str(predicted_class) + '\t' + str(actual_class)
        print 'Correctly Classified: ' + str(correctly_classified) + '\t' 'Incorrectly Classified: ' + str(incorrectly_classified) 
        return activation_values    

class NeuralNetworkWithoutHiddenUnits:
    def __init__(self, input_units_count):
               
        self.weights = numpy.asarray( numpy.random.uniform( low= -0.01, high=0.01, size=input_units_count ) )
        
        self.bias = numpy.random.uniform( low= -0.01, high=0.01, size=1 )

    def sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-x))

    def dot(self, x, y):
        dot_product = 0.0
        for index in range(0, len(x)):
            dot_product += float(x[index] * y[index]) 
        return dot_product
#         return numpy.dot(x, y.T)

    def get_activation_value(self, encoded_instance):
        return self.sigmoid(self.dot(encoded_instance, self.weights)+self.bias)

    def get_thresholded_value(self, sigmoid_output):
        if sigmoid_output > 0.5:
            return 1.0
        else:
            return 0.0

    def predict(self, test_instance):
        return self.get_thresholded_value(self.get_activation_value(test_instance))

    def test(self, test_set, class_range):
        correctly_classified = 0
        incorrectly_classified = 0
        activation_values = []
        for test_instance in test_set:
            test_instance = test_instance.tolist()
            activation_value = self.get_activation_value(test_instance[:-1])
            activation_values.append(activation_value)
            predicted_value = self.get_thresholded_value(activation_value)
            actual_value = test_instance[-1]
            predicted_class = class_range[int(predicted_value)]
            actual_class = class_range[int(actual_value)]
            
            if(predicted_class == actual_class):
                correctly_classified += 1
            else:
                incorrectly_classified +=1
            
            print "{0:.12f}".format(activation_value) + '\t' + str(predicted_class) + '\t' + str(actual_class)
        print 'Correctly Classified: ' + str(correctly_classified) + '\t' 'Incorrectly Classified: ' + str(incorrectly_classified)    
        return activation_values
    
    def train(self, training_set, learning_rate, epoch_count):

        for current_epoch in range(0, epoch_count):
            activation_values = []
            predicted_values = []
            actual_values = []
            correct_classifications = 0
            incorrect_classifications = 0
            epoch_cross_entropy_error = 0.0
            for training_instance in training_set:

                training_instance = training_instance.tolist()

                activation_value = self.get_activation_value(training_instance[:-1])
                activation_values.append(activation_value)
                predicted_output = self.get_thresholded_value(activation_value)
                predicted_values.append(predicted_output)
                error_derivates = []
                
                actual_output = training_instance[-1]
                actual_values.append(actual_output)
                
                if predicted_output == actual_output:
                    correct_classifications = correct_classifications + 1
                else :
                    incorrect_classifications = incorrect_classifications + 1
                
                self.bias += learning_rate * (actual_output - activation_value) * 1.0 
                
                for index in range(0, len(self.weights)):
                    x_i = training_instance[index] 
                    error_derivates.append((actual_output - activation_value) 
                                           * x_i)
                    self.weights[index] += learning_rate * error_derivates[index]

            for i in range(len(actual_values)):
                epoch_cross_entropy_error += get_cross_entropy(actual_values[i], activation_values[i])
            
            print str(current_epoch + 1) + '\t' + "{0:.12f}".format(epoch_cross_entropy_error) + '\t' + str(correct_classifications) + '\t' + str(incorrect_classifications)

def get_cross_entropy(y, o):
    return numpy.sum(numpy.nan_to_num(-y * numpy.log(o) - (1 - y) * numpy.log(1 - o)))

def encode_class_label(true_class_labels):
    encoded_class_labels = [float(class_range.index(x)) for x in true_class_labels]
    return encoded_class_labels

def standardize(data, column_name, feature_mean_map, feature_std_map):
    
    column = data[column_name]
    mean = numpy.mean(column, dtype=numpy.float64)
    feature_mean_map[column_name] = mean
    std = numpy.std(column, dtype=numpy.float64)
    feature_std_map[column_name] = std
    
    data[column_name] = [x - mean for x in data[column_name]]
    data[column_name] = [x / std for x in data[column_name]]
        
    return data, feature_mean_map, feature_std_map

def standardize_test(test_data, column_name, training_feature_mean_map, training_feature_std_map):
    training_mean = training_feature_mean_map[column_name]
    training_std = training_feature_std_map[column_name]
    
    test_data[column_name] = [x - training_mean for x in test_data[column_name]]
    test_data[column_name] = [x / training_std for x in test_data[column_name]]
    
    return test_data

def one_hot_encode(data, column_name, encoded_features):
    column = data[column_name].tolist()
    
    new_column_names = {}
    
    for column_value in feature_range_map[column_name]:
        new_column_names[column_value] = column_name+"_"+str(column_value)
        new_column = numpy.array( numpy.zeros(len(column)), dtype=[(new_column_names[column_value], int)])
        
        for index in range(len(data)):
            if (data[index][column_name] == column_value):
                new_column[index][0] =  1.0
        
        data = nlr.merge_arrays([data, new_column], flatten = True)
    
    encoded_features = numpy.delete(encoded_features, encoded_features.tolist().index( column_name))
    encoded_features = numpy.append(encoded_features, new_column_names.values())
    
    return data, encoded_features

def get_roc_plotting_data(roc_array, num_pos, num_neg):
    TP = 0
    FP = 0
    FPR = 0.0
    TPR = 0.0
    last_TP = 0
    x_outputs =[]
    y_outputs = []
    x_outputs.append(0)
    y_outputs.append(0)
    for i in range (1, len(roc_array)):
#     // find thresholds where there is a pos instance on high side, neg instance on low side
        if ( roc_array[i][0] != roc_array[i-1][0] ) and ( roc_array[i][1] == 0.0 ) and ( TP > last_TP ):
            FPR = float(FP) / float(num_neg)
            TPR = float(TP) / float(num_pos)
#             output (FPR, TPR) coordinate
            x_outputs.append(FPR)
            y_outputs.append(TPR)
            last_TP = TP
        if (roc_array[i][1] == 1.0):
            TP += 1
        else:
            FP +=1
    FPR = float(FP) / float(num_neg)
    TPR = float(TP) / float(num_pos)
#     x_outputs.append(FPR)
#     y_outputs.append(TPR)
    x_outputs.append(1)
    y_outputs.append(1)
    return x_outputs, y_outputs

if __name__ == '__main__':
 
    # 0) take input and handle incorrect number of arguments 
    argv = sys.argv[1:]
    assert len(argv) == 5, incorrectUsageMessage()
    
    learning_rate = float(argv[0])
    number_of_hidden_units = int(argv[1])
    number_of_epochs = int(argv[2])   
    training_data_file = argv[3]
    test_data_file = argv[4]
    
    # 1) load the training data set
    training_data, metadata = arff.loadarff(training_data_file) 
    
    
    features = numpy.array(metadata.names())
    class_label = features[-1]  
    features = features[:-1]
    
    feature_types = numpy.array(metadata.types())
    class_label_type = feature_types[-1]
    feature_types = feature_types[:-1]
    feature_type_map = dict(zip(features, feature_types))
    
    feature_range_map = {}
    for name in metadata.names():
        feature_range_map[name] = metadata[name][1]     
        
    class_range = feature_range_map[class_label]    
    feature_range_map.pop(class_label, None)
    
    feature_mean_map = {}
    feature_std_map = {}
      
    # 2) encode training data using one hot encoding.  
    encoded_training_data = training_data.copy()

    encoded_features = features.copy()
    for feature in features:
        if(feature_type_map[feature] != 'numeric'):
            encoded_training_data, encoded_features = one_hot_encode(encoded_training_data, feature, encoded_features)
        else:
            encoded_training_data, feature_mean_map, feature_std_map = standardize(encoded_training_data, feature, feature_mean_map, feature_std_map)
    
    true_class_labels = encoded_training_data[class_label]
    encoded_class_labels = encode_class_label(true_class_labels)
    
    reduced_features = [x for x in encoded_features]
    final_encoded_data = numpy.copy(encoded_training_data)
    final_encoded_data = final_encoded_data[reduced_features]
            
    #3) encode test data using one hot encoding
    test_data, metadata = arff.loadarff(test_data_file)
    encoded_test_data = test_data.copy()

    encoded_test_features = features.copy()
    for feature in features:
        if(feature_type_map[feature] != 'numeric'):
            encoded_test_data, encoded_test_features = one_hot_encode(encoded_test_data, feature, encoded_test_features)
        else:
            encoded_test_data = standardize_test(encoded_test_data, feature, feature_mean_map, feature_std_map)

    true_test_class_labels = encoded_test_data[class_label]
    encoded_test_class_labels = encode_class_label(true_test_class_labels)
    
    reduced_test_features = [x for x in encoded_test_features]
    final_test_encoded_data = numpy.copy(encoded_test_data)
    final_test_encoded_data = final_test_encoded_data[reduced_test_features]
         
    #4) Learn using Training set and predict on test set    
    if number_of_hidden_units == 0:
        final_encoded_data = nlr.merge_arrays([final_encoded_data, encoded_class_labels], flatten = True) 
        
        neural_network = NeuralNetworkWithoutHiddenUnits(len(reduced_features))
        random.shuffle(final_encoded_data)
        neural_network.train(final_encoded_data, learning_rate, number_of_epochs)
        
        final_test_encoded_data = nlr.merge_arrays([final_test_encoded_data, encoded_test_class_labels], flatten = True) 
        network_activation_values = neural_network.test(final_test_encoded_data, class_range)        
        
    else:
        final_encoded_data = nlr.merge_arrays([final_encoded_data, encoded_class_labels], flatten = True) 
        
        neural_network = NeuralNetworkWithHiddenUnits(len(reduced_features), number_of_hidden_units)
        random.shuffle(final_encoded_data)
        neural_network.train(final_encoded_data, learning_rate, number_of_epochs, number_of_hidden_units)
    
        final_test_encoded_data = nlr.merge_arrays([final_test_encoded_data, encoded_test_class_labels], flatten = True) 
        network_activation_values  = neural_network.test(final_test_encoded_data, class_range)
    
#     encoded_test_class_labels
    
    roc_array = []
    num_pos = 0
    num_neg = 0
    for i in range(0, len(final_test_encoded_data)):
        if(network_activation_values[i] < 0.5):
            network_activation_values[i] = 1.0 - network_activation_values[i]
        if(encoded_test_class_labels[i] == 1.0):
            num_pos += 1
        else:
            num_neg +=1
    roc_array =  zip(network_activation_values, encoded_test_class_labels) 
    print num_pos
    print num_neg
    print roc_array 
    
    roc_array.sort(key= lambda roc_array : roc_array[0], reverse=True)
    
    print roc_array
    
    x_axis, y_axis = get_roc_plotting_data(roc_array, num_pos, num_neg)
    print class_range[0]
    print class_range[1]
    print x_axis
    print y_axis
    
    pylab.plot( x_axis, y_axis, 'bx')
    pylab.plot( x_axis, y_axis, 'b')
    
    
    pylab.title("ROC for heart dataset with l = 0.01, h = 10, e = 20)")
    pylab.ylabel("True Positive Rate")
    pylab.xlabel("False Positive rate")
    pylab.legend(loc = 'best')
    pylab.savefig("VarunSah_Homework3_Part3_" + "lymph" + "_ROC" + ".jpg") 
    pylab.show()
