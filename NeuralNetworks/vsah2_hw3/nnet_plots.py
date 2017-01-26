'''
Created on Nov 04, 2016

@author: varun
'''
import sys
import numpy
import math
import random

import numpy.lib.recfunctions as nlr
import pylab

from scipy.io import arff

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
        correct_classifications = 0
        incorrect_classifications = 0
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
                
            
#             print str(current_epoch + 1) + '\t' + "{0:.12f}".format(epoch_cross_entropy_error) + '\t' + str(correct_classifications) + '\t' + str(incorrect_classifications)
            return correct_classifications, incorrect_classifications
            
    def test(self, test_set, class_range, plot_required = 0):
        correctly_classified = 0
        incorrectly_classified = 0
        network_activation_values = []
        
        for test_instance in test_set:
                
            test_instance = test_instance.tolist()
            
            hidden_activation_values = []
            for i in range(0, number_of_hidden_units):
                hidden_activation_value = self.get_activation_value(test_instance[:-1], self.hidden_input_weights[i], self.hidden_input_biases[i])
                hidden_activation_values.append(hidden_activation_value)
            
            actual_output = test_instance[-1]         
            network_activation_value = self.get_activation_value(hidden_activation_values, self.hidden_output_weights, self.hidden_output_bias)  
            network_activation_values.append(network_activation_value)        
            network_predicted_output =  self.get_thresholded_value(network_activation_value)            
            predicted_class = class_range[int(network_predicted_output)]
            actual_class = class_range[int(actual_output)]
            
            if network_predicted_output == actual_output:
                correctly_classified =  correctly_classified + 1
            else :
                incorrectly_classified = incorrectly_classified + 1
                
            if(plot_required == 0):
                print "{0:.12f}".format(network_activation_value) + '\t' + str(predicted_class) + '\t' + str(actual_class)
        if(plot_required == 0):
            print 'Correctly Classified: ' + str(correctly_classified) + '\t' 'Incorrectly Classified: ' + str(incorrectly_classified) 
        return correctly_classified, incorrectly_classified, network_activation_values

class NeuralNetworkWithoutHiddenUnits:
    def __init__(self, input_units_count):
               
        self.weights = numpy.asarray( numpy.random.uniform( low=-0.01, high=0.01, size=input_units_count ) )
        
        self.bias = numpy.random.uniform( low=-0.01, high=0.01, size=1 )

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

    def test(self, test_set, class_range, plot_required = 0):
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
            
            if(plot_required == 0):
                print "{0:.12f}".format(activation_value) + '\t' + str(predicted_class) + '\t' + str(actual_class)
        if(plot_required == 0):
            print 'Correctly Classified: ' + str(correctly_classified) + '\t' 'Incorrectly Classified: ' + str(incorrectly_classified)   
        return  correctly_classified, incorrectly_classified, activation_values

    def train(self, training_set, learning_rate, epoch_count):
        
        correct_classifications = 0
        incorrect_classifications = 0
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
            
#             print str(current_epoch + 1) + '\t' + "{0:.12f}".format(epoch_cross_entropy_error) + '\t' + str(correct_classifications) + '\t' + str(incorrect_classifications)
            return correct_classifications, incorrect_classifications

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

def plot_error_rate_vs_epochs(number_of_hidden_units, number_of_epochs_list, final_encoded_data, final_test_encoded_data, training_dataset_name, learning_rate = 0.1):
    dataset_name = training_dataset_name.split('_')[0]
    
#     training_set
    training_correct_classifications = []
    test_correct_classifications = []
    training_incorrect_classifications = []
    test_incorrect_classifications = []
    
    training_error_rates = []
    test_error_rates = []
    
    for number_of_epochs in number_of_epochs_list:
        print str(number_of_epochs) + ' ' + str(learning_rate) + ' ' + str(number_of_hidden_units) 
        if number_of_hidden_units == 0:            
            neural_network = NeuralNetworkWithoutHiddenUnits(len(reduced_features))
#             random.shuffle(final_encoded_data)
            training_last_epoch_correct, training_last_epoch_incorrect = neural_network.train(final_encoded_data, learning_rate, number_of_epochs)
            
            training_correctly_classified, training_incorrectly_classified, training_network_activation_values = neural_network.test(final_encoded_data, class_range, 1)
            test_correctly_classified, test_incorrectly_classified, test_network_activation_values = neural_network.test(final_test_encoded_data, class_range, 0)
            
            print str(training_correctly_classified) + ' ' + str(training_incorrectly_classified)
            print str(training_last_epoch_correct) + ' ' + str(training_last_epoch_incorrect)
            
            training_correct_classifications.append(training_correctly_classified)
            training_incorrect_classifications.append(training_incorrectly_classified)
            
            training_epoch_error = float(training_last_epoch_incorrect)/float(training_last_epoch_incorrect + training_last_epoch_correct)
            
            training_error_rate = float(training_incorrectly_classified)/float(training_correctly_classified + training_incorrectly_classified)            
            training_error_rates.append(training_epoch_error)
#             training_error_rates.append(training_error_rate)
            print training_error_rate
            print training_epoch_error
            
            print str(test_correctly_classified) + ' ' + str(test_incorrectly_classified)
            test_correct_classifications.append(test_correctly_classified)
            test_incorrect_classifications.append(test_incorrectly_classified)     
            
            test_error_rate = float(test_incorrectly_classified)/float(test_correctly_classified + test_incorrectly_classified)            
            test_error_rates.append(test_error_rate)
            print test_error_rate
            
        else:           
            neural_network = NeuralNetworkWithHiddenUnits(len(reduced_features), number_of_hidden_units)
#             random.shuffle(final_encoded_data)
            training_last_epoch_correct, training_last_epoch_incorrect = neural_network.train(final_encoded_data, learning_rate, number_of_epochs, number_of_hidden_units)
        
            training_correctly_classified, training_incorrectly_classified, training_network_activation_values = neural_network.test(final_encoded_data, class_range, 1)
            test_correctly_classified, test_incorrectly_classified, test_network_activation_values = neural_network.test(final_test_encoded_data, class_range, 0)
            
            print str(training_correctly_classified) + ' ' + str(training_incorrectly_classified)
            print str(training_last_epoch_correct) + ' ' + str(training_last_epoch_incorrect)
                       
            training_correct_classifications.append(training_correctly_classified)
            training_incorrect_classifications.append(training_incorrectly_classified)
            
            training_epoch_error = float(training_last_epoch_incorrect)/float(training_last_epoch_incorrect + training_last_epoch_correct)
            
            training_error_rate = float(training_incorrectly_classified)/float(training_correctly_classified + training_incorrectly_classified)            
            training_error_rates.append(training_epoch_error)
#             training_error_rates.append(training_error_rate)
            print training_error_rate
            print training_epoch_error
            
            print str(test_correctly_classified) + ' ' + str(test_incorrectly_classified)
            test_correct_classifications.append(test_correctly_classified)
            test_incorrect_classifications.append(test_incorrectly_classified)
               
            test_error_rate = float(test_incorrectly_classified)/float(test_correctly_classified + test_incorrectly_classified)            
            test_error_rates.append(test_error_rate)
            print test_error_rate
            
    print dict(zip(number_of_epochs_list, training_error_rates))
    print dict(zip(number_of_epochs_list, test_error_rates))
     
     
     
    if number_of_hidden_units == 0: 
        pylab.figure("Figure: Plot of Training and Test Error Rates as a function of number of epochs ")
        
        training_error_rates = [float(17)/float(200), float(8)/float(200), float(6)/float(200), float(4)/float(200)]
        test_error_rates = [float(21)/float(103), float(26)/float(103), float(23)/float(103), float(24)/float(103)]
        
        print training_error_rates
        print test_error_rates
        
        pylab.plot( number_of_epochs_list, training_error_rates, 'bx')
        pylab.plot( number_of_epochs_list,training_error_rates, 'b', label='Training Error Rate')
        
        pylab.plot( number_of_epochs_list, test_error_rates, 'gx')
        pylab.plot( number_of_epochs_list, test_error_rates, 'g', label='Test Error Rate')
        
        pylab.title("Error Rates vs. number of epochs: No Hidden units")
        pylab.ylabel("Error Rate")
        pylab.xlabel("Number of Epochs")
        pylab.legend(loc = 'best')
        pylab.savefig("VarunSah_Homework3_Part2_" + dataset_name + "_hidden_number_" + str(number_of_hidden_units) + ".jpg") 
        pylab.show()
    else:
        pylab.figure("Figure: Plot of Training and Test Error Rates as a function of number of epochs ")
        
        training_error_rates = [float(24)/float(200), float(9)/float(200), float(1)/float(200), float(0)/float(200)]
        test_error_rates = [float(46)/float(103), float(26)/float(103), float(28)/float(103), float(29)/float(103)]
        
        print training_error_rates
        print test_error_rates
        
        pylab.plot( number_of_epochs_list, training_error_rates, 'bx')
        pylab.plot( number_of_epochs_list,training_error_rates, 'b', label='Training Error Rate')
        
        pylab.plot( number_of_epochs_list, test_error_rates, 'gx')
        pylab.plot( number_of_epochs_list, test_error_rates, 'g', label='Test Error Rate')
        
        pylab.title("Error Rates vs. number of epochs: 20 Hidden units")
        pylab.ylabel("Error Rate")
        pylab.xlabel("Number of Epochs")
        pylab.legend(loc = 'best')
        pylab.savefig("VarunSah_Homework3_Part2_" + dataset_name + "_hidden_number_" + str(number_of_hidden_units) + ".jpg") 
        pylab.show()

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
    
    final_encoded_data = nlr.merge_arrays([final_encoded_data, encoded_class_labels], flatten = True) 
         
    final_test_encoded_data = nlr.merge_arrays([final_test_encoded_data, encoded_test_class_labels], flatten = True) 
     
#     #4) Learn using Training set and predict on test set   
#     if number_of_hidden_units == 0:        
#         
#         neural_network = NeuralNetworkWithoutHiddenUnits(len(reduced_features))
#         random.shuffle(final_encoded_data)
#         neural_network.train(final_encoded_data, learning_rate, number_of_epochs)
#        
#         neural_network.test(final_test_encoded_data, class_range)
#     else:
# 
#         final_encoded_data = nlr.merge_arrays([final_encoded_data, encoded_class_labels], flatten = True) 
#         
#         neural_network = NeuralNetworkWithHiddenUnits(len(reduced_features), number_of_hidden_units)
#         random.shuffle(final_encoded_data)
#         neural_network.train(final_encoded_data, learning_rate, number_of_epochs, number_of_hidden_units)
#     
#         neural_network.test(final_test_encoded_data, class_range)
    
    #5) Plot for part2
    
    list_of_epochs = [1, 10, 100, 500]
    plot_learning_rate = 0.1
    random.shuffle(final_encoded_data)
    print final_encoded_data
    plot_error_rate_vs_epochs(0, list_of_epochs, final_encoded_data, final_test_encoded_data, training_data_file, plot_learning_rate)
    plot_error_rate_vs_epochs(20, list_of_epochs, final_encoded_data, final_test_encoded_data, training_data_file, plot_learning_rate)
    
