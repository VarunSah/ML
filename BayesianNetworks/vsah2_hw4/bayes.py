'''
Created on Nov 26, 2016

@author: varun
'''
import math
import sys
import numpy
from scipy.io import arff
import random

def incorrectUsageMessage():
    print "Incorrect usage. Correct format : bayes <train-set-file> <test-set-file> <n|t>"

def class_value_formatter(class_value):
    if (class_value.startswith('\'') or class_value.startswith('"')) and (class_value.endswith('\'') or class_value.endswith('"')):
        class_value = class_value[1:-1]
    return class_value

class TAN:

    def __init__(self,training_data, ordered_features, features_possible_values, class_label):
        self.training_data =training_data
        self.class_label = class_label
        self.ordered_features = ordered_features #excludes class_label
        self.features_values = features_possible_values #includes class_label       
        self.individualFeatureCounts = {} 
        self.jointFeatureCounts = {} 
        self.classCounts = {} 
        self.MutualInformation = []
        self.parents = {}
               
        for feature in self.ordered_features:
            self.individualFeatureCounts[feature] = {}
            possible_feature_values = features_possible_values[feature]
            for possible_feature_value in possible_feature_values:
                self.individualFeatureCounts[feature][possible_feature_value] = {features_possible_values[class_label][0] : 0,features_possible_values[class_label][1] : 0}         
        for ith_feature in self.ordered_features:
            self.jointFeatureCounts[ith_feature] = {}
            possible_feature_i_values = features_possible_values[ith_feature]
            for jth_feature in self.ordered_features:
                self.jointFeatureCounts[ith_feature][jth_feature]={}
                possible_feature_j_values = features_possible_values[jth_feature]
                for feature_i_value in possible_feature_i_values:
                    self.jointFeatureCounts[ith_feature][jth_feature][feature_i_value] = {}
                    for feature_j_value in possible_feature_j_values:
                        self.jointFeatureCounts[ith_feature][jth_feature][feature_i_value][feature_j_value] = {features_possible_values[class_label][0] : 0,features_possible_values[class_label][1] : 0}       
        for class_value in features_possible_values[class_label]:
            self.classCounts[class_value] = 0        
        for feature in self.ordered_features:
            self.MutualInformation.append([-1]*len(self.ordered_features))
            self.parents[feature] = [class_label]

    def initializeCounts(self):
        for training_instance in self.training_data:
            expected_class = training_instance[self.class_label] 
            self.classCounts[expected_class] += 1
            for feature_i in self.ordered_features:
                feature_i_value = training_instance[feature_i]
                self.individualFeatureCounts[feature_i][feature_i_value][expected_class] += 1
                for feature_j in self.ordered_features:
                    feature_j_value = training_instance[feature_j]
                    self.jointFeatureCounts[feature_i][feature_j][feature_i_value][feature_j_value][expected_class] += 1

    def calculateMutualInformation(self):
        for feature_i_index in range(0,len(self.ordered_features)):
            feature_i = self.ordered_features[feature_i_index]
            for feature_j_index in range(feature_i_index,len(self.ordered_features)):
                feature_j = self.ordered_features[feature_j_index]
                if(feature_i_index != feature_j_index):
                    mutualInformation = 0.0 
                    for feature_i_value_index in range(0,len(self.features_values[feature_i])):
                        feature_i_value = self.features_values[feature_i][feature_i_value_index]
                        for feature_j_value_index in range(0,len(self.features_values[feature_j])):
                            feature_j_value = self.features_values[feature_j][feature_j_value_index]
                            for class_value in self.classCounts.keys():
                                joint_probability_term = ((self.jointFeatureCounts[feature_i][feature_j][feature_i_value][feature_j_value][class_value]+1)*1.0)/(len(self.training_data)+(len(self.features_values[feature_i])*len(self.features_values[feature_j])*len(self.classCounts.keys())))
                                probability_x_ij_y = ((self.jointFeatureCounts[feature_i][feature_j][feature_i_value][feature_j_value][class_value]+1)*1.0)/(self.classCounts[class_value]+(len(self.features_values[feature_i])*len(self.features_values[feature_j])))                                
                                probability_x_i_y = ((self.individualFeatureCounts[feature_i][feature_i_value][class_value]+1)*1.0)/(self.classCounts[class_value]+len(self.features_values[feature_i]))
                                probability_x_j_y = ((self.individualFeatureCounts[feature_j][feature_j_value][class_value]+1)*1.0)/(self.classCounts[class_value]+len(self.features_values[feature_j]))
                                log_term = (probability_x_ij_y/(probability_x_i_y*probability_x_j_y))
                                log_term = math.log(log_term,2)
                                mutualInformation += (joint_probability_term * log_term)
                    self.MutualInformation[feature_i_index][feature_j_index] = mutualInformation
                    self.MutualInformation[feature_j_index][feature_i_index] = mutualInformation

    def get_probability_class(self, class_value):
        probability_class = ((self.classCounts[class_value]+1)*1.0)/(len(self.training_data)+len(self.classCounts.keys()))
        return probability_class

    def get_probability_xi_class(self,attribute,attribute_value,class_value):
        probability_xi_class = ((self.individualFeatureCounts[attribute][attribute_value][class_value] + 1)*1.0)/(self.classCounts[class_value]+len(self.features_values[attribute]))
        return probability_xi_class

    def get_probability_xi_xj_class(self,attribute,parent_attribute,attribute_value,parent_attribute_value,class_value):
        probability_xi_xj_class = ((self.jointFeatureCounts[attribute][parent_attribute][attribute_value][parent_attribute_value][class_value] + 1)*1.0)/(self.individualFeatureCounts[parent_attribute][parent_attribute_value][class_value]+len(self.features_values[attribute]))
        return probability_xi_xj_class

    def createSpanningTree(self):
        maximum_spanning_tree = self.usePrimsAlgorithm(range(len(self.ordered_features)))       
        for edge in maximum_spanning_tree:
            source_node = self.ordered_features[edge[0]]
            destination_node = self.ordered_features[edge[1]]
            self.parents[destination_node].insert(0,source_node)

    def usePrimsAlgorithm(self,vertices):
        selected_vertices = set()
        selected_edges = set()
        selected_vertices.add(0) #Taking the vertex which first occurs in the file
        while len(selected_vertices) != len(vertices):
            candidate_edges = set()
            for selected_vertex in selected_vertices:
                for remaining_vertex in vertices:
                    if (remaining_vertex not in selected_vertices and self.MutualInformation[selected_vertex][remaining_vertex] != -1):
                        candidate_edges.add((selected_vertex,remaining_vertex))
            correct_edge = sorted(candidate_edges, cmp = self.compare)[0]
            selected_edges.add(correct_edge)
            selected_vertices.add(correct_edge[1])
        return selected_edges
    
    def compare(self,edge_a,edge_b):
        mutualInformation_edge_a = self.MutualInformation[edge_a[0]][edge_a[1]]
        mutualInformation_edge_b = self.MutualInformation[edge_b[0]][edge_b[1]]
        if (mutualInformation_edge_a > mutualInformation_edge_b):
            return -1
        elif (mutualInformation_edge_b > mutualInformation_edge_a):
            return 1
        elif (edge_a[0] < edge_b[0]): #tie-break first appearing in file
            return -1
        elif (edge_a[0] > edge_b[0]):
            return 1
        elif (edge_a[1] < edge_b[1]):
            return -1
        else:
            return 1 
           
    def classify(self,test_instance):
        class_value_1 = self.features_values[self.class_label][0]
        class_value_2 = self.features_values[self.class_label][1]
        probability_class_value_1 = self.get_probability_class(class_value_1)        
        probability_class_value_2 = self.get_probability_class(class_value_2)
        product_probability_class_value_1 = 1.0
        product_probability_class_value_2 = 1.0
        for feature in self.ordered_features:
            features_value = test_instance[feature]
            if (len(self.parents[feature]) == 1):
                product_probability_class_value_1 *= self.get_probability_xi_class(feature,features_value,class_value_1)
                product_probability_class_value_2 *= self.get_probability_xi_class(feature,features_value,class_value_2)
            else:
                parent_feature = self.parents[feature][0]
                parent_feature_value = test_instance[parent_feature]
                product_probability_class_value_1 *= self.get_probability_xi_xj_class(feature,parent_feature,features_value,parent_feature_value,class_value_1)
                product_probability_class_value_2 *= self.get_probability_xi_xj_class(feature,parent_feature,features_value,parent_feature_value,class_value_2)
        probability_class_value_1 = probability_class_value_1 * product_probability_class_value_1
        probability_class_value_2 = probability_class_value_2 * product_probability_class_value_2
        if (probability_class_value_1 > probability_class_value_2):
            posterior_probability = probability_class_value_1/(probability_class_value_1+probability_class_value_2)
            predicted_class = class_value_1
        else:
            posterior_probability = probability_class_value_2/(probability_class_value_1+probability_class_value_2)
            predicted_class = class_value_2
        return (predicted_class,posterior_probability)

    def test(self,testing_data):
        correctly_classified_count = 0
        for feature in self.ordered_features:
            output_str = feature
            for parent in self.parents[feature]:
                output_str += " " + parent
            print output_str
        print("")
        for instance in testing_data:
            predicted_class,posterior_probability = self.classify(instance)
            actual_class = instance[-1]
            if (predicted_class == actual_class):
                correctly_classified_count += 1
            print class_value_formatter(predicted_class) +" "+ class_value_formatter(actual_class) +" "+ "{0:.12f}".format(posterior_probability)
        print("\n"+str(correctly_classified_count))
        return correctly_classified_count, len(testing_data)

class NaiveBayes:

    def __init__(self,training_data, ordered_features, features_possible_values, class_label):
        self.training_data =training_data
        self.class_label = class_label
        self.ordered_features = ordered_features #excludes class_label
        self.features_values = features_possible_values #includes class_label      
        self.individualFeatureCounts = {} 
        for feature in self.ordered_features:
            self.individualFeatureCounts[feature] = {}
            possible_feature_values = features_possible_values[feature]
            for possible_feature_value in possible_feature_values:
                self.individualFeatureCounts[feature][possible_feature_value] = {features_possible_values[class_label][0] : 0,features_possible_values[class_label][1] : 0} 
        self.classCounts = {} 
        for class_value in features_possible_values[class_label]:
            self.classCounts[class_value] = 0

    def initializeCounts(self):
        for training_instance in self.training_data:
            expected_class = training_instance[self.class_label] 
            self.classCounts[expected_class] += 1
            for feature in self.ordered_features:
                feature_value = training_instance[feature]
                self.individualFeatureCounts[feature][feature_value][expected_class] += 1

    def get_probability_class(self, class_value):
        probability_class = ((self.classCounts[class_value]+1)*1.0)/(len(self.training_data)+len(self.classCounts.keys()))
        return probability_class

    def get_probability_xi_class(self,attribute,attribute_value,class_value):
        probability_xi_class = ((self.individualFeatureCounts[attribute][attribute_value][class_value] + 1)*1.0)/(self.classCounts[class_value]+len(self.features_values[attribute]))
        return probability_xi_class

    def classify(self,test_instance):
        class_value_1 = self.features_values[self.class_label][0]
        class_value_2 = self.features_values[self.class_label][1]
        probability_class_value_1 = self.get_probability_class(class_value_1)        
        probability_class_value_2 = self.get_probability_class(class_value_2)
        product_probability_class_value_1 = 1.0
        product_probability_class_value_2 = 1.0
        for feature in self.ordered_features:
            features_value = test_instance[feature]
            product_probability_class_value_1 *= self.get_probability_xi_class(feature,features_value,class_value_1)
            product_probability_class_value_2 *= self.get_probability_xi_class(feature,features_value,class_value_2)
        probability_class_value_1 = probability_class_value_1 * product_probability_class_value_1
        probability_class_value_2 = probability_class_value_2 * product_probability_class_value_2
        if (probability_class_value_1 > probability_class_value_2):
            posterior_probability = probability_class_value_1/(probability_class_value_1+probability_class_value_2)
            predicted_class = class_value_1
        else:
            posterior_probability = probability_class_value_2/(probability_class_value_1+probability_class_value_2)
            predicted_class = class_value_2
        return (predicted_class,posterior_probability)

    def test(self,testing_data):
        correctly_classified_count = 0
        for feature in self.ordered_features:
            print feature+" "+self.class_label
        print("")
        for instance in testing_data:
            predicted_class,posterior_probability = self.classify(instance)
            actual_class = instance[-1]
            if (predicted_class == actual_class):
                correctly_classified_count += 1
            print class_value_formatter(predicted_class) +" "+ class_value_formatter(actual_class) +" "+"{0:.12f}".format(posterior_probability)
        print("\n"+str(correctly_classified_count))
        return correctly_classified_count, len(testing_data)

def useNBAlgorithm(training_data, ordered_features, features_values, class_label, test_data):
    nb_structure = NaiveBayes(training_data, ordered_features, features_values, class_label)
    nb_structure.initializeCounts()
    return nb_structure.test(test_data)

def useTANAlgorithm(training_data, ordered_features, features_values, class_label, test_data):
    tan_structure = TAN(training_data, ordered_features, features_values, class_label)
    tan_structure.initializeCounts()
    tan_structure.calculateMutualInformation()
    tan_structure.createSpanningTree()
    return tan_structure.test(test_data)

def getKPartitions(total_data, total_class_label, total_class_range, k):
    partitions = []
    positive_training_instances = total_data[:]
    negative_training_instances = total_data[:]
    negative_indices = []
    positive_indices = []
    #Stratify
    for index in range(0, len(total_data)):
        if total_data[index][total_class_label] == total_class_range[0]:
            positive_indices.append(index)
        else:
            negative_indices.append(index)            
    positive_training_instances = numpy.delete(positive_training_instances,negative_indices)
    negative_training_instances = numpy.delete(negative_training_instances,positive_indices)
    #Partition
    postive_training_instances_avg_count = len(positive_training_instances)/k
    negative_training_instances_avg_count = len(negative_training_instances)/k    
    for partition_count in range(0,k):
        indices_to_be_removed = []
        random_positive_instances = random.sample(positive_training_instances,postive_training_instances_avg_count)
#         random_positive_instances = positive_training_instances[:postive_training_instances_avg_count]
        random_positive_instances = numpy.asarray(random_positive_instances)
        for index in range(0, len(random_positive_instances)):
            indices_to_be_removed.append(index)
        positive_training_instances = numpy.delete(positive_training_instances, indices_to_be_removed)
        partitions.append(random_positive_instances)        
    partition_count = 0
    while len(positive_training_instances) != 0:
        # Add remaining leftover instances one by one to the partitions until done
        partitions[partition_count] = numpy.append(partitions[partition_count], positive_training_instances[0])
        positive_training_instances = numpy.delete(positive_training_instances, 0)
        partition_count = (partition_count+1)% k
    for partition_count in range(0,k):
        indices_to_be_removed = []
        random_negative_instances = random.sample(negative_training_instances,negative_training_instances_avg_count)
#         random_negative_instances = negative_training_instances[:negative_training_instances_avg_count]
        random_negative_instances = numpy.asarray(random_negative_instances)
        for index in range(0, len(random_negative_instances)):
            indices_to_be_removed.append(index)
            partitions[partition_count] = numpy.append(partitions[partition_count], random_negative_instances[index])
        negative_training_instances = numpy.delete(negative_training_instances, indices_to_be_removed)         
    partition_count = 0
    while len(negative_training_instances) != 0:
        # Add remaining leftover instances one by one to the partitions until done
        partitions[partition_count] =  numpy.append( partitions[partition_count],negative_training_instances[0])
        negative_training_instances =  numpy.delete(negative_training_instances, 0)
        partition_count = (partition_count+1) % k          
    return partitions
    
if __name__ == '__main__':
   
    # 0) take input and handle incorrect number of arguments 
    argv = sys.argv[1:]
    assert len(argv) == 3, incorrectUsageMessage()
    training_data_file = argv[0]
    test_data_file = argv[1]
    algorithm = (argv[2])
     
    # 1) load the training data set and then test data set
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
     
    features_values = feature_range_map
    ordered_features = features
     
    test_data, test_metadata = arff.loadarff(test_data_file) 
     
    # 2) Create NB or TAN structure with training data and predict on test data
    if algorithm == 't':
        useTANAlgorithm(training_data, ordered_features, features_values, class_label, test_data)
    else:
        useNBAlgorithm(training_data, ordered_features, features_values, class_label, test_data)
        
#     #3) Open new combined training and test file
#     total_data, total_metadata =  arff.loadarff("chess-KingRookVKingPawn.arff") 
#       
#     total_features = numpy.array(total_metadata.names())
#     total_class_label = total_features[-1]
#     total_features =   total_features[:-1]
#       
#     total_feature_types = numpy.array(total_metadata.types())
#     total_class_label_type = total_feature_types[-1]
#     total_feature_types = total_feature_types[:-1]
#     total_feature_type_map = dict(zip(total_features, total_feature_types))
#       
#     total_feature_range_map = {}
#     for name in total_metadata.names():
#         total_feature_range_map[name] = total_metadata[name][1]     
#           
#     total_class_range = total_feature_range_map[total_class_label]    
#       
#     total_features_values = total_feature_range_map
#     total_ordered_features = total_features
#       
#     #4) Divide into 10 equal sized parts with equal proportions of different class_labels 
#     partitions = getKPartitions(total_data, total_class_label, total_class_range, 10)
#       
#     #5) Run loop over 10 divisions, making each division test set for that iteration
#     # Store accuracy of both TAN and NB for each of the 10 iterations.
#     test_accuracy_TAN = []
#     test_accuracy_NB = []
#       
#     for j in range(0, len(partitions)):
#         total_training_data = total_data[:]
#         total_training_data = numpy.delete(total_training_data, numpy.s_[::])
#         for i in range(0,len(partitions)) : 
#             if i != j:
#                 total_training_data = numpy.append(total_training_data, partitions[i])
#         total_training_data = numpy.array(total_training_data)
#         total_test_partition = numpy.array(partitions[j])
# #         print str(len(total_training_data)) + " " + str(len(total_test_partition))
#         correct_predictions_NB, total_predictions_NB = useNBAlgorithm(total_training_data, total_ordered_features, total_features_values, total_class_label, total_test_partition)
#         correct_predictions_TAN, total_predictions_TAN = useTANAlgorithm(total_training_data, total_ordered_features, total_features_values, total_class_label, total_test_partition)
#            
#         test_accuracy_NB.append(float(correct_predictions_NB)/float(total_predictions_NB))  
#         test_accuracy_TAN.append(float(correct_predictions_TAN)/float(total_predictions_TAN))  
#               
#     #6) Print the 10 tuples
#     print test_accuracy_NB
#     print test_accuracy_TAN
# #     for partition in partitions:
# #         print len(partition)