import math
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import random


min_samples_in_node = 1
df = pd.read_csv('/mnt/92026878026862E9/python_files/winequality-red.csv',encoding='latin-1')
X = df.iloc[:, :-1].values.tolist()
Y = df.iloc[:, -1].values.tolist()


class DTNode:
    splitFeatureIdx = None
    splitFeatureType = None
    exp = None
    lchild = None
    rchild = None

def getDistribution(indices, Y_train, weights):
    d = dict()
    expected = None
    maxWeight= 0
    for idx in indices:
        if Y_train[idx] in d:
            d[Y_train[idx]]+= weights[idx]
        else:
            d[Y_train[idx]]= weights[idx]
        if maxWeight < d[Y_train[idx]]:
            maxWeight = d[Y_train[idx]]
            expected = Y_train[idx]
        
    return d, expected

def getInformation(distribution, sumOfWeights):
    information = 0
    for key in distribution:
        p = distribution[key]/sumOfWeights
        information += p*math.log2(p)
    return -information 

def decisionTree(X_train, Y_train, feature_indices, weights, max_depth = 6):
    distribution, expected = getDistribution(range(0, len(X_train)), Y_train, weights)
    information = getInformation(distribution, 1)
    return makeDecisionTree(X_train, Y_train, set(range(len(X_train))), information, expected, 1, feature_indices, max_depth, weights)

def sumOfWeights(indices, weights):
    weightSum = 0
    for index in indices:
        weightSum += weights[index]
    return weightSum

def makeDecisionTree(X_train, Y_train, indices, information, exp, depth, feature_indices, max_depth, weights):
    if information == 0 or depth >= max_depth or len(indices) <= min_samples_in_node:
        node = DTNode()
        node.splitFeatureIdx = -1
        node.exp = exp
        return node
    featureCount = len(X_train[0])
    left_information = None
    right_information = None
    left_indices = set()
    right_indices = set()
    left_exp = None
    split_feature_idx = -1
    split_feature_type = None
    right_exp = None
    max_information_gain = 0
    for featureIdx in range(0, featureCount):
        data_points = []
        for i in indices:
            data_points.append((X_train[i][featureIdx],i))
        data_points.sort()        
        isDiscreteFeature = isinstance(data_points[0][0],str)
        distinct_points = OrderedDict()
        for data_point in data_points:
            if data_point[0] in distinct_points:
                distinct_points[data_point[0]].add(data_point[1])
            else:
                distinct_points[data_point[0]] = set([data_point[1]])

        prev_indices = set()
        for ty, type_indices in distinct_points.items():
            if isDiscreteFeature:
                non_type_indices = indices - type_indices
            else:
                type_indices = type_indices | prev_indices
                non_type_indices = indices - type_indices
                prev_indices = type_indices         
            type_indices_distribution, type_exp = getDistribution(type_indices, Y_train, weights)
            non_type_indices_distribution , non_type_exp = getDistribution(non_type_indices, Y_train, weights)
            type_information = getInformation(type_indices_distribution, sumOfWeights(type_indices, weights))
            non_type_information = getInformation(non_type_indices_distribution, sumOfWeights(non_type_indices, weights))
            information_gain = information - (type_information*len(type_indices) + non_type_information*len(non_type_indices))/len(indices)
    
            if information_gain > max_information_gain:
                max_information_gain = information_gain
                left_indices = type_indices
                left_exp = type_exp
                left_information = type_information
                right_indices = non_type_indices
                right_exp = non_type_exp
                right_information = non_type_information
                split_feature_idx = featureIdx
                split_feature_type = ty

    if split_feature_idx == -1:
        node = DTNode()
        node.splitFeatureIdx = -1
        node.exp = exp
        return node
    
    node = DTNode()
    node.lchild = makeDecisionTree(X_train, Y_train, left_indices, left_information, left_exp, depth + 1, feature_indices, max_depth, weights)
    node.rchild = makeDecisionTree(X_train, Y_train, right_indices, right_information, right_exp, depth + 1, feature_indices, max_depth, weights)
    node.splitFeatureIdx = feature_indices[split_feature_idx]
    node.splitFeatureType = split_feature_type
    return node

def regressionTree(X_train, Y_train, feature_indices, max_depth = 6):
    variance = getVariance(Y_train, list(range(0, len(Y_train))))
    expected = getExpectation(Y_train, list(range(0, len(Y_train))))
    return makeRegressionTree(X_train, Y_train, set(range(len(X_train))), variance, expected, 1, feature_indices, max_depth)

def getVariance(Y, indices):
    if len(indices) == 0:
        return 0
    mean = getExpectation(Y, indices)
    variance = 0
    for i in indices:
        variance = variance + (mean - Y[i])*(mean - Y[i])
    return variance/len(indices)

def getExpectation(Y, indices):
    if len(indices) == 0:
        return 0
    mean = 0
    for i in indices:
        mean = mean + Y[i]
    return mean/len(indices)

def makeRegressionTree(X_train, Y_train, indices, variance, exp, depth, feature_indices, max_depth):
    if depth >= max_depth or len(indices) <= min_samples_in_node:
        node = DTNode()
        node.splitFeatureIdx = -1
        node.exp = exp
        return node
    featureCount = len(X_train[0])
    left_variance = None
    right_variance = None
    left_indices = set()
    right_indices = set()
    left_exp = None
    split_feature_idx = -1
    split_feature_type = None
    right_exp = None
    max_variance_reduction = 0
    for featureIdx in range(0, featureCount):
        data_points = []
        for i in indices:
            data_points.append((X_train[i][featureIdx],i))
        data_points.sort()        
        isDiscreteFeature = isinstance(data_points[0][0],str)
        distinct_points = OrderedDict()
        for data_point in data_points:
            if data_point[0] in distinct_points:
                distinct_points[data_point[0]].add(data_point[1])
            else:
                distinct_points[data_point[0]] = set([data_point[1]])

        prev_indices = set()
        for ty, type_indices in distinct_points.items():
            if isDiscreteFeature:
                non_type_indices = indices - type_indices
            else:
                type_indices = type_indices | prev_indices
                non_type_indices = indices - type_indices
                prev_indices = type_indices     
            type_exp = getExpectation(Y_train, type_indices)
            non_type_exp = getExpectation(Y_train, non_type_indices)
            type_variance = getVariance(Y_train, type_indices)
            non_type_variance = getVariance(Y_train, non_type_indices)
            variance_reduction = variance - (len(type_indices)*type_variance + len(non_type_indices)*non_type_variance)/(len(type_indices)+len(non_type_indices))  
    
            if variance_reduction > max_variance_reduction:
                max_variance_reduction = variance_reduction
                left_indices = type_indices
                left_exp = type_exp
                left_variance = type_variance
                right_indices = non_type_indices
                right_exp = non_type_exp
                right_variance = non_type_variance
                split_feature_idx = featureIdx
                split_feature_type = ty

    if split_feature_idx == -1:
        node = DTNode()
        node.splitFeatureIdx = -1
        node.exp = exp
        return node
    
    node = DTNode()
    node.lchild = makeRegressionTree(X_train, Y_train, left_indices, left_variance, left_exp, depth + 1, feature_indices, max_depth)
    node.rchild = makeRegressionTree(X_train, Y_train, right_indices, right_variance, right_exp, depth + 1, feature_indices, max_depth)
    node.splitFeatureIdx = feature_indices[split_feature_idx]
    node.splitFeatureType = split_feature_type
    return node



def evaluate(root : DTNode, x):
    if root.splitFeatureIdx == -1:
        return root.exp
    isDiscrete = isinstance(root.splitFeatureType, str)
    if isDiscrete and root.splitFeatureType == x[root.splitFeatureIdx]:
        return evaluate(root.lchild, x)
    elif isDiscrete and root.splitFeatureType != x[root.splitFeatureIdx]:
        return evaluate(root.rchild, x)
    elif root.splitFeatureType >= x[root.splitFeatureIdx]:
        return evaluate(root.lchild, x)
    else:
        return evaluate(root.rchild, x)

def test(trees, X_test, Y_test):
    B = len(trees)
    total_score = len(X_test)
    score = 0
    for i in range(0, len(X_test)):
        predictions = dict()
        for tree in range(0, B):
            prediction = evaluate(trees[tree], X_test[i])
            if prediction in predictions:
                predictions[prediction] += 1
            else:
                predictions[prediction] = 1
        majority_vote = None
        majority_vote_count = 0
        for prediction, count in predictions.items():
            if count > majority_vote_count:
                majority_vote_count = count
                majority_vote = prediction
        if majority_vote == Y_test[i]:
            score += 1
    return (score/total_score)*100

def testRegressionTree(trees, X_test, Y_test):
    B = len(trees)
    err = 0
    for i in range(0, len(X_test)):
        avg_prediction = 0
        for tree in range(0, B):
            prediction = evaluate(trees[tree], X_test[i])
            avg_prediction += prediction
        avg_prediction /= B
        err += (Y_test[i] - avg_prediction)^2
        
    return err/len(Y_test)
    

def sampling_with_replacement(X, Y, isClassification, B = 150):
    no_of_samples_for_each_tree = int(len(X) *0.8)
    trees = []
    total_features = len(X[0])
    for b in range(0, B):
        X_train = []
        Y_train = []
        for sample in range(0, no_of_samples_for_each_tree):
            training_example = random.randint(0, len(X)-1)
            X_train.append(X[training_example])
            Y_train.append(Y[training_example])
        if isClassification:
            trees.append(decisionTree(X_train, Y_train, list(range(total_features)), [1/no_of_samples_for_each_tree]*no_of_samples_for_each_tree))
        else:
            trees.append(regressionTree(X_train, Y_train, list(range(total_features))))
 
    return trees

def randomForest(X, Y, isClassification, B = 150):
    total_features = len(X[0])
    k = int(math.sqrt(total_features))
    no_of_samples_for_each_tree = int(len(X) *0.8)
    trees = []
    for b in range(0, B):
        X_train = []
        Y_train = []
        selected_features = random.sample(list(range(total_features)),k)
        for sample in range(0, no_of_samples_for_each_tree):
            training_example = random.randint(0, len(X) - 1) 
            row =[]
            for feature in selected_features:
                row.append(X[training_example][feature])
            X_train.append(row)
            Y_train.append(Y[training_example])
        if isClassification:
            trees.append(decisionTree(X_train, Y_train, list(range(total_features)), [1/no_of_samples_for_each_tree]*no_of_samples_for_each_tree))
        else:
            trees.append(regressionTree(X_train, Y_train, list(range(total_features))))
    return trees

def adaBoost(X, Y, count, weights):
    if count == 0:
        return []
    root = decisionTree(X, Y, list(range(0, len(X[0]))), weights, 2)
    total_samples = len(X)
    error_samples = set()
    for i in range(0, total_samples):
        if evaluate(root, X[i]) != Y[i]:
            error_samples.add(i)
    incorrect_weight = math.sqrt((total_samples - len(error_samples))/(len(error_samples)))
    amount_of_say = math.log(incorrect_weight)
    sum_of_all_weights = 0
    for i in range(0, total_samples):
        if i in error_samples:
            weights[i] *= incorrect_weight
        else:
            weights[i] /= incorrect_weight
        sum_of_all_weights += weights[i]
    
    for i in range(0, total_samples):
        weights[i] /= sum_of_all_weights
    trees = adaBoost(X, Y, count - 1, weights)
  
    trees.append((root, amount_of_say))
    return trees

def testAdaBoost(trees, X_test, Y_test):
    totalScore = len(X_test)
    score = 0
    for i in range(0, len(X_test)):
        countMap = dict()
        largestVotedCategory = None
        largestCategoryAmountOfSay = -1000
        for tree in trees:           
            amountOfSay = tree[1]
            y_predict = evaluate(tree[0], X_test[i])
            if y_predict in countMap:
                countMap[y_predict] += amountOfSay
            else:
                countMap[y_predict] = amountOfSay
            if largestCategoryAmountOfSay < countMap[y_predict]:
                largestCategoryAmountOfSay = countMap[y_predict]
                largestVotedCategory = y_predict
        if largestVotedCategory == Y_test[i]:
            score += 1
    return (score/totalScore)*100


no_of_samples_in_training_data = int(0.3 * len(X))
X_train = X[:no_of_samples_in_training_data]
Y_train = Y[:no_of_samples_in_training_data]
X_val = X[no_of_samples_in_training_data:]
Y_val = Y[no_of_samples_in_training_data:]


sampling_with_replacement_trees = sampling_with_replacement(X_train, Y_train, False)
print(f"---Sampling with replacement-- Your model score on training data is {test(sampling_with_replacement_trees, X_train, Y_train)}") 
print(f"---Sampling with replacement-- Your model score on test data is {test(sampling_with_replacement_trees, X_val, Y_val)}") 

random_forest_trees = randomForest(X_train, Y_train, 128, False)
print(f"---Random Forest-- Your model score on training data is {test(random_forest_trees, X_train, Y_train)}") 
print(f"---Random Forest-- Your model score on test data is {test(random_forest_trees, X_val, Y_val)}") 


'''trees = adaBoost(X_train, Y_train, 128, [1/len(X_train)]*len(X_train))
print(f"---AdaBoost -- Your model score on training data is {testAdaBoost(trees, X_train, Y_train)}") 
print(f"---AdaBoost -- Your model score on test data is {testAdaBoost(trees, X_val, Y_val)}") '''

'''tree = decisionTree(X_train, Y_train, list(range(0, len(X_train[0]))), [1/len(X_train)]*len(X_train),6) 
print(f"---DT-- Your model score on training data is {test([tree], X_train, Y_train)}") 
print(f"---DT-- Your model score on test data is {test([tree], X_val, Y_val)}")'''

tree = regressionTree(X_train, Y_train, list(range(0, len(X_train[0]))),6) 
print(f"---DT-- Your model score on training data is {testRegressionTree([tree], X_train, Y_train)}") 
print(f"---DT-- Your model score on test data is {testRegressionTree([tree], X_val, Y_val)}") 
              
