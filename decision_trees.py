import math
import random
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def decisionTreeRegressor():
    regressor = DecisionTreeRegressor(max_depth=5, random_state=42)
    regressor.fit(X_train, Y_train)
    y_pred = regressor.predict(X_val)
    y_train_pred = regressor.predict(X_train)
    mse_train = mean_squared_error(Y_train, y_train_pred)
    print(f"Mean squared error on training {mse_train}")
    mse = mean_squared_error(Y_val, y_pred)
    print(f"Mean Squared Error: {mse}")
    plt.figure(figsize=(20, 10))
    tree.plot_tree(regressor, filled=True)
    tree = regressor.tree_
    for node in range(tree.node_count):
        if tree.children_left[node] == tree.children_right[node]:
            print(f"Leaf Node {node}: {tree.n_node_samples[node]} samples")


class DTNode:
    splitFeatureIdx = None
    splitFeatureType = None
    exp = None
    lchild = None
    rchild = None
    indices = None


def getDistribution(indices, Y_train, weights):
    d = dict()
    expected = None
    maxWeight = 0
    for idx in indices:
        if Y_train[idx] in d:
            d[Y_train[idx]] += weights[idx]
        else:
            d[Y_train[idx]] = weights[idx]
        if maxWeight < d[Y_train[idx]]:
            maxWeight = d[Y_train[idx]]
            expected = Y_train[idx]

    return d, expected


def getInformation(distribution, sumOfWeights):
    information = 0
    for key in distribution:
        p = distribution[key] / sumOfWeights
        information += p * math.log2(p)
    return -information


def decisionTree(
    X_train, Y_train, feature_indices, weights, discrete_features, max_depth=6
):
    distribution, expected = getDistribution(range(0, len(X_train)), Y_train, weights)
    information = getInformation(distribution, 1)
    return makeDecisionTree(
        X_train,
        Y_train,
        set(range(len(X_train))),
        information,
        expected,
        1,
        feature_indices,
        max_depth,
        weights,
        discrete_features,
    )


def sumOfWeights(indices, weights):
    weightSum = 0
    for index in indices:
        weightSum += weights[index]
    return weightSum


def makeDecisionTree(
    X_train,
    Y_train,
    indices,
    information,
    exp,
    depth,
    feature_indices,
    max_depth,
    weights,
    discrete_features,
):
    if information == 0 or depth >= max_depth or len(indices) <= min_samples_in_node:
        node = DTNode()
        node.splitFeatureIdx = -1
        node.exp = exp
        node.indices = indices
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
            data_points.append((X_train[i][featureIdx], i))
        data_points.sort()
        isDiscreteFeature = feature_indices[featureIdx] in discrete_features
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
            type_indices_distribution, type_exp = getDistribution(
                type_indices, Y_train, weights
            )
            non_type_indices_distribution, non_type_exp = getDistribution(
                non_type_indices, Y_train, weights
            )
            type_information = getInformation(
                type_indices_distribution, sumOfWeights(type_indices, weights)
            )
            non_type_information = getInformation(
                non_type_indices_distribution, sumOfWeights(non_type_indices, weights)
            )
            information_gain = information - (
                type_information * len(type_indices)
                + non_type_information * len(non_type_indices)
            ) / len(indices)

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
        node.indices = indices
        return node

    node = DTNode()
    node.lchild = makeDecisionTree(
        X_train,
        Y_train,
        left_indices,
        left_information,
        left_exp,
        depth + 1,
        feature_indices,
        max_depth,
        weights,
        discrete_features,
    )
    node.rchild = makeDecisionTree(
        X_train,
        Y_train,
        right_indices,
        right_information,
        right_exp,
        depth + 1,
        feature_indices,
        max_depth,
        weights,
        discrete_features,
    )
    node.splitFeatureIdx = feature_indices[split_feature_idx]
    node.splitFeatureType = split_feature_type
    return node


def regressionTree(X_train, Y_train, feature_indices, discrete_features, max_depth=6):
    variance = getVariance(Y_train, list(range(0, len(Y_train))))
    expected = getExpectation(Y_train, list(range(0, len(Y_train))))
    return makeRegressionTree(
        X_train,
        Y_train,
        set(range(len(X_train))),
        variance,
        expected,
        1,
        feature_indices,
        max_depth,
        discrete_features,
    )


def getVariance(Y, indices):
    if len(indices) == 0:
        return 0
    mean = getExpectation(Y, indices)
    variance = 0
    for i in indices:
        variance = variance + (mean - Y[i]) ** 2
    return variance / len(indices)


def getExpectation(Y, indices):
    if len(indices) == 0:
        return 0
    mean = 0
    for i in indices:
        mean = mean + Y[i]
    return mean / len(indices)


def gradient_boost(X_train, y_train, discrete_features, max_depth):
    dist, exp = getDistribution(
        list(range(0, len(y_train))), y_train, [1 / len(y_train)] * len(y_train)
    )
    classes = []
    prob = []
    for c, p in dist.items():
        classes.append(c)
        prob.append(p)

    classIndices = dict()
    for i in range(0, len(classes)):
        classIndices[classes[i]] = i

    probabilities = do_gradient_boost(
        X_train,
        y_train,
        [prob] * len(X_train),
        classes,
        max_depth,
        1,
        discrete_features,
    )
    loss = 0
    for i in range(0, len(X_train)):
        loss = loss - math.log(probabilities[i][classIndices[y_train[i]]])
    score = 0
    for i in range(0, len(X_train)):
        max_prob = 0
        c = None
        for k in range(0, len(probabilities[i])):
            if max_prob < probabilities[i][k]:
                max_prob = probabilities[i][k]
                c = classes[k]
        if c == y_train[i]:
            score += 1

    print(f"score = {score/len(X_train)*100}")


def do_gradient_boost(
    X_train, y_train, probabilities, classes, max_depth, curr_depth, discrete_features
):
    if curr_depth == max_depth:
        return probabilities
    total_classes = len(probabilities[0])
    m = len(X_train)
    gammas = []
    for c in range(0, total_classes):
        residuals = [None] * m
        for i in range(0, m):
            if y_train[i] == classes[c]:
                residuals[i] = 1 - probabilities[i][c]
            else:
                residuals[i] = -probabilities[i][c]
        gammas.append(
            getGamma(
                X_train, residuals, [p[c] for p in probabilities], discrete_features
            )
        )
    new_prob = [None] * m

    for i in range(0, m):
        new_prob[i] = updateProbability(probabilities[i], [g[i] for g in gammas])
    return do_gradient_boost(
        X_train,
        y_train,
        new_prob,
        classes,
        max_depth,
        curr_depth + 1,
        discrete_features,
    )


def getGamma(X_train, residuals, prob, discrete_features):
    root = regressionTree(
        X_train, residuals, list(range(0, len(X_train[0]))), discrete_features, 2
    )
    gamma = [None] * len(X_train)
    dfs(root, prob, gamma)
    return gamma


def updateProbability(prob, gamma):
    k = len(prob)
    new_prob = [None] * k
    exp_logit_updated = [None] * k
    sum = 0
    for c in range(0, k):
        if prob[c] == 0.0:
            prob[c] = 1e-10
        if prob[c] == 1.0:
            prob[c] = 1.0 - 1e-10
        logit = math.log(prob[c] / (1 - prob[c]))
        exp_logit_updated[c] = math.exp(logit + gamma[c])
        sum += exp_logit_updated[c]
    for c in range(0, k):
        new_prob[c] = exp_logit_updated[c] / sum
    return new_prob


def dfs(root: DTNode, prob: list, gamma: list):
    if root.splitFeatureIdx == -1:
        hessian = 0
        for idx in root.indices:
            hessian += prob[idx] * (1 - prob[idx])

        for idx in root.indices:
            if hessian == 0:
                gamma[idx] = 0
            else:
                gamma[idx] = root.exp * len(root.indices) / hessian
        return
    dfs(root.lchild, prob, gamma)
    dfs(root.rchild, prob, gamma)


def makeRegressionTree(
    X_train,
    Y_train,
    indices,
    variance,
    exp,
    depth,
    feature_indices,
    max_depth,
    discrete_features,
):
    if variance == 0 or depth >= max_depth or len(indices) <= min_samples_in_node:
        node = DTNode()
        node.splitFeatureIdx = -1
        node.exp = exp
        node.indices = indices
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
            data_points.append((X_train[i][featureIdx], i))
        data_points.sort()
        isDiscreteFeature = feature_indices[featureIdx] in discrete_features
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
            variance_reduction = variance - (
                len(type_indices) * type_variance
                + len(non_type_indices) * non_type_variance
            ) / (len(type_indices) + len(non_type_indices))

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
        node.indices = indices
        node.exp = exp
        return node

    node = DTNode()
    node.lchild = makeRegressionTree(
        X_train,
        Y_train,
        left_indices,
        left_variance,
        left_exp,
        depth + 1,
        feature_indices,
        max_depth,
        discrete_features,
    )
    node.rchild = makeRegressionTree(
        X_train,
        Y_train,
        right_indices,
        right_variance,
        right_exp,
        depth + 1,
        feature_indices,
        max_depth,
        discrete_features,
    )
    node.splitFeatureIdx = feature_indices[split_feature_idx]
    node.splitFeatureType = split_feature_type
    node.indices = indices
    return node


def evaluate(root: DTNode, x):
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
    return (score / total_score) * 100


def testRegressionTree(trees, X_test, Y_test):
    B = len(trees)
    err = 0
    y_pred = []
    for i in range(0, len(X_test)):
        avg_prediction = 0
        for tree in range(0, B):
            prediction = evaluate(trees[tree], X_test[i])
            avg_prediction += prediction
        avg_prediction /= B
        y_pred.append(avg_prediction)

    return mean_squared_error(Y_test, y_pred)


def sampling_with_replacement(X, Y, isClassification, discrete_features, B=150):
    no_of_samples_for_each_tree = int(len(X) * 0.8)
    trees = []
    total_features = len(X[0])
    for b in range(0, B):
        X_train = []
        Y_train = []
        for sample in range(0, no_of_samples_for_each_tree):
            training_example = random.randint(0, len(X) - 1)
            X_train.append(X[training_example])
            Y_train.append(Y[training_example])
        if isClassification:
            trees.append(
                decisionTree(
                    X_train,
                    Y_train,
                    list(range(total_features)),
                    [1 / no_of_samples_for_each_tree] * no_of_samples_for_each_tree,
                    discrete_features,
                )
            )
        else:
            trees.append(
                regressionTree(
                    X_train, Y_train, list(range(total_features)), discrete_features
                )
            )

    return trees


def randomForest(X, Y, isClassification, discrete_features, B=150):
    total_features = len(X[0])
    k = int(math.sqrt(total_features))
    no_of_samples_for_each_tree = int(len(X) * 0.8)
    trees = []
    for b in range(0, B):
        X_train = []
        Y_train = []
        selected_features = random.sample(list(range(total_features)), k)
        for sample in range(0, no_of_samples_for_each_tree):
            training_example = random.randint(0, len(X) - 1)
            row = []
            for feature in selected_features:
                row.append(X[training_example][feature])
            X_train.append(row)
            Y_train.append(Y[training_example])
        if isClassification:
            trees.append(
                decisionTree(
                    X_train,
                    Y_train,
                    list(range(total_features)),
                    [1 / no_of_samples_for_each_tree] * no_of_samples_for_each_tree,
                )
            )
        else:
            trees.append(
                regressionTree(
                    X_train, Y_train, list(range(total_features)), discrete_features
                )
            )
    return trees


def adaBoost(X, Y, count, weights, discrete_features):
    if count == 0:
        return []
    root = decisionTree(X, Y, list(range(0, len(X[0]))), weights, discrete_features, 2)
    total_samples = len(X)
    error_samples = set()
    for i in range(0, total_samples):
        if evaluate(root, X[i]) != Y[i]:
            error_samples.add(i)
    incorrect_weight = math.sqrt(
        (total_samples - len(error_samples)) / (len(error_samples))
    )
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
    return (score / totalScore) * 100


min_samples_in_node = 20
df = pd.read_csv("/mnt/92026878026862E9/python_files/drug200.csv", encoding="latin-1")

categorical_cols = df.select_dtypes(include=["object", "category"]).columns
le = LabelEncoder()
discrete_features = set()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
    discrete_features.add(col)

X = df.iloc[:, :-1].values.tolist()
Y = df.iloc[:, -1].values.tolist()

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)


def sampling_with_replacements():
    sampling_with_replacement_trees = sampling_with_replacement(
        X_train, Y_train, True, discrete_features
    )
    print(
        f"---Sampling with replacement-- Your model score on training data is {test(sampling_with_replacement_trees, X_train, Y_train)}"
    )
    print(
        f"---Sampling with replacement-- Your model score on test data is {test(sampling_with_replacement_trees, X_val, Y_val)}"
    )


def randomForests():
    random_forest_trees = randomForest(X_train, Y_train, False, discrete_features, 128)
    print(
        f"---Random Forest-- Your model score on training data is {testRegressionTree(random_forest_trees, X_train, Y_train)}"
    )
    print(
        f"---Random Forest-- Your model score on test data is {testRegressionTree(random_forest_trees, X_val, Y_val)}"
    )


def adaBoosting():
    trees = adaBoost(
        X_train, Y_train, 128, [1 / len(X_train)] * len(X_train), discrete_features
    )
    print(
        f"---AdaBoost -- Your model score on training data is {testAdaBoost(trees, X_train, Y_train)}"
    )
    print(
        f"---AdaBoost -- Your model score on test data is {testAdaBoost(trees, X_val, Y_val)}"
    )


def decisionTrees():
    tree = decisionTree(
        X_train,
        Y_train,
        list(range(0, len(X_train[0]))),
        [1 / len(X_train)] * len(X_train),
        discrete_features,
        6,
    )
    print(
        f"---DT-- Your model score on training data is {test([tree], X_train, Y_train)}"
    )
    print(f"---DT-- Your model score on test data is {test([tree], X_val, Y_val)}")


def regressionTrees():
    tree = regressionTree(
        X_train, Y_train, list(range(0, len(X_train[0]))), discrete_features, 6
    )
    print(
        f"---DT-- Your model score on training data is {testRegressionTree([tree], X_train, Y_train)}"
    )
    print(
        f"---DT-- Your model score on test data is {testRegressionTree([tree], X_val, Y_val)}"
    )


# regressionTrees()
# decisionTrees()
# sampling_with_replacements()
gradient_boost(X_train, Y_train, discrete_features, 100)
