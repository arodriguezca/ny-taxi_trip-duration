import random
import numpy as np
from math import log2, sqrt
from numbers import Number
import pickle
import pandas as pd
import matplotlib.pyplot as plt

"""
    Represents a node / leaf in a tree.
"""
class DecisionNode:

    def __init__(self, col=-1, value=None, result=None, left_branch=None, right_branch=None):
        self.col = col
        self.value = value
        self.result = result
        self.left_branch = left_branch
        self.right_branch = right_branch


"""
        max_depth: Maximum number of splits during training
        min_leaf_samples:  Minimum number of examples in a leaf node.
"""
class DecisionTreeRegressor():

    def __init__(self, max_depth=-1, min_leaf_samples=6):
        self.root_node = None
        self.max_depth = max_depth
        self.min_leaf_samples = min_leaf_samples

    """
        Trains the algorithm using the given dataset. feature_space:    Features used to train the tree.
        Array-like object of dimensions (n_examples, n_features)
    """
    def fit(self, feature_space, targets):
        # print("Sasas")
        if len(feature_space) < 1:
            raise ValueError("Not enough samples in the given dataset")

        feature_datatypes = ["category" if len(set(feature_space[:, i])) <= 12 else "number" for i in
                             range(len(feature_space[0]))]
        self.root_node = self.growTree(feature_space, targets, self.max_depth, feature_datatypes = feature_datatypes)

    """
        Predict a value for the given features.
    """
    def predict(self, features):
        res = []
        for i in range(features.shape[0]):
            res.append(self.propagate(features[i], self.root_node))
        return res  # self.propagate(features, self.root_node)

    """
            Calculate the variance in the given list.
    """
    def variance(self, targets):
        if len(targets) == 0:
            return None

        mean = np.average(targets)
        variance = sum([(x - mean) ** 2 for x in targets])
        return variance

    '''
        calculate the mean absolute error in the given list
    '''
    def mae(self, targets):
        if len(targets) == 0:
            return None

        mean = np.average(targets)
        mae = np.sum([np.abs(x - mean) for x in targets])
        return mae


    """
            Divide the given dataset depending on the value at the given column index.
    """
    def splitNumericCat(self, features, targets, column, feature_value, feature_datatypes):
        split_function = None
        if feature_datatypes[column] == "number":
            split_function = lambda row: row[column] >= feature_value
        else:
            split_function = lambda row: row[column] == feature_value

        set1 = [row for row in zip(features, targets) if split_function(row[0])]
        set2 = [row for row in zip(features, targets) if not split_function(row[0])]

        feat1, targs1 = [x[0] for x in set1], [x[1] for x in set1]
        feat2, targs2 = [x[0] for x in set2], [x[1] for x in set2]
        return feat1, targs1, feat2, targs2

    """
            Makes a prediction using the given features.
            observation: The features to use to predict
    """
    def propagate(self, observation, tree):
        resulta = []

        if tree.result is not None:
            # result.append(tree.result)
            return tree.result
        else:
            #print("else block of propogation")
            v = observation[tree.col]
            branch = None
            if isinstance(v, Number):
                #print(v, tree.value)
                if v >= tree.value:
                    branch = tree.left_branch
                else:
                    branch = tree.right_branch
            else:
                if tree.value in v:
                    branch = tree.left_branch
                else:
                    branch = tree.right_branch
            return self.propagate(observation, branch)



    """
            Recursively create the decision tree by splitting the dataset until there
            is no real reduce in variance, or there is less examples in a node than
            the minimum number of examples, or until the max depth is reached.
    """
    def getSorted(self, features, column, targets):
        features_sorted = []
        targets_sorted = []
        sorted_features_targets = sorted(zip(features, targets), key=lambda feature: feature[0][column])
        for pair in sorted_features_targets:
            features_sorted.append(pair[0])
            targets_sorted.append(pair[1])
        return features_sorted, targets_sorted

    def growTree(self, features, targets, depth, feature_datatypes):
        if len(features) == 0:
            return DecisionNode()
        if depth == 0:
            return DecisionNode(result=np.mean(targets))

        lowest_variance = None
        best_split_criteria = None
        best_sets = None
        #get feature data types
        #print(features.shape)
        #print(feature_datatypes)
        #print(features_selected)  # to be removed
        for column in range(len(feature_datatypes)):
            feature_values = [feature[column] for feature in features]
            if feature_datatypes[column] == "number":
                features_sorted, targets_sorted  = self.getSorted(features, column, targets)
                feature_values_sorted = [feature[column] for feature in features_sorted]
                #print(feature_values_sorted)
                if len(set(feature_values_sorted)) <= 200:
                    for set_value in  list(set(feature_values_sorted)):
                        feature_value = set_value
                        index = feature_values_sorted.index(set_value)
                        feats1 = features_sorted[index:]
                        targs1 = targets_sorted[index:]
                        feats2 = features_sorted[0:index - 1]
                        targs2 = targets_sorted[0:index - 1]
                        var1 = self.variance(targs1)
                        var2 = self.variance(targs2)
                        #print("coninuous cat")
                        if var1 is None or var2 is None:
                            continue
                        variance = var1 + var2
                        if (lowest_variance is None) or np.all(variance < lowest_variance):
                            lowest_variance = variance
                            best_split_criteria = (column, feature_value)
                            # print(best_split_criteria)
                            best_sets = ((feats1, targs1), (feats2, targs2))

                else:
                    for index, feature_value  in enumerate(feature_values_sorted):
                            #feats1, targs1, feats2, targs2 = self.splitNumericCat(features, targets, column, feature_value, feature_datatypes, index)
                        feats1 = features_sorted[index:]
                        targs1 = targets_sorted[index:]
                        feats2 = features_sorted[0:index-1]
                        targs2 = targets_sorted[0:index-1]
                        var1 = self.variance(targs1)
                        var2 = self.variance(targs2)
                        if var1 is None or var2 is None:
                            continue
                        variance = var1 + var2
                        if (lowest_variance is None) or np.all(variance < lowest_variance):
                            lowest_variance = variance
                            best_split_criteria = (column, feature_value)
                                #print(best_split_criteria)
                            best_sets = ((feats1, targs1), (feats2, targs2))
            else:
                for feature_value in set(feature_values):
                    feats1, targs1, feats2, targs2 = self.splitNumericCat(features, targets, column, feature_value, feature_datatypes)
                    var1 = self.variance(targs1)
                    var2 = self.variance(targs2)
                    if var1 is None or var2 is None:
                        continue
                    variance = var1 + var2
                    if (lowest_variance is None) or np.all(variance < lowest_variance):
                        lowest_variance = variance
                        best_split_criteria = (column, feature_value)
                        best_sets = ((feats1, targs1), (feats2, targs2))


                    # Check variance value also
        if lowest_variance is not None and \
                        len(best_sets[0][0]) >= self.min_leaf_samples and \
                        len(best_sets[1][0]) >= self.min_leaf_samples:
            #print(best_sets[0][0])
            left_branch = self.growTree(best_sets[0][0], best_sets[0][1], depth - 1, feature_datatypes = feature_datatypes)
            right_branch = self.growTree(best_sets[1][0], best_sets[1][1], depth - 1, feature_datatypes = feature_datatypes)
            return DecisionNode(col=best_split_criteria[0], value=best_split_criteria[1],
                                left_branch=left_branch, right_branch=right_branch)
        else:
            return DecisionNode(result=np.mean(targets))

    def rmlse(self, predicted, target):
        np.sqrt(np.average(np.square(np.subtract(target, np.array(predicted)))))
        return None


if __name__ == "__main__":

    train_df = pd.read_pickle('data_train_preprocessed.pkl')

    binary_df = pd.get_dummies(train_df)
    # shuffle dataset and then subset it (30% for test)
    binary_df.sample(frac=1, random_state=5)  # shuffle
    binary_df_train = binary_df.iloc[:101925, :]
    binary_df_test = binary_df.iloc[101925:, :]
    X_train = np.array(binary_df_train.drop(columns='trip_duration'))
    y_train = np.log1p(np.array(binary_df_train['trip_duration']))
    X_test = np.array(binary_df_test.drop(columns='trip_duration'))
    y_test = np.log1p(np.array(binary_df_test['trip_duration']))

    # hyper-parameters for DT
    MAX_DEPTHS = [2, 4, 6]
    MIN_SAMPLES_IN_LEAF = [5, 10, 15]
    K = 5  # number of iterations

    # init matrix to save average the mse
    mse = np.empty([3, 3, K])

    for i in range(K):
        # shuffle rows
        np.random.seed(i + 1)
        shuffle_indexes = np.random.permutation(len(y_train))
        X_train = X_train[shuffle_indexes, :]
        y_train = y_train[shuffle_indexes]
        # split train in two: train and validation (30% for validation)
        # the dataset is already shuffled
        X_train_2 = X_train[:71347, :]
        y_train_2 = y_train[:71347]
        X_validation = X_train[71347:, :]
        y_validation = y_train[71347:]
        row = 0
        for md in MAX_DEPTHS:
            col = 0
            for msl in MIN_SAMPLES_IN_LEAF:
                print("maximum depth is {} and min leaf samples i {}".format(md, msl))
                dt = DecisionTreeRegressor(max_depth=md, min_leaf_samples=msl)
                dt.fit(X_train_2, y_train_2)
                y_pred = dt.predict(X_validation)
                squared_errors = np.power(y_pred - y_validation, 2)
                e_v = pow(sum(squared_errors) / len(squared_errors), 0.5)
                mse[row, col, i] = e_v
                col += 1
            row += 1

        # the mean of mse
        mse = np.load("mse_dt.npy")
        avg_mse = np.mean(mse, axis=2)

        X, Y = np.meshgrid(MIN_SAMPLES_IN_LEAF, MAX_DEPTHS)
        # contour plot
        plt.pcolor(X, Y, avg_mse)
        plt.xlabel("Min samples in leaf")
        plt.ylabel("Maximum depth")
        plt.xticks(MIN_SAMPLES_IN_LEAF)
        plt.yticks(MAX_DEPTHS)
        plt.colorbar()
        plt.title('Average RMSLE for DT with MAE split criteria')

        # predict in test data with tuned parameters
        dt = DecisionTreeRegressor(max_depth=4, min_leaf_samples=5)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        squared_errors = np.power(y_pred - y_test, 2)
        e_v = pow(sum(squared_errors) / len(squared_errors), 0.5)