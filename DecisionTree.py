import random
import numpy as np
from math import log2, sqrt
from numbers import Number
import pickle
import pandas as pd

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
        max_features:  Maximum number of features considered at each split 
"""
class DecisionTreeRegressor():

    def __init__(self, max_depth=-1, min_leaf_samples=6, max_features="auto"):
        self.root_node = None
        self.max_depth = max_depth
        self.min_leaf_samples = min_leaf_samples

        if max_features in ["auto", "sqrt", "log2"] or isinstance(max_features,
                                                                  int) or max_features is None:
            self.max_features = max_features
            self.selected_features = None
        else:
            raise ValueError("Argument max_features must be 'auto', 'sqrt', 'log2', an int or None")

    """
        Trains the algorithm using the given dataset. feature_space:    Features used to train the tree.
        Array-like object of dimensions (n_examples, n_features)
    """
    def fit(self, feature_space, targets):
        # print("Sasas")
        if len(feature_space) < 1:
            raise ValueError("Not enough samples in the given dataset")
        self.getMaxFeaturesAtSplit(feature_space[0])
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
            Sets the number of considered features at each split depending on the
            max_features parameter.row: A single row of the features of shape (n_features)
    """
    def getMaxFeaturesAtSplit(self, row):
        if isinstance(self.max_features, int):
            self.selected_features = self.max_features if \
                self.max_features <= len(row) else len(row)
        elif isinstance(self.max_features, str):
            if self.max_features in ['auto', 'sqrt']:
                self.selected_features = int(sqrt(len(row)))
            elif self.max_features == 'log2':
                self.selected_features = int(log2(len(row)))
        else:
            self.selected_features = len(row)

    """
        Returns the randomly selected values in the given features.
        row: One-dimensional array of features
    """
    def get_features_subset(self, row):
        return [row[i] for i in self.features_indexes]

    """
            Calculate the variance in the given list.
    """
    def variance(self, targets):
        if len(targets) == 0:
            return None

        mean = np.average(targets)
        variance = sum([(x - mean) ** 2 for x in targets])
        return variance

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
            print("else block of propogation")
            v = observation[tree.col]
            branch = None
            if isinstance(v, Number):
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

        print(feature_datatypes)


        #print(features_selected)  # to be removed
        for column in range(len(feature_datatypes)):
            feature_values = [feature[column] for feature in features]
            if feature_datatypes[column] == "number":
                for feature_value in feature_values:
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
    train_df = pd.read_pickle('train.pkl')
    train_x = train_df.drop('trip_duration', 1)
    # train_y = train_df['trip_duration']
    train_y = np.log1p(train_df.trip_duration)
    num_arr = train_x.values
    num_arr_y = train_y
    num_arr_y = np.array(num_arr_y[:5000])
    dt = DecisionTreeRegressor(max_depth=5)
    print("decision tree starerted")
    dt.fit(num_arr[0:5000, :], num_arr_y.reshape(len(num_arr_y), 1))
    x = dt.predict(num_arr[5001:5100])
    print(x)
    rmlse = np.sqrt(np.average(np.square(np.subtract(np.array(train_y[5001:5100]), np.array(x).T))))
    print(rmlse)
    # rmlse = dt.getlrmse(np.array(train_y[1001:1050]), np.array(x).T)
    y = train_y.values
    print(train_y[10001])
    # if True:
    #     #newfile = sys.argv[1]
    #     # load the data file and do the preprocessing
    #     num_arr = pd.read_csv("JohnsonJohnson.csv")
    # num_arr = num_arr.values
    # dt = DecisionTreeRegressor(max_depth=4)
    # dt.fit(num_arr[:,:-1], num_arr[:,-1].reshape(len(num_arr),1))
    # x = dt.predict(num_arr[:,:-1])
    # print(x)
