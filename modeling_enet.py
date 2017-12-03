
import ElasticNet as enet
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
import pandas as pd
import importlib

enet = importlib.reload(enet)

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

"""
    Tuning our model
"""
# hyper-parameters
LEARNING_RATES = np.array([0.001, 0.0001, 0.00001, 0.000001, 0.0000001])
L_RATIOS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
K = 10  # number of iterations
mod_enet = enet.ElasticNet()
# init matrix to save average the mse
mse = np.empty([5, 10, K])

for i in range(K):
    # shuffle rows
    np.random.seed(i+1)
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
    for learning_rate in LEARNING_RATES:
        col = 0
        for l_ratio in L_RATIOS:
            mod_enet.set_param(learning_rate=learning_rate, l1_ratio=l_ratio)
            mod_enet.fit(X_train_2, y_train_2)
            e_v = mod_enet.rmse(X_validation, y_validation)
            mse[row, col, i] = e_v
            col += 1
        row += 1
# the mean of mse
avg_mse = np.mean(mse, axis=2)

X, Y = np.meshgrid(L_RATIOS, LEARNING_RATES)
# contour plot
plt.pcolor(X, Y, avg_mse)
plt.xlabel("l ratio")
plt.ylabel("alpha")
plt.colorbar()
plt.title('Average MSE for hyperparameter grid')

"""
    Predicting in test dataset with tuned parameters
"""
errors = np.zeros(30)
mod_enet.set_param()
for j in range(30):
    mod_enet.fit(X_train, y_train, seed=j)
    mod_enet.get_coef()
    errors[j] = mod_enet.rmse(X_test, y_test)

"""
    Comparison with scikit-learn
"""
# modeling with scikit-learn elastic nets
mod_enet_scikit = lm.SGDRegressor(penalty='elasticnet', l1_ratio=0.5, tol=0.0001)  # learning_rate='constant'
mod_enet_scikit.fit(X_train, y_train)
y_pred_scikit = mod_enet_scikit.predict(X_test)
# coefficient
mod_enet_scikit.coef_

squared_errors = np.power(y_pred_scikit - y_test, 2)
pow(sum(squared_errors)/len(squared_errors), 0.5)


# from sklearn import datasets

# iris = datasets.load_iris()

# mod_enet.fit(iris.data[:,0:2], iris.data[:,3])
#
# mod2 = lm.SGDRegressor()
# mod2.fit(iris.data[:,0:2], iris.data[:,3])
# mod2.coef_
# mod2.get_params()
# mod_enet.mse(np.c_[np.ones(150), iris.data[:,0:2]], iris.data[:,3])

# old
# mod_enet.loss( , iris.data[0,3])
