"""
Coded by Alexander Rodriguez
"""

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
# hyper-parameters for SGD
LEARNING_RATES = np.array([0.001, 0.0001, 0.00001, 0.000001])
L_RATIOS = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
K = 5  # number of iterations
mod_enet = enet.ElasticNet()
# init matrix to save average the mse
mse = np.empty([4, 5, K])

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
            mod_enet.fit_SGD(X_train_2, y_train_2)
            e_v = mod_enet.rmse(X_validation, y_validation)
            mse[row, col, i] = e_v
            col += 1
        row += 1
# the mean of mse
mse = np.load("mse_sgd.npy")
avg_mse = np.mean(mse, axis=2)  # min= 0.001, 0.2

X, Y = np.meshgrid(L_RATIOS, LEARNING_RATES)
# contour plot
plt.pcolor(X, np.log10(Y), avg_mse)
plt.xlabel("l ratio")
plt.ylabel("log10 of learning rates")
plt.xticks(L_RATIOS)
plt.yticks(np.log10(LEARNING_RATES))
plt.colorbar()
plt.title('Average RMSLE for Elastic Net with SDG')


# hyper-parameters for SA
SA_RATES = np.array([0.0001, 0.00001, 0.000001])  # for neighbor generation
BATCH_SIZE = np.array([50, 250, 500, 750])
K = 5  # number of iterations
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
    for sa_rate in SA_RATES:
        col = 0
        for batch_size in BATCH_SIZE:
            mod_enet.set_param(sa_rate=sa_rate, batch_size=batch_size)
            mod_enet.fit_SGD(X_train_2, y_train_2)
            e_v = mod_enet.rmse(X_validation, y_validation)
            mse[row, col, i] = e_v
            col += 1
        row += 1
# the mean of mse
mse = np.load("mse_simulated_annealing.npy")
avg_mse = np.mean(mse, axis=2)


X, Y = np.meshgrid(BATCH_SIZE, SA_RATES)
# contour plot
plt.pcolor(X, np.log10(Y), avg_mse)
plt.xlabel("batch size")
plt.ylabel("log10 of neighbor generation std")
plt.xticks(BATCH_SIZE)
plt.yticks(np.log10(SA_RATES))
plt.colorbar()
plt.title('Average RMSLE for Elastic Net with SA')

"""
    Predicting in test dataset with tuned parameters
"""
# for SGD
errors = np.zeros(30)
mod_enet = enet.ElasticNet()
mod_enet.set_param(learning_rate=0.00001, l1_ratio=0.4)
for j in range(30):
    mod_enet.fit_SGD(X_train, y_train, seed=j)
    mod_enet.get_coef()
    errors[j] = mod_enet.rmse(X_test, y_test)

# for SA
errors = np.zeros(30)
mod_enet = enet.ElasticNet()
mod_enet.set_param(sa_rate=0.00001, batch_size=500)
for j in range(30):
    mod_enet.fit_SA(X_train, y_train, seed=j)
    mod_enet.get_coef()
    errors[j] = mod_enet.rmse(X_test, y_test)


"""
    Tune scikit-learn regressor
"""

# hyper-parameters for SGD
LEARNING_RATES = np.array([0.001, 0.0001, 0.00001, 0.000001])
L_RATIOS = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
K = 5  # number of iterations
mod_enet = mod_enet_scikit = \
    lm.SGDRegressor(penalty='elasticnet', learning_rate="constant", tol=0.0001)
# init matrix to save average the mse
mse = np.empty([4, 5, K])

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
            mod_enet_scikit.set_params(eta0=learning_rate, l1_ratio=l_ratio)
            mod_enet_scikit.fit(X_train_2, y_train_2)
            y_pred_scikit = mod_enet_scikit.predict(X_test)
            squared_errors = np.power(y_pred_scikit - y_test, 2)
            e_v = pow(sum(squared_errors) / len(squared_errors), 0.5)
            mse[row, col, i] = e_v
            col += 1
        row += 1
# the mean of mse
mse = np.load("mse_scikit-learn.npy")
avg_mse = np.mean(mse, axis=2)  # min= 0.001, 0.2

X, Y = np.meshgrid(L_RATIOS, LEARNING_RATES)
# contour plot
plt.pcolor(X, np.log10(Y), avg_mse)
plt.xlabel("l ratio")
plt.ylabel("log10 of learning rates")
plt.xticks(L_RATIOS)
plt.yticks(np.log10(LEARNING_RATES))
plt.colorbar()
plt.title('Average RMSLE for Elastic Net with SDG')

"""
    Predict with scikit-learn regressor
"""


# modeling with scikit-learn elastic nets
mod_enet_scikit = lm.SGDRegressor(penalty='elasticnet', learning_rate="constant", tol=0.0001,
                                  eta0=0.00001, l1_ratio=0.6)  # learning_rate='constant'
mod_enet_scikit.fit(X_train, y_train)
y_pred_scikit = mod_enet_scikit.predict(X_test)
squared_errors = np.power(y_pred_scikit - y_test, 2)
pow(sum(squared_errors)/len(squared_errors), 0.5)
# coefficient
mod_enet_scikit.coef_

squared_errors = np.power(y_pred_scikit - y_test, 2)
pow(sum(squared_errors)/len(squared_errors), 0.5)


errors = np.zeros(30)

for j in range(30):
    mod_enet_scikit.set_params(random_state=j)
    mod_enet_scikit.fit(X_train, y_train)
    y_pred_scikit = mod_enet_scikit.predict(X_test)
    squared_errors = np.power(y_pred_scikit - y_test, 2)
    errors[j] = pow(sum(squared_errors) / len(squared_errors), 0.5)



mod = lm.ElasticNet(l1_ratio=0.1)
mod.fit(X_train, y_train)

