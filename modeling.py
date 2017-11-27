
from sklearn import datasets
import enet
import numpy as np
from sklearn import linear_model as lm
import pandas as pd
import importlib

enet = importlib.reload(enet)

train_df = pd.read_pickle('data_train_preprocessed.pkl')

binary_df = pd.get_dummies(train_df)
# shuffle dataset
binary_df.sample(frac=1)
binary_df_train = binary_df.iloc[:101925, :]
binary_df_test = binary_df.iloc[101925:, :]
X_train = np.array(binary_df_train.drop(columns='trip_duration'))
y_train = np.log1p(np.array(binary_df_train['trip_duration']))
X_test = np.array(binary_df_test.drop(columns='trip_duration'))
y_test = np.log1p(np.array(binary_df_test['trip_duration']))


# modeling with our elastic nets
mod_enet = enet.elastic_net()
mod_enet.fit(X_train, y_train)
y_pred = mod_enet.predict(X_test)

squared_errors = np.power(y_pred - y_test, 2)
pow(sum(squared_errors)/len(squared_errors), 0.5)

# modeling with scikit-learn elastic nets
mod_enet_scikit = lm.ElasticNet()
mod_enet_scikit.fit(X_train, y_train)
y_pred_scikit = mod_enet_scikit.predict(X_test)

squared_errors = np.power(y_pred_scikit - y_test, 2)
pow(sum(squared_errors)/len(squared_errors), 0.5)


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
