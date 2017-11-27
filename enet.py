"""Elastic net regressor with stochastic gradient descend

"""
import numpy as np
import pandas as pd


class elastic_net:
    # alpha =  1
    # l1_ratio = 0.5
    # tol = 0.0001
    # max_iter = 1000

    def __init__(self, user_alpha=1, user_l1_ratio=0.5, user_tol=0.001,
                 user_max_iter=1000, user_learning_rate=0.0001):
        self.alpha = user_alpha
        self.l1_ratio = user_l1_ratio
        self.tol = user_tol
        self.max_iter = user_max_iter
        self.learning_rate = user_learning_rate
        self.betas = np.random.random(3)

    def _rmse(self, X, y):

        squared_errors = np.power(np.matmul(X, self.betas) - y, 2)
        return pow(sum(squared_errors)/len(squared_errors), 0.5)

    def _grad(self, x_i, y_i):
        """ Gradient for stochastic gradient descend

        :param x_i: numpy array
        :param y_i: number
        :return:
        """
        squared_loss_derivate = -1 * (y_i - np.vdot(x_i, self.betas)) * x_i
        l1_loss_derivate = self.alpha * self.l1_ratio * np.sign(self.betas)
        l2_loss_derivate = self.alpha * (1 - self.l1_ratio) * self.betas
        return squared_loss_derivate + l1_loss_derivate + l2_loss_derivate

    def _update_betas_SGD(self, X, y):
        for i in range(len(y)):  # loop through all observation
            gradient = self._grad(X[i, :], y[i])
            self.betas -= self.learning_rate * gradient
            # for j in np.arange(0, len(self.betas)):
            #     self.betas[j] -= self.learning_rate * my_err * inputs[e, j])
        print("Betas: ", self.betas)
        return gradient

    def fit(self, X, y):
        # add a column of ones
        X = np.c_[np.ones(len(y)), X]
        # init coefficients
        self.betas = np.random.random(X.shape[1])
        prev_mean_loss = 0
        for i in range(0, self.max_iter):
            # shuffle rows
            np.random.shuffle(X)
            gradient = self._update_betas_SGD(X, y)
            # compute loss:
            mean_loss = self._rmse(X, y)
            # break when loss change is insignificant
            if np.sum(gradient) < self.tol:  # abs(mean_loss - prev_mean_loss) < self.tol:
                print("Break in iteration ", i)
                print("Last loss: ", mean_loss)
                break
            # save current loss as previous loss
            prev_mean_loss = mean_loss
            if i == 0:
                print("First loss: ", mean_loss)
            elif i == self.max_iter - 1:
                print("Reached max iterations")
                print("Last loss: ", mean_loss)

    def predict(self, X):
        return np.matmul(np.c_[np.ones(X.shape[0]), X], self.betas)
