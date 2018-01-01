"""
Elastic net regressor with stochastic gradient descend
Coded by Alexander Rodriguez
"""
import numpy as np
import pandas as pd
import math
import random

class ElasticNet:

    def __init__(self, alpha=0.0001, l1_ratio=0.15, tol=0.001,
                 max_iter=50, learning_rate=0.0001,
                 sa_rate=0.5, batch_size=500):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.tol = tol
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.betas = np.random.random(3)
        self.sa_rate = sa_rate  # rate of neighbor generation for SA
        self.batch_size = batch_size  # number of iterations at a given temperature

    def _rmse(self, X, y):
        """ root mean squared error

        :param X: feature matrix
        :param y: predictant
        :return: rmse
        """
        squared_errors = np.power(np.matmul(X, self.betas) - y, 2)
        return math.sqrt(sum(squared_errors)/len(squared_errors))

    def _grad(self, x_i, y_i):
        """ Gradient for stochastic gradient descend

        :param x_i: numpy array
        :param y_i: number
        :return:
        """
        squared_loss_derivate = (y_i - np.vdot(x_i, self.betas)) * x_i
        l1_loss_derivate = self.alpha * self.l1_ratio * np.sign(self.betas)
        l2_loss_derivate = self.alpha * (1 - self.l1_ratio) * self.betas
        return squared_loss_derivate - l1_loss_derivate - l2_loss_derivate

    def _update_betas_SGD(self, X, y):
        for i in range(len(y)):  # loop through all observation
            gradient= self._grad(X[i, :], y[i])
            self.betas += self.learning_rate * gradient

    def fit_SGD(self, X, y, seed=17):
        """Fit with stochastic gradient descend

        :param X: 2D numpy array
        :param y: numpy array
        :param seed: seed number
        """
        # add a column of ones
        X = np.c_[np.ones(len(y)), X]
        # init coefficients
        self.betas = np.random.random(X.shape[1])
        previous_loss = 100
        for i in range(0, self.max_iter):
            # shuffle rows
            np.random.seed(seed*(i+1))
            shuffle_indexes = np.random.permutation(len(y))
            X = X[shuffle_indexes, :]
            y = y[shuffle_indexes]
            # update coefficients
            self._update_betas_SGD(X, y)
            # compute loss:
            loss = self._rmse(X, y)
            # break when loss change is insignificant
            if loss > previous_loss - self.tol:  # abs(mean_loss - prev_mean_loss) < self.tol:
                print("Break in iteration ", i)
                self.betas = prev_betas
                print("Last loss: ", previous_loss)
                break
            elif i == self.max_iter - 1:
                print("Reached max iterations")
                print("Last loss: ", loss)
            else:
                print("Loss per iteration: ", i, "-", loss)
            previous_loss = loss
            prev_betas = self.betas.copy()

    def _evaluate_loss(self, X, y, betas):
        """ Evaluate

        :param x_i: numpy array
        :param y_i: number
        :return: loss
        """
        squared_loss = np.power(np.matmul(X, betas) - y, 2)
        l1_loss = self.alpha * self.l1_ratio * np.repeat(np.linalg.norm(betas, 1), len(y))
        l2_loss = self.alpha * (1 - self.l1_ratio) * np.repeat(np.linalg.norm(betas, 2), len(y))
        return np.mean(squared_loss + l1_loss + l2_loss)

    def _pick_neighbor(self):
        """ Evaluate

        :param x_i: numpy array
        :return: one neighbor
        """
        variability = np.random.normal(0, self.sa_rate, len(self.betas))
        indexes = np.random.choice(np.arange(len(self.betas)), size=math.floor(4*len(self.betas)/5))
        variability[indexes] = 0
        return self.betas + variability

    def fit_SA(self, X, y, seed=17):
        """Fit with simulated annealing

        :param X: 2D numpy array
        :param y: numpy array
        :param seed: seed number
        """
        # add a column of ones
        X = np.c_[np.ones(len(y)), X]
        # init coefficients
        self.betas = np.random.random(X.shape[1])
        current_loss = 100
        # number of iterations
        n_iter = 0
        tk = 2000
        iter = 2000
        alpha_SA = 0.95  # cooling ratio
        TEMPERATURE_LENGTH = 50
        for i in range(0, iter):
            m = 0
            while m < TEMPERATURE_LENGTH:
                np.random.seed(seed * (i + 1) + m)
                # batch: get some indexes
                selected_indexes = np.random.choice(np.arange(len(y)), self.batch_size)
                X_selected = X[selected_indexes, :]
                y_selected = y[selected_indexes]
                # pick a neighbor
                neighbor = self._pick_neighbor()
                # evaluate the quality of the solution (with MSE, L1 and L2 loss)
                neighbor_loss = self._evaluate_loss(X_selected, y_selected, neighbor)
                # if the one picked is better than the current solution go for it
                if neighbor_loss <= current_loss:
                    self.betas = neighbor
                    current_loss = neighbor_loss
                    print("improved")
                else:
                    # otherwise pick it with certain probability, or discard it
                    delta = neighbor_loss - current_loss
                    epsilon = random.uniform(0, 1)
                    if epsilon <= math.pow(math.e, -delta / tk):
                        self.betas = neighbor
                        current_loss = neighbor_loss
                        print("Random walk: ", i, m)
                m += 1
            # cooling schedule
            tk *= alpha_SA
            if i % 100 == 0:
                print("Iterations: ", i)

    def predict(self, X):
        return np.matmul(np.c_[np.ones(X.shape[0]), X], self.betas)

    def get_raw_coef(self):
        return self.betas

    def get_coef(self):
        betas_non_zero = self.betas
        betas_non_zero[betas_non_zero < 0.01] = 0
        return betas_non_zero

    def rmse(self, X, y):
        """ root mean squared error

        :param X: feature matrix
        :param y: predictant
        :return: rmse
        """
        squared_errors = np.power(self.predict(X) - y, 2)
        return math.sqrt(sum(squared_errors)/len(squared_errors))

    def set_param(self, l1_ratio=0.15, learning_rate=0.0001, sa_rate=0.5, batch_size=500):
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        # simulated annealing
        self.sa_rate = sa_rate  # rate of neighbor generation for SA
        self.batch_size = batch_size  # number of iterations at a given temperature
