import abc

import matplotlib.pyplot as plt
import numpy as np
from sklearn import base, exceptions


class LinearRegression(base.BaseEstimator, base.TransformerMixin):
    def __init__(self):
        self.preds = None
        self.y = None
        self.theta = None
        self.X = None

    @abc.abstractmethod
    def fit(self, X, y):
        pass

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.preds = X_b.dot(self.theta)
        return self.preds

    def reg_plot(self, X):
        if X.shape[1] > 2:
            print('Can not plot more than 2 dimensions in the current version')
            return
        if self.theta is None:
            raise exceptions.NotFittedError('The model has not been fit yet. Call the fit() method with appropriate '
                                            'arguments')
        plt.plot(self.X, self.y, 'b.', label='Original Data')
        plt.plot(X, self.predict(X), 'r-', label='Predicted Data')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend(loc='best')
        plt.show()


class NormalEquation(LinearRegression):
    def __init__(self):
        super().__init__()
        self.intercept = 0

    def fit(self, X, y):
        self.X = X
        self.y = y
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept = self.theta[:, 0]
        return self


class SVDLR(LinearRegression):
    def __init__(self, use_lstsq=True, use_pinv=False):
        super().__init__()
        self.s = None
        self.rank = None
        self.residuals = None
        if use_lstsq and use_pinv:
            raise ValueError('Only one of `use_lstsq` and `use_pinv` can be True at once.')
        self.use_lstsq = use_lstsq
        self.use_pinv = use_pinv

    def fit(self, X, y):
        self.X = X
        self.y = y

        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        if self.use_lstsq:
            self.theta, self.residuals, self.rank, self.s = np.linalg.lstsq(X_b, y, rcond=1e-6)
        else:
            self.theta = np.linalg.pinv(X_b).dot(y)

        return self


class GradientDescent(LinearRegression):
    def __init__(self, gd_type='batch', n_epochs=None, lr=0.1, tolerance=np.inf, ls0=5, ls1=50, mb_size=None):
        super().__init__()
        self.type = gd_type
        if n_epochs is None:
            if self.type == 'batch':
                self.n_epochs = 1000
            elif self.type == 'sgd':
                self.n_epochs = 50
            elif self.type == 'minibatch':
                self.n_epochs = 100
            else:
                raise ValueError('`type` should be one of "batch", "sgd" or "minibatch')
        else:
            self.n_epochs = n_epochs
        self.lr = lr
        self.tolerance = tolerance
        self.ls0 = ls0
        self.ls1 = ls1
        self.mb_size = mb_size

    def fit(self, X, y):
        self.X = X
        self.y = y

        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]
        self.theta = np.random.rand(X_b.shape[1], 1)

        if self.type == 'batch':
            for epoch in range(self.n_epochs):
                gradients = 2 / m * X_b.T.dot(X_b.dot(self.theta) - y)
                self.theta = self.theta - self.lr * gradients
                if np.linalg.norm(gradients, ord='fro') <= self.tolerance:
                    break
            return self
        elif self.type == 'sgd':
            for epoch in range(self.n_epochs):
                for i in range(m):
                    rand_index = np.random.randint(m)
                    xi = X_b[rand_index:rand_index + 1]
                    yi = y[rand_index:rand_index + 1]
                    gradients = 2 * xi.T.dot(xi.dot(self.theta) - yi)
                    self.lr = self.learning_schedule(epoch*m + i)   # Aurelien Geron
                    self.theta = self.theta - self.lr*gradients
            return self
        else:
            if self.mb_size is None:
                self.mb_size = X.shape[0]//10
            permutation = np.random.permutation(X.shape[0])
            X_b_shuffled = X_b[permutation]
            y_shuffled = y[permutation]

            for i in range(0, X.shape[0], self.mb_size):
                start_ind = i
                end_ind = min(i + self.mb_size, X.shape[0])

                X_mini_batch = X_b_shuffled[start_ind:end_ind]
                y_mini_batch = y_shuffled[start_ind:end_ind]

                gradients = 2 / self.mb_size * X_mini_batch.T.dot(X_mini_batch.dot(self.theta) - y_mini_batch)
                self.theta = self.theta - self.lr * gradients
            return self


    def learning_schedule(self, t):
        return self.ls0 / (self.ls1 + t)

