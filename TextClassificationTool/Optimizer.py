from .LossFunctions import LossFunction, MeanSquaredError
import pandas as pd
import numpy as np


class Optimizer:
    def __init__(self, **kwargs):
        self.learn_rate = kwargs.get('learn_rate', 0.01)
        self.L1 = kwargs.get('L1', 0.0)
        self.L2 = kwargs.get('L2', 0.0)
        self.epochs = kwargs.get('epochs', 100)
        self.epsilon = kwargs.get('epsilon', 0.001)
        self.batch_size = kwargs.get('batch_size', None)
        self.momentum_beta = kwargs.get('momentum_beta', 0.5)
        self.loss_function: LossFunction = kwargs.get('loss_function', MeanSquaredError())

    def gradientDescent(self, X: pd.DataFrame[int, np.ndarray], y: pd.DataFrame[int, np.ndarray], **kwargs):
        """Linear Regression optimization function implementing gradient descent (defaults to loss function: MSE).
        Combines loss gradients, L1&L2 regularization and momentum.
        """
        self._updateAttributes(**kwargs)
        n_features = X.iloc[0, 0].shape[0]
        n_samples = len(X)
        weights = np.zeros(n_features)
        bias = 0.0
        velocity_w = np.zeros(n_features)
        velocity_b = 0.0

        for _ in range(self.epochs):
            gradient_w, gradient_b = self._computeGradients(X, y, n_features, n_samples, weights, bias)
            # L1 & L2 REGULARIZATION
            gradient_w = self._lassoRegularizedGradient(gradient_w, weights)
            gradient_w = self._ridgeRegularizedGradient(gradient_w, weights)
            # MOMENTUM
            velocity_w = self._updateMomentum(gradient_w, velocity_w)
            velocity_b = self._updateMomentum(gradient_b, velocity_b)
            # Adjust Weights & Biases
            weights -= self.learn_rate * velocity_w
            bias -= self.learn_rate * velocity_b
        return weights, bias

    # ............................................................................................
    def _updateAttributes(self, **kwargs):
        self.learn_rate = kwargs.get('learn_rate', self.learn_rate)
        self.L1 = kwargs.get('L1', self.L1)
        self.L2 = kwargs.get('L2', self.L2)
        self.epochs = kwargs.get('epochs', self.epochs)
        self.epsilon = kwargs.get('epsilon', self.epsilon)
        self.batch_size = kwargs.get('batch_size', self.batch_size)
        self.momentum_beta = kwargs.get('momentum_beta', self.momentum_beta)
        self.loss_function: LossFunction = kwargs.get('loss_function', self.loss_function)

    # ========================================================================================== COMPUTE GRADIENTS
    def _computeGradients(self, X, y, n_features, n_samples, weights, bias):
        gradient_w = np.zeros(n_features)
        gradient_b = 0.0
        for key, x_vector in X.iterrows():
            y_actual = y.loc[key]
            y_hat = np.dot(weights, x_vector) + bias
            error = y_hat - y_actual
            gradient_w, gradient_b = self.loss_function.calculateLoss(gradient_w, gradient_b, x_vector, error)
        if isinstance(self.loss_function, MeanSquaredError):
            gradient_w, gradient_b = self.loss_function.scaleLoss(gradient_w, gradient_b, n_samples)
        return gradient_w, gradient_b

    # ========================================================================================== L1 & L2 REGULARIZATION
    def _lassoRegularizedGradient(self, gradient, weights):
        if self.L1 > 0.0:
            lasso_regularization = self.L1 * np.sign(weights)
            gradient += lasso_regularization
        return gradient

    def _ridgeRegularizedGradient(self, gradient, weights):
        if self.L2 > 0.0:
            ridge_regularization = 2.0 * self.L2 * weights
            gradient += ridge_regularization
        return gradient

    # ========================================================================================== MOMENTUM
    def _updateMomentum(self, gradient, velocity):
        inertia = self.momentum_beta * velocity
        driving_force = (1 - self.momentum_beta) * gradient
        return inertia + driving_force

