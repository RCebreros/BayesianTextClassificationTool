from abc import ABC, abstractmethod
import numpy as np

class LossFunction(ABC):
    @abstractmethod
    def calculateLoss(self, gradient_w, gradient_b, x_vector, error):
        ...


class MeanSquaredError(LossFunction):
    def calculateLoss(self, gradient_w, gradient_b, x_vector, error):
        gradient_w += error * x_vector
        gradient_b += error
        return gradient_w, gradient_b

    def scaleLoss(self, gradient_w, gradient_b, n_samples):
        gradient_w *= 2.0 / n_samples
        gradient_b *= 2.0 / n_samples
        return gradient_w, gradient_b

class MeanAbsoluteError(LossFunction):
    def calculateLoss(self, gradient_w, gradient_b, x_vector, error):
        gradient_loss = np.sign(error)
        gradient_w += gradient_loss * x_vector
        gradient_b += gradient_loss
        return gradient_w, gradient_b

class LogCoshLoss(LossFunction):
    def calculateLoss(self, gradient_w, gradient_b, x_vector, error):
        gradient_loss = np.tanh(error)
        gradient_w += gradient_loss * x_vector
        gradient_b += gradient_loss
        return gradient_w, gradient_b

