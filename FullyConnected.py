from Optimization import Optimizers
import numpy as np

class FullyConnected:
    def __init__(self, input_size, output_size):
        self.input_size=input_size
        self.output_size=output_size
        self._optimizer = None
        self._gradient_weights = None
        self.weights = np.random.uniform(0,1,(input_size + 1, output_size))
        self.previouslayer = None

    def get_optimizer(self):
        return self._optimizer

    def set_optimizer(self,optimizer):
        self._optimizer=optimizer

    optimizer = property(get_optimizer,set_optimizer)

    def get_gradient_weights(self):
        return self._gradient_weights
    def set_gradient_weights(self,gradient_weights):
        self._gradient_weights=gradient_weights

    gradient_weights = property(get_gradient_weights,set_gradient_weights)

    def forward(self,input_tensor):
        bias = np.ones((input_tensor.shape[0], 1))
        input_tensor = np.hstack((input_tensor, bias))

        self.previouslayer=input_tensor
        return np.matmul(input_tensor, self.weights)

    def backward(self,error_tensor):
        previouslayer =self.previouslayer
        gradient = np.matmul(error_tensor, self.weights[0:self.input_size,:].T)


        gradient_weight = np.matmul(previouslayer.T,error_tensor)
        self.set_gradient_weights(gradient_weight)

        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, gradient_weight)

        return gradient





