import numpy as np
import copy
class NeuralNetwork:
    def __init__(self,optimizer):
        self.optimizer=optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.labels =None

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        self.labels = label_tensor
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        return self.loss_layer.forward(input_tensor, self.labels)


    def backward(self):
        error_tensor = self.loss_layer.backward(self.labels)
        for layer in self.layers[::-1]:
            error_tensor = layer.backward(error_tensor)

        return error_tensor

    def append_trainable_layer(self,layer):
        optimizercopy = copy.deepcopy(self.optimizer)
        layer.optimizer = optimizercopy
        self.layers.append(layer)

    def train(self,iterations):
        for i in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self,input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        prediction = input_tensor
        return prediction