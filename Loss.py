
import numpy as np
class CrossEntropyLoss:
    def __init__(self):
        self.previouslayer = None

    def forward(self, input_tensor, label_tensor):
        self.previouslayer = input_tensor
        return np.sum(-np.log(input_tensor + np.finfo(float).eps)*label_tensor)

    def backward(self,label_tensor):
        return - label_tensor/self.previouslayer