import numpy as np

class SoftMax:
    def __init__(self):
        self.previouslayer=None
        self.currentlayer=None

    def forward(self,input_tensor):
        self.previouslayer=input_tensor
        shifted = input_tensor - np.max(input_tensor)
        exponential_of_shifted_input_tensor = np.exp(shifted)

        output_tensor= exponential_of_shifted_input_tensor/np.sum(exponential_of_shifted_input_tensor,axis=1,keepdims=True)
        self.currentlayer=output_tensor
        return output_tensor

    def backward(self,label_tensor):
        helper=label_tensor - np.sum(label_tensor * self.currentlayer, axis=1, keepdims=True)
        updated = self.currentlayer * helper

        return updated
