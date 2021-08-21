import numpy as np

class ReLU:
    def __init__(self):
        self.previouslayer = None

    def forward(self,input_tensor):
        self.previouslayer=input_tensor
        #output_tensor = input_tensor[:,:] if input_tensor[:,:] > 0 else 0
        return  np.maximum(0,input_tensor)

    def backward(self,error_tensor):
        previouslayer = self.previouslayer
        previouslayer[previouslayer<=0] = 0
        previouslayer[previouslayer>0] =1
        return previouslayer*error_tensor