import numpy as np
from collections import Counter
from itertools import groupby
class Knn():
    
    def __init__(self,k)->None:
        self.k = k

    def fit(self, Xtrain,Ytrain):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain


    def get_top_k(self,array):
        new_array =np.column_stack((list(range(len(array))),array)).tolist()
        topk = []
        min_index = 0
        for j in range(self.k):
            min_ = new_array[0][1]
            for index in range(len(new_array)):
            if new_array[index][1] <= min_:
                min_ = new_array[index][1]
                min_index = index
            topk.append(new_array[min_index])
            new_array.pop(min_index)
        return np.array(topk) # Thuật toán có độ phức tạp k*n
    
    def predict(self, x_input:np.array):
        distance_vector = np.square(x_input-self.Xtrain)
        distance = np.sum(distance_vector,axis=1)
        topk = self.get_top_k(distance)
        y_frequency = self.Ytrain[topk[:,0]]
        frequency = [len(list(group)) for key, group in groupby(sorted(y_frequency))]
        return frequency[0]
        