import numpy as np 
import math 


def euclidean_dist(x1, x2):
    distance = math.sqrt(np.sum((x1 - x2) ** 2))
    return distance


class KNNRegressor():
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors 


    def fit(self, X, y):
        self.X = X
        self.y = y 
        return self 
    

    def predict(self, X):
        predictions = [self._predict(x) for x in self.X]
        


    def _predict(self, x):
        distances = []
        for sample in self.X:
            distances.append(euclidean_dist(x, sample))
        
        sorted_indices = np.argsort(np.array(distances))[:self.n_neighbors]
        
