import numpy as np

def euclidean_metric(x, y):
    return np.norm(x - y)

## Skeleton code to be filled in
class NearestNeighbourClassifier:
    ## Initialise the neighbours with a specific metric function and dataset
    def __init__(self, data, metric):
        self.metric = metric
        self.data = data
        self.n_
        pass
    ## predict the most likely label
    def predict(self, x):
        pass
    ## return a vector of 
    def get_probabilities(self, x):
        pass
    
    
