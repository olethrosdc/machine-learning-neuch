import numpy as np

def euclidean_metric(x, y):
    return np.linalg.norm(x - y)

## Skeleton code to be filled in
class NearestNeighbourClassifier:
    ## Initialise the neighbours with a specific metric function and dataset
    ## Assume labels are in {1, ..., m}
    def __init__(self, data, labels, metric, K):
        self.metric = metric
        self.data = data
        self.labels = labels
        self.n_classes = int(max(labels)) # VERY CRUDE
        self.K = K
        self.n_points = data.shape[0]
        self.n_features = data.shape[1]
        print("classes: ", self.n_classes)
        pass
    # Gives a utility for every possible choice made by the algorithm
    def decide(self, U):
        n_actions = U.shape[0]
        n_labels = U.shape[1]
        assert(n_labels == self.n_classes)
        # Need to use the get_probabilities function to return the action with the highest
        # expected utility
        # i.e. maximising sum_y P(y|x) U(a,y)
    ## predict the most likely label
    def predict(self, x):
        proportions = self.get_probabilities(x)
        return np.argmax(proportions) # is that a good idea?

    ## return a vector of 
    def get_probabilities(self, x):
        # calculate distances
        distances = np.zeros(self.n_points)
        for t in range(self.n_points):
            distances[t] = self.metric(x, self.data[t])
        # sort data
        
        indices = np.argsort(distances)
        # get K closest neighbours
        proportions = np.zeros(self.n_classes)
        for i in range(self.K):
            index = indices[i]
            label = self.labels[index] 
            proportions[label - 1] += 1
        proportions /= self.K
        
        # get the proportion of each label
        return proportions
    
    
