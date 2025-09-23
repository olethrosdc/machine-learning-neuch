import numpy as np


## Returns \|x - y\|_2
def euclidean_metric(x, y):
    return np.linalg.norm(x - y)

## return \|x - y\|_1
def manhattan_metric(x, y):
    ## fill in!
    pass


## Skeleton code to be filled in
class NearestNeighbourClassifier:
    ## Initialise the neighbours with a specific metric function and dataset
    ## Assume labels are in {1, ..., m}
    def __init__(self, data, labels, metric, K):
        self.metric = metric
        self.data = data # features
        self.labels = labels
        self.n_classes = len(np.unique(labels))  # VERY CRUDE
        self.K = K
        self.n_points = data.shape[0]
        self.n_features = data.shape[1]
        print("classes: ", self.n_classes)
        pass

    # Gives a utility for every possible choice made by the algorithm
    def decide(self, U, x):
        """
        A method that return the action that maximise the expected utility.
        :param U: is a 2 denominational array that indicated the utility of each action based on y.
                    example: U = np.array([ [ 1 , -1000],
                                            [ -1 ,    0]  ])
                            so the U[1,0] indicated the utility of tanking the action a=1 based on y=0.
        :param x: the test point.
        :return: the action that maximises the expected utility max_a E[U|a,x].
                 where E[U|a,x] = sum_y P(y|x) U(a,y).
        """
        n_actions = U.shape[0]
        n_labels = U.shape[1]
        assert (n_labels == self.n_classes)
        # HINT:
        # Need to use the get_probabilities function to return the action with the highest
        # expected utility
        # i.e. maximising sum_y P(y|x) U(a,y)
        pass
    
    
    ## predict the most likely label for a data point x
    def predict(self, x):
        """ 
        A method that predicts the most likely label.

        :param x: the test point
        :return: the label y with the highest probability P(y|x)
        """
        p = self.get_probabilities(x)
        retun np.argmax(p)
        # HINT: use get_probabilities() to get the most likely label
        

    ## return a vector of probabilities, one for each label
    def get_probabilities(self, x):
        """Return a vector of all label probabilities for a test
        point. The probability of each label should be proportional to
        the number of neighbors sharing that label. That is, if $y_1,
        \ldots, y_K$ are the labels of the K nearest neighbours, then

        $P(y | x) = 1/K \sum_{i=1}^K I\{y_k = y\}$

        :param x: the test point
        :return: a vector p so that p[i] is the probability of label i.

        """
        # 1. Calculate distances to all points
        distances = [self.metric(x, x[i]) for i in range(self.n_points)]

        # 2. Sort the points        
        neighbours = np.argsort(distances)[:self.K]

        # 3. Calculate the proportion of labels for the K closest points 
        proportions = np.zeros(self.n_classes)
        # go through every neigbhour and increase counts of labels
        for k in range(self.K):
            idx = neighbours[k] # get the corresponding example
            label = self.label[idx] # get label
            proportions[label] += 1

        proportions /= self.K
            # return the proportion of each label
        return proportions


x = np.random.uniform(size=[10, 4])
y = 1 + np.random.choice(2, size=10)

kNN = NearestNeighbourClassifier(x, y, euclidean_metric, 10)

kNN.get_probabilities(x[0])

import pandas as pd

data = pd.read_csv("./class.csv")
x = data[["Height (cm)", "Weight (kg)"]].to_numpy()
y = data["Biking (0/1)"].to_numpy()
print(y)
y[np.isnan(y)] = 0
y += 1
y = y.astype(int)
print(y)

kNN = NearestNeighbourClassifier(x, y, euclidean_metric, 3)
for t in range(x.shape[0]):
    x_t = x[t]
    p = kNN.get_probabilities(x_t)
    print(y[t], p)

    print(p[y[t] - 1])

# Assignment 1
# fill the kNN.decide method of the knn class above.
U = np.array([[1, -1000],
              [-1, 0]])
print("Utility matrix")
print(U)
final_decision = kNN.decide(U=U, x=x_t)
print("final decision")
print(final_decision)
