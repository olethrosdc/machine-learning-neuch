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
        self.n_classes = len(np.unique(labels))  # Counts actual number of labels
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
        # HINT:
        # Need to use the get_probabilities function to return the action with the highest
        # expected utility
        # i.e. maximising sum_y P(y|x) U(a,y)

        return
    
    ## predict the most likely label
    def predict(self, x):
        # calculate the probabilities of different clases

        # return the y value for the closest point
        return np.argmax(p)
    

    ## return a vector of probabilities, one for each label
    ## Each component of the vector corresponds to the ratio of that same label in the set of neighbours
    def get_probabilities(self, x):
        # calculate distances
        distances = [self.metric(x, self.data[t]) for t in range(self.n_points)] 
        # sort data using argsort
        # get K closest neighbours
        neighbours =
        proportions =
        # get the proportion of each label
        for k in range(self.K):


        return proportions


if __name__== "__main__":
    
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
