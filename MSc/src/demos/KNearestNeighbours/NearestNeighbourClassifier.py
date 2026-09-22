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
        self.n_points = data.shape[0] # np array dimensions
        self.n_features = data.shape[1]
        print("classes: ", self.n_classes)

    # Gives a utility for every possible choice made by the algorithm
    def decide(self, U, x):
        """
        A method that return the action that maximise the expected utility.
        :param U: is a 2 denominational array that indicated the utility of each action a given y
                    example: U = np.array([ [ 1 , -1000],
                                            [ -1 ,    0]  ])
                            so U[1,0]=-1 is the utility of taking action a=1 when y=0.
        :param x: the test point.
        :return: the action that maximises the expected utility max_a E[U|a,x].
                 where E[U|a,x] = sum_y P(y|x) U(a,y).
        """
        # HINT:
        # Need to use the get_probabilities function to return the action with the highest
        # expected utility
        # i.e. maximising sum_y P(y|x) U(a,y)
        pass
    
    ## predict the most likely label
    def predict(self, x):
        # calculate the probabilities of different clases
        pass
    

    ## return a vector of probabilities, one for each label
    ## Each component of the vector corresponds to the ratio of that same label in the set of neighbours
    def get_probabilities(self, x):
        # 1. calculate distances
        # 2. sort data using argsort
        # 3. get K closest neighbours
        # 4. get the proportion of each label so that proportions[y] is the proportion of label y in the neighbourhood
        # 5. Return proportion
        pass


if __name__== "__main__":
    

    import pandas as pd

    data = pd.read_csv("class.csv")
    x = data[["Height (cm)", "Weight (kg)"]].to_numpy()
    y = data["Sex"].to_numpy()
    print(y)
    y[np.isnan(y)] = 0
    y = y.astype(int)
    print(y)

    kNN = NearestNeighbourClassifier(x, y, euclidean_metric, 1)
    
    U_acc = np.array([[1, 0],
                      [0, 1]])


    U = np.array([[1, -1000],
                  [-1, 0]])


    print("Utility matrix")
    print(U)
    print("Class probability and utility of decision");
    print("-----------------------------------------");

    neg_log_loss = 0
    utility = 0
    for t in range(x.shape[0]):
        x_t = x[t]
        p = kNN.get_probabilities(x_t)
        
        # Assignment 1
        # fill the kNN.decide method of the knn class above.
        decision = kNN.decide(U=U, x=x_t)
#        print(p)
#        print(p[y[t]], U[decision, y[t]])
        utility +=  U[decision, y[t]]
        neg_log_loss += np.log(p[y[t]])

    print("Mean util:", utility/x.shape[0])
    print("Mean logprob:", neg_log_loss/x.shape[0])
                            
        
