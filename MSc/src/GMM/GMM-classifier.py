import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp

class GMMClassifier:
    # n_components: the number of components per class
    def __init__(self, n_components):
        self.n_components = n_components

    ## X: The input features
    ## y: The labels
    ## Returns: nothing
    ## Effect: Creates N Gaussian mixture models, one for each class, as well as the prior class probabilities
    def fit(self, X, y):
        classes = set(y)
        self.n_classes = len(set(y))
        # self.models = -- fill in
        # self.class_probabilities = -- fill in


    ## Calculate the probability of a class given the features, i.e. P(y|x).
    ## You can calculate this with Bayes's theorem:
    ## $P(y|x) = P(x|y) P(y) / P(x)$.
    ## We can obtain $P(x|y)$ from the y'th GMM
    ## $P(y)$ is simply the prior class probabilities
    ## $P(x) = \sum_{y'} P(x|y') P(y')$
    def predict_proba(self, X):
        n_examples = X.shape[0]
        posterior = np.zeros([self.n_classes, n_examples])
        ## FILL IN
        return posterior
    
# test code
from sklearn.datasets._samples_generator import make_blobs
import matplotlib.pyplot as plt
X, y_true = make_blobs(n_samples=4000, centers=4, cluster_std=0.60, random_state=0)
# make it into two labels
y_true[y_true<=2]=0
y_true[y_true>2]=1

model = GMMClassifier(4)
model.fit(X, y_true)
posterior = model.predict_proba(X)
print(np.mean(posterior[y_true]))
    




