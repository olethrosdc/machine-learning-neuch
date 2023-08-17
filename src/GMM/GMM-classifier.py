import numpy as np
from sklearn.mixture import GaussianMixture


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
        self.models = [GaussianMixture(n_components = self.n_components).fit(X[y==c]) for c in classes]
        self.class_probabilities = [sum(np.equal(y, c)) for c in classes]
        self.class_probabilities /= sum(self.class_probabilities)
        print(self.class_probabilities)

    ## Calculate the probability of a class given the features, i.e. P(y|x).
    ## You can calculate this with Bayes's theorem:
    ## $P(y|x) = P(x|y) P(y) / P(x)$.
    ## We can obtain $P(x|y)$ from the y'th GMM
    ## $P(y)$ is simply the prior class probabilities
    ## $P(x) = \sum_{y'} P(x|y') P(y')$
    def predict_proba(self, X):
        log_likelihood = [self.models[c].score_samples(X) for c in range(self.n_classes)]
        for t in range(X.shape[0]):
            for c in range(self.n_classes):
                log_likelihood[c][t] += log(self.class_probabilities[c])
                # need to use logsumexp
# test code
from sklearn.datasets._samples_generator import make_blobs
import matplotlib.pyplot as plt
X, y_true = make_blobs(n_samples=4000, centers=4, cluster_std=0.60, random_state=0)
# make it into two labels
y_true[y_true<=2]=0
y_true[y_true>2]=1

model = GMMClassifier(4)
model.fit(X, y_true)
model.predict_proba(X[0:3])




