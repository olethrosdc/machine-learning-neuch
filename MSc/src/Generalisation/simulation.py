# Here we have a small experiment for 

import numpy as np
import scipy


# generate data
n_training_data = 100
n_test_data = 100
n_limit_data = 1000

class GaussianGenerator:
    def __init__(self, n_dim, n_classes, class_prob):
        self.mean = np.zeros([n_classes, n_dim])
        self.covariance = np.zeros([n_classes, n_dim, n_dim])
        for c in range(n_classes):
            self.mean[c] = np.random.normal(n_dim)
            self.covariance[c] = scipy.stats.wishart.rvs(n_dim, np.identity(n_dim))
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.class_prob = class_prob
    def generate(self, n_data):
        X = np.zeros([n_data, n_dim])
        y = np.zeros(n_data)
        for t in range(n_data):
            c = np.random.choice(range(self.n_classes), p = self.class_prob)
            X[t] = scipy.stats.multivariate_normal.rvs(self.mean[c], self.covariance[c])
            y[t] = c
        return X, y

n_dim = 2
n_classes = 2
class_prob = [0.8, 0.2]
generator = GaussianGenerator(n_dim, n_classes, class_prob)
generator.mean[0] = np.array([1, 1])
generator.mean[1] = np.array([-1, -1])
generator.covariance[0] = np.array([[1,0],[0,1]])
generator.covariance[1] = np.array([[2,0.5],[0.5,2]])

from sklearn import neighbors
from sklearn.metrics import accuracy_score

def run_experiment(generator, n_training_data, n_test_data, n_limit_data):
    train_X, train_y = generator.generate(n_training_data)
    test_X, test_y = generator.generate(n_test_data)
    X, y = generator.generate(n_limit_data)


    max_neighbours = len(train_y)
    acc_train = np.zeros(max_neighbours)
    acc_test = np.zeros(max_neighbours)
    acc_lim = np.zeros(max_neighbours)

    for n_neighbours in range(1, max_neighbours + 1):
        clf = neighbors.KNeighborsClassifier(n_neighbours)
        clf.fit(train_X, train_y)

        ## Calculate the accuracy score, based on the predicted labels 
        y_predicted = clf.predict(train_X)
        acc_train[n_neighbours - 1] = accuracy_score(train_y, y_predicted)

        ## Calculate the test score
        y_predicted = clf.predict(test_X)
        acc_test[n_neighbours - 1] = accuracy_score(test_y, y_predicted)

        ## Calculate the actual score
        y_predicted = clf.predict(X)
        acc_lim[n_neighbours - 1] = accuracy_score(y, y_predicted)

    return acc_train, acc_test, acc_lim


import matplotlib.pyplot as plt

acc_train, acc_test, acc_lim = run_experiment(generator, n_training_data, n_test_data, n_limit_data)



plt.clf()
plt.plot(acc_train)
plt.plot(np.argmax(acc_train), acc_train[np.argmax(acc_train)], '*')
plt.legend(["train"])
#plt.show()
plt.savefig("knn-gaussian-train.pdf")

plt.clf()
plt.plot(acc_train)
plt.plot(acc_test)
plt.plot(np.argmax(acc_train), acc_train[np.argmax(acc_train)], '*')
plt.plot(np.argmax(acc_test), acc_test[np.argmax(acc_test)], 'x')
plt.legend(["train", "test"])
#plt.show()
plt.savefig("knn-gaussian-test.pdf")

plt.clf()
plt.plot(acc_train)
plt.plot(acc_test)
plt.plot(acc_lim)
plt.plot(np.argmax(acc_train), acc_train[np.argmax(acc_train)], '*')
plt.plot(np.argmax(acc_test), acc_test[np.argmax(acc_test)], 'x')
plt.plot(np.argmax(acc_lim), acc_lim[np.argmax(acc_lim)], 'o')
plt.legend(["train", "test", "actual"])
#plt.show()
plt.savefig("knn-gaussian-all.pdf")

max_neighbours = n_training_data
av_acc_train = np.zeros(max_neighbours)
av_acc_test  = np.zeros(max_neighbours)
av_acc_lim  = np.zeros(max_neighbours)

n_experiments = 100
for experiment in range(n_experiments):
    print("Experiment ", experiment)
    acc_train, acc_test, acc_lim = run_experiment(generator, n_training_data, n_test_data, n_limit_data)
    av_acc_train += acc_train / n_experiments
    av_acc_test += acc_test / n_experiments
    av_acc_lim += acc_lim / n_experiments


plt.clf()
plt.plot(av_acc_train)
plt.plot(np.argmax(av_acc_train), av_acc_train[np.argmax(av_acc_train)], '*')
plt.legend(["train"])
#plt.show()
plt.savefig("knn-gaussian-train-average.pdf")

plt.clf()
plt.plot(av_acc_train)
plt.plot(av_acc_test)
plt.plot(np.argmax(av_acc_train), av_acc_train[np.argmax(av_acc_train)], '*')
plt.plot(np.argmax(av_acc_test), av_acc_test[np.argmax(av_acc_test)], 'x')
plt.legend(["train", "test"])
#plt.show()
plt.savefig("knn-gaussian-test-average.pdf")

plt.clf()
plt.plot(av_acc_train)
plt.plot(av_acc_test)
plt.plot(av_acc_lim)
plt.plot(np.argmax(av_acc_train), av_acc_train[np.argmax(av_acc_train)], '*')
plt.plot(np.argmax(av_acc_test), av_acc_test[np.argmax(av_acc_test)], 'x')
plt.plot(np.argmax(av_acc_lim), av_acc_lim[np.argmax(av_acc_lim)], 'o')
plt.legend(["train", "test", "actual"])
#plt.show()
plt.savefig("knn-gaussian-all-average.pdf")
