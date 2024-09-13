import numpy as np
import matplotlib.pyplot as plt

## Generate some random data of two clases
mean = [0, 0]
covariance = [[10,1], [1,10]]
n_samples = 1000
features = np.random.multivariate_normal(mean, covariance, n_samples)
classes = np.random.choice(2, size=n_samples)
features[classes==1]+=5
plt.plot(features[classes==0,0], features[classes==0,1], '.', alpha=0.5)
plt.plot(features[classes==1,0], features[classes==1,1], '.', alpha=0.5)
plt.axis('equal')
plt.grid()
plt.show()

## Iterate
def perceptron(features, classes, iterations=100):
    # initialise parameters
    w_t = np.random.uniform(size=1 + features.shape[1]) # add one more fake feature
    n_data = features.shape[0]
    labels = np.zeros(n_data)
    for k in range(iterations):
        # calculate output
        n_errors = 0
        for t in range(n_data):
            ## TO DO: fill in
            x_t = np.concatenate([np.array([1]), features[t]])
            #print(x_t)
            labels[t] = np.sign(np.dot(w_t, x_t))
            if (classes[t] == 0):
                c = -1
            else:
                c = +1
            if (c != labels[t]):
                w_t += c * x_t
                n_errors += 1
        print("error rate: ", n_errors / n_data)
    return w_t, labels


## call the algorithm
w, labels = perceptron(features, classes, 100)

## plot the results

plt.plot(features[labels==-1,0], features[labels==-1,1], 'r.', alpha=0.5)
plt.plot(features[labels==1,0], features[labels==1,1], 'g.', alpha=0.5)
X = np.linspace(-10,10)
# in our parametrisation, we have
# a = w[0] + x w[1] + y w[2] 
# so as a = 0 is the decision boundary, we can solve for the y coordinate
Y = - (w[0] + X * w[1])/w[2]
plt.plot(X, Y)
plt.axis([-10,10,-10,10])
plt.grid()
plt.show()
