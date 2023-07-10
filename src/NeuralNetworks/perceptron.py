import numpy as np
import matplotlib.pyplot as plt

## Generate some random data of two clases
mean = [0, 0]
covariance = [[10,0], [0,10]]
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
    params = np.zeros(1 + features.shape[1]) # add one more fake feature

    for t in range(iterations):
        # calculate output
        x_t = np.block([features[t],np.array([1])])
        y_t = 
