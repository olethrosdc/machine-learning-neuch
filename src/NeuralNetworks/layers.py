import numpy as np
import matplotlib.pyplot as plt
mean = [0, 0]
covariance = [[10,0], [0,10]]
n_samples = 1000
data = np.random.multivariate_normal(mean, covariance, n_samples)

plt.plot(data[:,0], data[:,1], '.', alpha=0.5)
plt.axis('equal')
plt.grid()
plt.show()

# A linear transformation

A = np.random.normal(size=[2,2])

z = np.dot(data, A)

plt.plot(z[:,0], z[:,1], '.', alpha=0.5)
plt.axis('equal')
plt.grid()
plt.show()

# A non-linear transformation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10,10)
plt.plot(x, sigmoid(x))

# Transform the final layer
y = sigmoid(z)


plt.plot(y[:,0], y[:,1], '.', alpha=0.5)
plt.axis('equal')
plt.grid()
plt.show()





