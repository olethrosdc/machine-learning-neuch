import numpy as np

class GaussianTimeSeries:
    def __init__(self, scale):
        self.state = 0
        self.scale = scale
    def generate(self):
        x = self.state + self.scale * np.random.normal()
        self.state = x
        return x
    
class LinearGaussianTimeSeries:
    def __init__(self, scale, order):
        self.order = order
        self.state = np.zeros(order)
        self.scale = scale
        self.coeffs = np.random.uniform(size=order) - 0.5
    def generate(self):
        x = np.dot(self.state, self.coeffs) + self.scale * np.random.normal()
        self.state[:-1] = self.state[1:]
        self.state[-1] = x
        return x
    
class DiscreteTimeSeries:
    def __init__(self, n_symbols, order):
        self.state = np.zeros(order, dtype=int)
        self.n_symbols = n_symbols
        self.order = order
        shape = np.ones(order, dtype=int) * n_symbols
        alpha = np.ones(n_symbols)
        self.transitions = np.random.dirichlet(alpha, size=shape)
    def generate(self):
        P = self.transitions[tuple(self.state)]
        x = np.random.choice(self.n_symbols, p = P)
        self.state[:-1] = self.state[1:]
        self.state[-1] = x
        return x

T = 100

gts = GaussianTimeSeries(0.1)
x = np.zeros(T)
for t in range(T):
    x[t] = gts.generate()
print(x)

import matplotlib.pyplot as plt
plt.plot(x)
plt.title("Gaussian time series")
plt.show()


for order in range(1,4):
    lgts = LinearGaussianTimeSeries(0.1, order)
    for t in range(T):
        x[t] = lgts.generate()
    plt.plot(x)
plt.title("Linear-Gaussian time series")
plt.legend(["1", "2", "3"])
plt.show()


dts = DiscreteTimeSeries(2, 2)
for t in range(T):
    x[t] = dts.generate()
print(x)








