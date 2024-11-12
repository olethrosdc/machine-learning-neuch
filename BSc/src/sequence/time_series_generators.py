import numpy as np

class GaussianTimeSeries:
    def __init__(self, scale):
        self.state = 0
        self.scale = scale
    def generate(self):
        x = self.state + self.scale * np.random.normal()
        self.state = x

class LinearGaussianTimeSeries:
    def __init__(self, scale):
        self.state = 0
        self.scale = scale
    def generate(self):
        x = self.state + self.scale * np.random.normal()
        self.state = x

class DiscreteTimeSeries:
    def __init__(self, n_symbols, order):
        self.state = np.zeros(order, dtype=int)
        self.n_symbols = n_symbols
        self.order = order
        shape = np.ones(order, dtype=int) * n_symbols
        self.transitions = np.random.dirichlet(alphas, size=shape)
    def generate(self):
        P = self.transitions(tuple(self.state))
        x = np.random.choice(self.n_symbols, p = P)
        
        
        
