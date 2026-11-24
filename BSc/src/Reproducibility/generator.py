import numpy as np

class GWASData:
    def __init__(self, signal_strength = 0, prevalence = 0.1):
        self.signal_strength = signal_strength
        self.prevalence = 0.1
    def generate(self, n_individuals=500, n_features=10000):
        p_neg = np.random.uniform(0.05, 0.95, size=n_features)
        p_pos = p_neg * (1 - self.signal_strength) + self.signal_strength * np.random.uniform(0.05, 0.95, size=n_features)
        X = np.zeros([n_individuals, n_features])
        y = np.zeros(n_individuals)
        for t in range(n_individuals):
            if (np.random.uniform() < self.prevalence):
                p = p_pos
                y[t] = 1
            else:
                p = p_neg
                y[t] = 0
            for j in range(n_features):
                X[t,j] = np.random.choice(2, p=[1 - p[j], p[j]])
        return X, y
    

simulator = GWASData()
X, y = simulator.generate(n_individuals = 10, n_features = 5)


