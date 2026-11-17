import numpy as np

class GWASData:
    def __init__(singal_strength = 0, prevalence = 0.1):
        self.signal_strength = singal_strength
        self.prevalence = 0.1
    def generate(n_individuals=500,n_feature):
        p_pos = np.random.dirichlet(np.ones(n_features))
        p_neg = np.random.dirichlet(np.ones(n_features))
        gene = np.random.choice(n_features)
        p_pos[gene] = 0.5 + signal_strength/2
        p_neg[gene] = 0.5 - signal_strength/2
        X = np.zeros([n_individuals, n_features])
        for t in range(n_individuals):
            if (np.random.uniform() < self.prevalence):
                p = p_pos
                y[t] = 1
            else:
                p = p_neg
                y[t] = 0
            for j in range(n_features):
                X[t,k] = np.random.choice(2, p=[1 - p[j], p[j]])
        return X, y
    


