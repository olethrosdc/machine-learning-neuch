import numpy as np

def xv_shuffler(x, estimator, scoring, n_folds):
    rng = np.random.default_rng()
    # shuffle data
    T = len(x)
    indices = np.arange(T)
    rng.shuffle(indices)
    fold_size = np.ceil(T / n_folds).astype(int)
    fold_start = np.zeros(n_folds, dtype=int)
    fold_end = np.zeros(n_folds, dtype=int)
    # create folds
    for k in range(n_folds):
        fold_start[k] = k * fold_size
        fold_end[k] = min((k+1) * fold_size, T)
    # for each fold:
    fold_start, fold_end
    # 1. Run estimator on k-1 folds
    for k in range(n_folds):
        train = x[indices].copy()
        valid = train[fold_start[k]:fold_end[k]].copy()
        for t in range(fold_start[k], fold_end[k]):
            train = np.delete(train, fold_start[k])

        print("Train:")
        print(train)
        print("Valid:")
        print(valid)

    # 2. Score estimator on k-th set


def sample_mean(x):
    return np.mean(x)

def mse(estimate, x):
    return np.mean((x - estimate)**2)
    
rng = np.random.default_rng()
#x = np.round(np.random.uniform(150, 210, size = 10))
x = np.arange(10)
print (x)
xv_shuffler(x, sample_mean, mse, 3)
