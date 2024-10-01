import numpy as np

def xv_shuffler(x, estimator, scoring, n_folds):
    rng = np.random.default_rng()
    # shuffle data
    T = len(x)
    indices = np.arange(T)
    rng.shuffle(indices)
    fold_size = np.ceil(T / n_folds)
    fold_start = np.zeros(n_folds)
    fold_end = np.zeros(n_folds)
    # create folds
    for k in range(n_folds):
        fold_start = k * fold_size
        fold_end = np.min((k+1) * fold_size, T)
    # for each fold:
    fold_start, fold_end
    # 1. Run estimator on k-1 folds
    for k in range(n_folds):
        train = x.copy()
        valid = x.copy([fold_start[k], fold_end[k]])
        for t in range(fold_start[k], fold_end[k]):
            train.delete(fold_start[k])

        print(train)
        print(valid)

    # 2. Score estimator on k-th set


def sample_mean(x):
    return np.mean(x)

def mse(estimate, x):
    return np.mean((x - estimate)**2)
    
rng = np.random.default_rng()
x = np.round(np.random.uniform(150, 210, size = 20))
print (x)
xv_shuffler(x, sample_mean, mse, 3)
