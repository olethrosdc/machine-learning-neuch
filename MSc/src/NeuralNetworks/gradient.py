import numpy as np

mu = 2
n_samples = 100
x = np.random.normal(size=n_samples)
x += mu

n_iterations = 100 #n_samples
exact_estimate = np.zeros(n_iterations)
alpha = 0.1*np.ones(n_iterations)

beta = 0
for t in range(n_iterations):
    beta += alpha[t] * (mu - beta)
    exact_estimate[t] = beta

print("Ex", mu)
print("Mean", np.mean(x))
print("Exact estimate", exact_estimate[n_iterations - 1])

import matplotlib.pyplot as plt
plt.ion()
plt.plot(exact_estimate)
plt.pause(1)

n_iterations = n_samples*10
sample_estimate = np.zeros(n_iterations)
beta = 0
#alpha = 0.01*np.ones(n_iterations)
alpha = 1/(1 + np.arange(n_iterations))
for t in range(n_iterations):
    beta += alpha[t] * (x[t % n_samples] - beta)
    sample_estimate[t] = beta

print("Sample estimate", sample_estimate[n_iterations - 1])
plt.plot(sample_estimate)
plt.pause(10)








