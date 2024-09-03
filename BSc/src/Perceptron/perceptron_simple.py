import numpy as np
import matplotlib.pyplot as plt

data = [[180, 1],
        [167, 1],
        [160, 0],
        [170, 0],
        [190, 1],
        [185, 0],
        [190, 1],
        [195, 0],
        [178, 0],
        [165, 1]]

X = np.array(data)

T = X.shape[0]
plt.ion()
threshold = 150
for t in range(T):
    plt.clf()
    plt.axis([150,220,-1, 2])
    plt.plot(X[:t,0], X[:t,1], 'o')
    plt.plot([threshold, threshold], [-1, 2], '--')
    plt.pause(0.1)
    threshold = int(input("place threshold: "))
