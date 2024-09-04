import numpy as np
import matplotlib.pyplot as plt


data = [[190, 1],
        [160, -1],
        [170, -1],
        [180, 1],
        [165, 1],
        [185, -1],
        [191, -1],
        [195, -1],
        [178, 1],
        [169, -1],
        [160, 1],
        [210, -1],
        [180, 1]]


X = np.array(data)

T = X.shape[0]
plt.ion()
threshold = 150
direction = 1
for t in range(2, T):
    plt.clf()
    plt.axis([150,220,-1.5, 1.5])
    plt.plot(X[:t,0], X[:t,1], 'o')
    plt.plot([threshold, threshold], [-2, 2], '--')
    plt.arrow(threshold, 0, 5*direction, 0, width = 0.1)
    plt.pause(0.1)
    ## measure classification error


    correct = direction * X[:t,1] * np.sign(X[:t, 0] - threshold)
    cerr = sum(correct<=0)/len(correct)
    print("Error rate: ", correct, cerr)
    threshold = int(input("place threshold: "))
    direction = 1
    if (threshold < 0):
        direction = -1
    else:
        direction = 1
    threshold = abs(threshold)
    
plt.clf()
plt.axis([150,220,-1.5, 1.5])
plt.plot(X[:t,0], X[:t,1], 'o')
plt.plot([threshold, threshold], [-1, 2], '--')
plt.arrow(threshold, 0.5, 5*direction, 0, width = 0.1)
plt.pause(0.1)
## measure classification error

correct = direction * X[:t,1] * np.sign(X[:t, 0] - threshold)
cerr = sum(correct<=0)/len(correct)
print("Error rate: ", cerr)
plt.pause(1)
