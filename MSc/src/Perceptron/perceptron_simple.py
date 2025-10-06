## Run this code to observe how the perceptron works
## At each step, enter the location of the threshold
## Use a + sign if you want to use the +1 label on the right.
## Use a - sign if you want to use the -1 label on the right.
import numpy as np
import matplotlib.pyplot as plt


data = [[178,-1],
        [176,-1],
        [158,1],
        [170,1],
        [185,-1],
        [165,1],
        [165,-1],
        [150,1],
        [165,1],
        [174,-1],
        [173,-1],
        [158,1],
        [182,-1],
        [175,-1],
        [160,1],
        [170,1],
        [192,-1],
        [174,1],
        [162,1],
        [181,-1],
        [185,-1],
        [179,-1],
        [188,-1],
        [183,-1],
        [190, 1]]


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
