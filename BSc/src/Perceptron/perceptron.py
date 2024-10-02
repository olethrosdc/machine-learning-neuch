import numpy as np
import matplotlib.pyplot as plt

## Generate some random data of two clases
mean = [0, 0]
covariance = [[10,1], [1,10]]
n_samples = 100
features = np.random.multivariate_normal(mean, covariance, n_samples)
classes = np.random.choice(2, size=n_samples)
features[classes==1]+=5

#plt.axis('equal')
#plt.grid()
#plt.show()
def display_classifier(features, classes, w, labels):
    X = np.linspace(-10,10)
    # in our parametrisation, we have
    # a = w[0] + x w[1] + y w[2] 
    # so as a = 0 is the decision boundary, we can solve for the y coordinate
    Y = - (w[0] + X * w[1])/w[2]
    plt.clf()
    plt.plot(features[classes==0,0], features[classes==0,1], '.', alpha=0.5)
    plt.plot(features[classes==1,0], features[classes==1,1], '.', alpha=0.5)
    plt.plot(features[labels==-1,0], features[labels==-1,1], 'r.', alpha=0.5)
    plt.plot(features[labels==1,0], features[labels==1,1], 'g.', alpha=0.5)
    plt.plot(X, Y)
    plt.axis([-10,10,-10,10])
    plt.grid()
    plt.pause(0.1)


## Iterate over all the examples
## features: A matrix so that features[t] are the features of the t-th example
## classes: a vector so that classes[t] is the class label of the t-th example, in {-1, 1}
## iterations: one iteration should be enough if the data is separable
def perceptron(features, classes, iterations=1):
    # initialise parameters randomly
    w_t = np.random.uniform(size=1 + features.shape[1]) # add one more fake feature
    n_data = features.shape[0]
    labels = np.zeros(n_data)
    for k in range(iterations):
        # calculate output
        n_errors = 0
        for t in range(n_data):
            # We add this feature to every data point
            x_t = np.concatenate([np.array([1]), features[t]])
            ## TO DO: Fill in the from the pseudocode
            ## - Classify example
            labels[t] = np.sign(np.dot(w_t, x_t))
            ## - If the label is wrong...
            if (labels[t] != classes[t]):
                ##   - Move the hyperplane w_t
                n_errors += 1
                w_t += labels[t] * x_t
            ## - Else do nothing
            ## Then save the label in labels
            #print(x_t)
        display_classifier(features, classes, w_t, labels)
        print("error rate: ", n_errors / n_data)
    return w_t, labels


plt.ion()
## call the algorithm
w, labels = perceptron(features, classes, 100)

## plot the results
display_classifier(features, classes, w, labels)


