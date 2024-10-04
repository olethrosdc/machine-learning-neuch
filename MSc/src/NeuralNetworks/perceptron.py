import numpy as np
import matplotlib.pyplot as plt

## Generate some random data of two clases
mean = [0, 0]
covariance = [[1,0], [0,1]]
n_samples = 20
features = np.random.multivariate_normal(mean, covariance, n_samples)
classes = np.random.choice([-1, 1], size=n_samples)
features[classes==1]+=2


#plt.axis('equal')
#plt.grid()
#plt.show()
def display_classifier(features, classes, w_old, w, labels):
    X = np.linspace(-7.5,7.5)
    # in our parametrisation, we have
    # a = w[0] + x w[1] + y w[2] 
    # so as a = 0 is the decision boundary, we can solve for the y coordinate
    Y = - (w[0] + X * w[1])/w[2]
    Y_old = - (w_old[0] + X * w_old[1])/w_old[2]
    plt.clf()
    plt.scatter(features[classes==-1,0], features[classes==-1,1], marker='o', c='r', alpha=0.5, s=80)
    plt.scatter(features[classes==1,0], features[classes==1,1], marker='o', c='b', alpha=0.5, s=80)
    plt.scatter(features[labels==-1,0], features[labels==-1,1], marker='+', c='r', alpha=1, s=80)
    plt.scatter(features[labels==1,0], features[labels==1,1], marker='+', c='b', alpha=1, s=80)
    plt.plot(X, Y)
    plt.plot(X, Y_old)
    plt.axis([-7.5,7.5,-5,5])
    plt.grid()
    plt.pause(0.1)


## Iterate over all the examples
## features: A matrix so that features[t] are the features of the t-th example
## classes: a vector so that classes[t] is the class label of the t-th example, in {-1, 1}
## iterations: we may need to go many times through the data, even if the classes are separable
def perceptron(features, classes, iterations=1):
    # initialise parameters randomly
    w_t = np.random.uniform(size=1 + features.shape[1]) # add one more fake feature
    w_t_old = w_t.copy()
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
                if (classes[t] > 0):
                    c = +1
                else:
                    c = -1
                ##   - Move the hyperplane w_t
                n_errors += 1
                w_t +=  c * x_t
            ## - Else do nothing
            ## Then save the label in labels
            #print(x_t)
            display_classifier(features[:t+1], classes[:t+1], w_t_old, w_t, labels[:t+1])
            w_t_old = w_t.copy()
        print("errors: ", n_errors, "error rate:", n_errors / n_data)
        if (n_errors == 0):
            break
    return w_t, labels


plt.ion()
## call the algorithm
w, labels = perceptron(features, classes, 10)

## plot the results
display_classifier(features, classes, w, w, labels)
plt.pause(10)

