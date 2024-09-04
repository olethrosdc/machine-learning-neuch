import numpy as np

mean = 170
std = 7
n_samples = 1000
height = np.random.normal(mean, std, n_samples)
classes = np.random.choice(2, p=[0.55, 0.45], size=n_samples)
height[classes==1]+=13

import matplotlib.pyplot as plt
plt.hist([height[classes==1], height[classes==0]], 20, histtype='bar')
plt.savefig("histogram_heights.png")
plt.show()

