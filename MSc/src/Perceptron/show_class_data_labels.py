## This code just shows the classes of class data
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

data = pd.read_csv("../../data/class.csv")
data.head()

F = data["Sex"]==1
M = data["Sex"]==0

print(F)
print(M)


lower_bound = 40
upper_bound = 220
data['Weight'] = np.where((data['Weight'] < lower_bound) | (data['Weight'] > upper_bound), data['Weight'].median(), data['Weight'])

lower_bound = 80
upper_bound = 220
data['Height'] = np.where((data['Height'] < lower_bound) | (data['Height'] > upper_bound), data['Height'].median(), data['Height'])


plt.plot(data[M]["Weight"], data[M]["Height"], 'x')
plt.plot(data[F]["Weight"], data[F]["Height"], 'o')
plt.xlabel("Weight")
plt.ylabel("Height")
plt.show()
