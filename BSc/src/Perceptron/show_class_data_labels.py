## This code just shows the classes of class data
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

data = pd.read_excel("../../data/class.xlsx")
data.head()

F = data["Gender"]=="F"
M = data["Gender"]=="M"

plt.plot(data[M]["Weight"], data[M]["Height"], 'x')
plt.plot(data[F]["Weight"], data[F]["Height"], 'o')
plt.xlabel("Weight")
plt.ylabel("Height")
plt.show()
