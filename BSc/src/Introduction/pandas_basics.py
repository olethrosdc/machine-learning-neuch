import pandas as pd
import numpy as np

X = pd.read_excel("../../data/class.xlsx")

## Accessing data

X["First Name"]

## Plotting data
X.hist()

import matplotlib.pyplot as plt
plt.show()

