## This code just shows the classes of class data
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

data = pd.read_csv("../../Data/class.csv")
data.head()

C = data["Continuous"]=="Y"
NC = data["Continuous"]=="N"

import datetime as dt
wake = pd.to_datetime(data["Wake"])
sleep = pd.to_datetime(data["Sleep"])
hours = (wake - sleep).dt.components['hours'] + (wake - sleep).dt.components['minutes']/60

plt.plot(data[C]["Sleep"], data[C]["Wake"], 'x')
plt.plot(data[NC]["Sleep"], data[NC]["Wake"], 'o')
plt.xlabel("Sleep")
plt.ylabel("Wake")
plt.show()

plt.plot(hours[C], data[C]["Screen"], 'x')
plt.plot(hours[NC], data[NC]["Screen"], 'o')
plt.xlabel("ZZZ")
plt.ylabel("Screen")
plt.show()
