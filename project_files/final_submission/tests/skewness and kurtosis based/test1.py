
# coding: utf-8

# In[18]:


import scipy.stats as ss
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

skewness = []
kurtosis = []
plt.figure(figsize=(20, 10))
i = 0
data_path = '/home/mytrah-pc/Mytrah_Adithya/data_turbine/'

def rounding_method(multiply_factor, round_off_by_factor):
    return lambda number: int(number * multiply_factor) + round_off_by_factor - int(number * multiply_factor) % round_off_by_factor

for f in listdir(data_path):
    if f == 'dt':
        continue
    i = i + 1
    data_set = pd.read_csv(data_path + f)

    x = ss.skew(data_set['ActivePower'])
    y = ss.kurtosis(data_set['ActivePower'])
    skewness.append(x)
    kurtosis.append(y)
    plt.text(x, y, i)

plt.scatter(skewness, kurtosis, c='#ff0000')
plt.show()


# In[19]:


i = 0
for f in listdir(data_path):
    if f == 'dt':
        continue
    i = i + 1
    data_set = pd.read_csv(data_path + f)
    if data_set.shape[0] < 10:
        continue


    
    plt.figure(figsize=(20, 10))
    plt.scatter(data_set['WindSpeed'], data_set['ActivePower'])
    print i
    x = ss.skew(data_set['ActivePower'])
    y = ss.kurtosis(data_set['ActivePower'])
    print x, y
    plt.show()


# In[ ]:




