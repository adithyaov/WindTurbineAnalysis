
# coding: utf-8

# In[2]:


import numpy as np


# In[ ]:


class layer_class():
    def __init__(self, weights, biases, activation_function):
        self.weights = weights
        self.biases = biases
        self.activation_function = activation_function

