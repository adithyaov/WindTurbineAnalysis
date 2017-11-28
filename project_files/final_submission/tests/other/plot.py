
# coding: utf-8

# In[7]:


import matplotlib.pyplot as plt_obj

fig_obj = plt_obj.figure(figsize = (20, 10))
axes_obj = fig_obj.add_subplot(111)


# In[8]:


def my_plotter(axes_obj, x, y, param_dict):
    out = axes_obj.plot(x, y, **param_dict)
    return out


# In[9]:


my_plotter(axes_obj, [1,2,3,4], [1,2,3,4], {})


# In[10]:


plt_obj.show()


# In[ ]:




