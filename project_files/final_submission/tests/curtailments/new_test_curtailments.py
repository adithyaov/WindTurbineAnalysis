
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from sklearn.cluster import *


# In[2]:


data_set = pd.read_csv('/home/mytrah-pc/Mytrah_Adithya/data_turbine/ScatterData_Mokal_MK014.csv')


# In[3]:


required_data = data_set[['ActivePower', 'WindSpeed', 'Timestamp']].copy()
required_data = required_data[required_data['ActivePower'] > 0]


# In[4]:


max_wind_speed = required_data['WindSpeed'].max()
min_wind_speed = required_data['WindSpeed'].min()
max_active_power = required_data['ActivePower'].max()
min_active_power = required_data['ActivePower'].min()
normalization_factor_wind_speed = max_wind_speed - min_wind_speed
normalization_factor_active_power = max_active_power - min_active_power
scale_factor_wind_speed = 4.0/3
scale_factor_active_power = 1


# In[5]:


required_data['ScaledWindSpeed'] = ((required_data['WindSpeed'] - min_wind_speed) * scale_factor_wind_speed)                                     / normalization_factor_wind_speed
required_data['ScaledActivePower'] = ((required_data['ActivePower'] - min_active_power) * scale_factor_active_power)                                     / normalization_factor_active_power


# In[6]:


required_data['cluster_number'] = -2


# In[7]:


clustering_algorithm = DBSCAN(
    eps=0.3/20,
    min_samples=15
)


# In[8]:


required_data['cluster_number'] = clustering_algorithm.fit_predict(required_data[['ScaledWindSpeed', 'ScaledActivePower']])


# In[9]:


import random
r = lambda: random.randint(0,255)
random_color = lambda: '#%02X%02X%02X' % (r(),r(),r())


# In[10]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))


# In[11]:


max_elements_cluster = -2
max_elements = 0
for group in required_data.groupby('cluster_number'):
    if(group[1].shape[0] > max_elements):
        max_elements = group[1].shape[0]
        max_elements_cluster = group[0]
    plt.scatter(
        group[1]['ScaledWindSpeed'],
        group[1]['ScaledActivePower'],
        c=random_color(),
        s=np.pi*(2**2)
    )


# In[12]:


from scipy.optimize import curve_fit
push_down_value = 0

filter1 = required_data[required_data['cluster_number'] == max_elements_cluster]
filter2 = filter1[(filter1['ScaledActivePower'] > 0.03) & (filter1['ScaledActivePower'] < 0.97)]


def a_sig(x, d, c):
    return 1/(1 + np.exp(-d * x + c))



popt, pcov = curve_fit(a_sig, filter2['ScaledWindSpeed'], filter2['ScaledActivePower'])

x = np.linspace(0, scale_factor_wind_speed)
y = a_sig(x, popt[0] + 1, popt[1] + 1) - push_down_value

plt.plot(x, y, 'r-')


plt.show()

for group in required_data.groupby('WindSpeed'):
    print group[1]['ActivePower'].mean()


# In[ ]:





# In[ ]:





# In[ ]:




