
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn.cluster import DBSCAN, KMeans

data_set = pd.read_csv('./data/ScatterData_Burgula_B-547.csv')

# Algorithm Object
alg = DBSCAN(eps=1, min_samples=8)
alg_km = KMeans(n_clusters=12)

# Filter Data
data_set = data_set[data_set['TurbineState'] == 100]
data_set = data_set[data_set['ActivePower'] > 0]
data_set = data_set[data_set['RotorSpeed'] > 20]

# Scale Data
AP_min = data_set['ActivePower'].min()
RS_min = data_set['RotorSpeed'].min()
AP_max = data_set['ActivePower'].max()
RS_max = data_set['RotorSpeed'].max()

data_set['s_AP'] = ((data_set['ActivePower'] - AP_min) * 100) / (AP_max - AP_min)
data_set['s_RS'] = ((data_set['RotorSpeed'] - RS_min) * 100) / (RS_max - RS_min)

# Vars
bucket_size =  10

# Fit alg
data_set['dbscan_label'] = alg.fit_predict(data_set[['s_RS', 's_AP']])
data_set = data_set[data_set['dbscan_label'] != -1]

# Make Clusters and add em up
crop_data = data_set[(data_set['s_RS'] > 20) & (data_set['s_RS'] < 80)].copy()
crop_data['km_label'] = alg_km.fit_predict(crop_data[['s_RS', 's_AP']])


# Print stats
print ss.skew(data_set[(data_set['s_RS'] > 20) & (data_set['s_RS'] < 80)]['s_AP'])
print ss.kurtosis(data_set[(data_set['s_RS'] > 20) & (data_set['s_RS'] < 80)]['s_AP'])

# Plot Data
plt.figure(figsize=(20, 10))
plt.scatter(data_set['s_RS'], data_set['s_AP'])
plt.show()

# Rotate
theta = 22
rot_mat = np.array([
    [np.cos((np.pi * theta) / 180), -np.sin((np.pi * theta) / 180)],
    [np.sin((np.pi * theta) / 180), np.cos((np.pi * theta) / 180)]
])
crop_data['rot_RS'] = 0
crop_data['rot_AP'] = 0
crop_data[['rot_RS', 'rot_AP']] = np.dot(crop_data[['s_RS', 's_AP']], rot_mat)


# Plot Data
total_std = 0
total_sk = 0
total_kurt = 0
plt.figure(figsize=(10, 10))
for g in crop_data.groupby('km_label'):
    print  np.std(g[1]['rot_AP']), ss.skew(g[1]['rot_AP']), ss.kurtosis(g[1]['rot_AP'])
    total_std = total_std + np.std(g[1]['rot_AP'])
    total_sk = total_sk + np.abs(ss.skew(g[1]['rot_AP']))
    total_kurt = total_kurt + ss.kurtosis(g[1]['rot_AP'])
    plt.scatter(g[1]['rot_RS'], g[1]['rot_AP'])
print '----------------------------------------------------'
print total_std, total_sk, total_kurt
plt.show()

