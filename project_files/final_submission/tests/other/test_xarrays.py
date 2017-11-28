
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import xarray as xr

from os import listdir
from os.path import isfile, join


# In[2]:


data_path = '/home/mytrah-pc/Data/Mokal/'
data_files = [f for f in listdir(data_path) if (f[-4:] == '.csv')]
data_files = sorted(data_files)
num_data_files = len(data_files)
frame_list = []


# In[3]:


time_stamps = [file_name[6:21] for file_name in data_files]


# In[4]:


properties_list = ['TempBottomControlSection_AVG', 'TempBottomPowerSection_AVG', 'TurbineState_AVG']
machine_list = ['MK014', 'MK015', 'MK016']


# In[5]:


sliced_file_list = data_files[-1100: -1098]


# In[6]:


time_stamp_list = pd.to_datetime([file_name[6:21] for file_name in sliced_file_list])


# In[7]:


for csv_file in sliced_file_list:
    temp_frame = pd.read_csv(join(data_path, csv_file))
    temp_frame.index.name = 'Machine'
    temp_frame.columns.name = 'Properties'
    frame_list.append(temp_frame.set_index('Machine').loc[machine_list][properties_list].as_matrix())


# In[8]:


data_3d_array = np.array(frame_list)


# In[9]:


data_3d_array


# In[10]:


xr_data = xr.Dataset()


# In[11]:


for prop in properties_list:
    xr_data[prop] = (('time_stamp', 'machine'), data_3d_array[:, :, properties_list.index(prop)])

xr_data.coords['time_stamp'] = time_stamp_list
xr_data.coords['machine'] = machine_list


# In[12]:


mask = xr_data['TurbineState_AVG'] != 9001


# In[13]:


xr_data['TempBottomControlSection_AVG'].where(mask)


# In[ ]:




