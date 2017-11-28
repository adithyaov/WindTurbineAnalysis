
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from os import listdir, mkdir
from os.path import isfile, join



data_path = '/home/mytrah-pc/Data/Burgula/'
save_path = '/home/mytrah-pc/Data/Burgula_extract'
properties_list = ['ActivePower_AVG', 'WindSpeed_AVG', 'TurbineState_AVG', 'TurbineState_SD']

machine_string = 'B-508B-509B-510B-511B-512B-513B-514B-515B-516B-517B-518B-519B-520B-521B-522B-523B-524B-525B-526B-527B-528B-529B-530B-531B-532B-533B-534B-535B-536B-537B-538B-539B-540B-541B-542B-543B-544B-545B-546B-547B-548B-549B-550B-551'
machine_list = machine_string.split('B')
machine_list = ['B' + x for x in machine_list]
machine_list = machine_list[1:]

number_of_recent_files_to_combine = 5000
timestamp_slice_offset = 8
filter_TurbineState = 100

data_files = [f for f in listdir(data_path) if (f[-4:] == '.csv')]
data_files = sorted(data_files)
num_data_files = len(data_files)
dataframe_dict = {}

sliced_file_list = data_files[-number_of_recent_files_to_combine:]

for mach_name in machine_list:
    dataframe_dict[mach_name] = {}
    dataframe_dict[mach_name]['frame_list'] = []
    dataframe_dict[mach_name]['time_stamp_list'] = []

for csv_file in sliced_file_list:
    temp_frame = pd.read_csv(join(data_path, csv_file))
    temp_frame = temp_frame[(temp_frame['TurbineState_AVG'] == filter_TurbineState) & (temp_frame['TurbineState_SD'] == 0)]
    temp_frame = temp_frame.set_index('Machine')
    
    for mach_name in machine_list:
        if mach_name in temp_frame.index.values:
            dataframe_dict[mach_name]['frame_list'].append(temp_frame.loc[mach_name][properties_list])
            dataframe_dict[mach_name]['time_stamp_list'].append(csv_file[timestamp_slice_offset:timestamp_slice_offset + 15])

mkdir(save_path)
for mach_name in dataframe_dict.keys():
    if len(dataframe_dict[mach_name]['frame_list']) != 0:
        final_df = pd.concat(dataframe_dict[mach_name]['frame_list'], axis=1).T
        final_df['Timestamp'] = dataframe_dict[mach_name]['time_stamp_list']

        required_df = final_df.reset_index()

        del required_df['index']

        required_df = required_df.rename(columns={
            'ActivePower_AVG': 'ActivePower',
            'WindSpeed_AVG': 'WindSpeed',
            'TurbineState_AVG': 'TurbineState'
        })

        new_name = mach_name + '[' + dataframe_dict[mach_name]['time_stamp_list'][0] + '][' + dataframe_dict[mach_name]['time_stamp_list'][-1] + '].csv'
        required_df.to_csv(save_path + '/' + new_name)

