
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join


# In[2]:


data_path = '/home/mytrah-pc/Data/Mokal/'
data_files = [f for f in listdir(data_path) if (f[-4:] == '.csv')]
data_files = sorted(data_files)
num_data_files = len(data_files)


# In[3]:


frame_list = []
sliced_file_list = data_files[-1100: -1000]

for csv_file in sliced_file_list:
    temp_frame = pd.read_csv(join(data_path, csv_file))
    temp_frame['file_name'] = pd.Series(csv_file, index = temp_frame.index)
    frame_list.append(temp_frame)


# In[4]:


import matplotlib.pyplot as pyplot


# In[5]:


def plot_graph(**kwargs):

    if 'figure_size' in kwargs:
        pyplot.figure(figsize = ((kwargs['figure_size']['width'], kwargs['figure_size']['height'])))
    else:
        pyplot.figure(figsize = (20, 10))
    
    if 'plot_data' in kwargs:
        for plot_data_item in kwargs['plot_data']:
            pyplot.plot(
                plot_data_item['x'], plot_data_item['y'],
                **plot_data_item['extra_data']
            )
    
    if 'x_label' in kwargs:
        pyplot.xlabel(kwargs['x_label'])
    else:
        pyplot.xlabel('X-axis')
        
    if 'y_label' in kwargs:
        pyplot.ylabel(kwargs['y_label'])
    else:
        pyplot.ylabel('Y-axis')
        
    if 'plot_title' in kwargs:
        pyplot.title(kwargs['plot_title'])
    else:
        pyplot.title('Plot')
        
    if 'grid' in kwargs:
        if(kwargs['grid'] == True):
            pyplot.grid(True)
            
    if 'plot_text' in kwargs:
        for plot_text_item in kwargs['plot_text']:
            pyplot.text(plot_text_item['x'], plot_text_item['y'], plot_text_item['text'],
                        color = (plot_text_item['color'] if ('color' in plot_text_item.keys()) else 'b'),
                        fontsize = (plot_text_item['fontsize'] if ('fontsize' in plot_text_item.keys()) else 10))

    if 'x_ticks' in kwargs:
        pyplot.xticks(range(len(kwargs['x_ticks']['ticks'])), kwargs['x_ticks']['ticks'], rotation = kwargs['x_ticks']['rotation'])

    return pyplot
    


# In[6]:


data_3d = pd.concat(frame_list, axis = 0)


# In[7]:


def immediate_non_uniform_changes(data_3d, column_list, machine_list):
    data_hold = data_3d.set_index('Machine')
    result = []
    for column_name in column_list:
        for machine in machine_list:
            temp_data = data_hold.loc[machine]

            find_dev = temp_data.copy()[[column_name]]
            find_dev['prev_' + column_name] = find_dev.shift(1)[column_name]
            find_dev['next_' + column_name] = find_dev.shift(-1)[column_name]
            find_dev['std'] = find_dev.std(axis = 1)
            t_mean = find_dev[['std']].mean()
            t_std = find_dev[['std']].std()
            std_multiplier = 3
            limit = (t_mean - std_multiplier * t_std, t_mean + std_multiplier * t_std)
            bool_cols = (find_dev['std'] < limit[0]['std']) | (find_dev['std'] > limit[1]['std'])

            result.append(
                {
                    'machine': machine,
                    'column_name': column_name,
                    'volatile_points': list(temp_data[list(bool_cols)]['file_name'])
                }
            )
    return result


# In[8]:


immediate_non_uniform_changes(data_3d, ['TempBottomControlSection_AVG'], ['MK014', 'MK163'])


# In[9]:


def get_data_by_file_name(file_names, data_to_filter):
    return data_to_filter.set_index(['file_name']).loc[file_names].reset_index()

def get_data_by_machine(machine_names, data_to_filter):
    return data_to_filter.set_index(['Machine']).loc[machine_names].reset_index()


# In[10]:


from random import randint, choice


# In[ ]:


machine_list = ['B-' + str(i) for i in range(509, 540)]
machine_str = 'MK014.MK015.MK016.MK017.MK021.MK039.MK040.MK042.MK043.MK066.MK067.MK068.MK069.MK092.MK093.MK094.MK161.MK163.MK164.MK165'
machine_list_mokal = machine_str.split('.')


# In[ ]:


def statistic_op(file_list, machine_list, column_names, apply_functions, data_3d, **function_args):
    data_for_statistics = data_3d.set_index(['Machine']).loc[machine_list].set_index('file_name').loc[file_list]
    
    return_dict = {}
    for column_name in column_names:
        return_dict[column_name] = {}
        for file_name in file_list:
            return_dict[column_name][file_name] = {}
            for function in apply_functions:
                return_dict[column_name][file_name][function] = function_args[function](list(data_for_statistics.loc[file_name][column_name]))

    return return_dict


# In[ ]:


function_args = {
    'mean': np.nanmean,
    'std': np.nanstd,
    'min': np.nanmin,
    'max': np.nanmax
}


# In[ ]:


def time_serial(machine_list, file_list, column_name, data_3d, **kwargs):
    style_dict = {}

    for machine in machine_list:
        style_dict[machine] = {}
        style_dict[machine]['color'] = '#' + (str(randint(0,999999)) + '00000')[0:6]
        style_dict[machine]['line_style'] = choice(['--', '-.', '-', ':'])
    
    num_files = len(file_list)
    x_range_list = range(num_files)

    return {
        'plot_object': plot_graph(
            plot_data = [
                {
                    'y': list(get_data_by_machine(machine_name, data_3d)[column_name]),
                    'x': x_range_list,
                    'extra_data': {
                        'color': style_dict[machine_name]['color'],
                        'linestyle': style_dict[machine_name]['line_style'],
                        'label': machine_name,
                        'marker': '.'
                    }
                }
                for machine_name in machine_list
            ],
            grid = True,
            y_label = column_name,
            x_label = 'Ascending File names',
            plot_title = column_name + ' per Machine',
            x_ticks = {
                'ticks': file_list,
                'rotation': 90
            },
            **kwargs
        ),
        'style_dict': style_dict        
    }

def make_additional_plots(plot_object, plot_data):
    for plot_data_item in plot_data:
        plot_object.plot(
            plot_data_item['x'], plot_data_item['y'],
            **plot_data_item['extra_data']
        )


# In[ ]:


def plot_permissable_limits(plot_object, data_statistics, column_name):
    upper_std_limit = [data_statistics[column_name][item]['mean'] + 3*data_statistics[column_name][item]['std']
                       for item in sorted(data_statistics[column_name].keys())]
    lower_std_limit = [data_statistics[column_name][item]['mean'] - 3*data_statistics[column_name][item]['std']
                       for item in sorted(data_statistics[column_name].keys())]


    make_additional_plots(plot_object, [
        {
            'y': upper_std_limit,
            'x': range(len(upper_std_limit)),
            'extra_data': {
                'label': 'Stat Line',
                'marker': 'o',
                'linestyle': '',
                'label': 'Upper Limit'
            }
        },
        {
            'y': lower_std_limit,
            'x': range(len(lower_std_limit)),
            'extra_data': {
                'label': 'Stat Line',
                'marker': 'o',
                'linestyle': '',
                'label': 'Lower Limit'
            }
        }
    ])


# In[ ]:


def final_renders(plot_object, **kwargs):
     plot_object.legend(bbox_to_anchor=(1.02, 1), loc = 2, borderaxespad = 0.)


# In[ ]:


def clean_data(data_3d, column_name):
    cleaned_data = data_3d.copy()
    cleaned_data.loc[(data_3d.TurbineState_AVG == 9001), column_name] = np.nan
    return cleaned_data


# In[ ]:


def visualize_v1(machine_list, file_list, column_name, data_3d, **kwargs):
    cleaned_data = clean_data(data_3d, column_name)
    data_statistics = statistic_op(file_list, machine_list, [column_name], ['mean', 'std', 'min', 'max'], cleaned_data, **function_args)
    
    time_object = time_serial(
        machine_list, file_list, column_name, cleaned_data,
        **kwargs
    )
    
    plot_permissable_limits(time_object['plot_object'], data_statistics, column_name)
    
    final_renders(time_object['plot_object'])
    time_object['plot_object'].show()


# In[ ]:


for x in list(data_3d.columns):
    if 'AVG' in x:
        print x


# In[ ]:


visualize_v1(machine_list_mokal, sliced_file_list, 'TempBottomControlSection_AVG', data_3d, figure_size = {
    'width': 80,
    'height': 15
})


# In[ ]:




