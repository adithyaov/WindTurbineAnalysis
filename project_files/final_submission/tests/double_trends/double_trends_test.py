
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.cluster import *


# In[2]:


data_set = pd.read_csv('/home/mytrah-pc/Mytrah_Adithya/data_turbine/ScatterData_Mokal_MK014.csv')
data_set.drop_duplicates(subset=['ActivePower', 'WindSpeed'], keep=False)


# In[3]:


def process_dbscan(data_set, dict_params_dbscan, scale_factor_wind_speed=4.0/3, scale_factor_active_power=1):

    # Only take the required data, only positive power points
    required_data = data_set[data_set['ActivePower'] > 0.08][['ActivePower', 'WindSpeed', 'Timestamp']].copy()

    # Get some params
    max_wind_speed = required_data['WindSpeed'].max()
    min_wind_speed = required_data['WindSpeed'].min()
    max_active_power = required_data['ActivePower'].max()
    min_active_power = required_data['ActivePower'].min()
    normalization_factor_wind_speed = max_wind_speed - min_wind_speed
    normalization_factor_active_power = max_active_power - min_active_power

    # Add some columns to required data
    required_data['ScaledWindSpeed'] = ((required_data['WindSpeed'] - min_wind_speed) * scale_factor_wind_speed)                                         / normalization_factor_wind_speed
    required_data['ScaledActivePower'] = ((required_data['ActivePower'] - min_active_power) * scale_factor_active_power)                                         / normalization_factor_active_power
    
    # Cluster default Number
    required_data['cluster_number'] = -2
    
    # Clustring Algorithm
    clustering_algorithm = DBSCAN(**dict_params_dbscan)
    
    # run the algorithm
    required_data['cluster_number'] = clustering_algorithm.fit_predict(required_data[['ScaledWindSpeed',                                                                                       'ScaledActivePower']])

    return required_data.copy()
    
def curve_filter(data_set, return_columns, push_down_value, curve_straighten_value, push_right_value):

    # Find the cluster with max number of elements
    max_elements_cluster_number = -2
    max_elements = 0
    for group in data_set.groupby('cluster_number'):
        if(group[1].shape[0] > max_elements):
            max_elements = group[1].shape[0]
            max_elements_cluster_number = group[0]

    
    # define limits
    scales_active_power_limits = (0.03, 0.97)
    
    # Apply some filters to get the required
    global_high_cluster = data_set[data_set['ScaledActivePower'] > 0.99]   
    max_elements_cluster = data_set[data_set['cluster_number'] == max_elements_cluster_number]
    filter_limits_active_power = max_elements_cluster[                                     (max_elements_cluster['ScaledActivePower'] > scales_active_power_limits[0])                                      & (max_elements_cluster['ScaledActivePower'] < scales_active_power_limits[1])]


    # Customised Sigmoid function for curve fitting
    def custom_sigmoid(x, a, b):
        return 1/(1 + np.exp(-a * x + b))
    
    optimize_on = pd.concat([filter_limits_active_power, global_high_cluster])
    
    params_optimal, params_covariance = curve_fit(custom_sigmoid, optimize_on['ScaledWindSpeed'],                                                   optimize_on['ScaledActivePower'])
    
    ################################### Filter elements below the Curve ############################
    filtered_data = data_set[data_set['ScaledActivePower'] > (custom_sigmoid(data_set['ScaledWindSpeed'],                                                                         params_optimal[0] + curve_straighten_value,                                                                         params_optimal[1] + push_right_value)                                                                         - push_down_value)]
    
    points_above_the_curve = filtered_data[return_columns].copy()
    
    return points_above_the_curve


# In[4]:


points_above_curve = curve_filter(
    process_dbscan(
        data_set=data_set,
        dict_params_dbscan={
            'eps': 0.3/20,
            'min_samples': 15
        },
        scale_factor_wind_speed=4.0/3,
        scale_factor_active_power=1
    ),
    return_columns=['ActivePower', 'WindSpeed', 'Timestamp', 'ScaledActivePower'],
    push_down_value=0.1,
    curve_straighten_value=0,
    push_right_value=1.3
)

work_data = points_above_curve[(points_above_curve['ScaledActivePower'] > 0.25) & (points_above_curve['ScaledActivePower'] < 0.85)].copy()


# In[5]:


def rounding_method(number):
    multiply_factor = 100
    int_format = int(number * multiply_factor)
    round_off_by_factor = 2
    delta = int_format % round_off_by_factor
    return int_format + round_off_by_factor - delta

work_data['trend_indexer'] = work_data['ScaledActivePower'].apply(rounding_method)


# In[6]:


import matplotlib.pyplot as plt
from scipy.stats import norm


# In[7]:


# hist = None
# mid = -10
# tot_scr = 0
# tot_ex = 0
# for group in work_data.groupby('trend_indexer'):
#     plt.hist(group[1]['WindSpeed'])
#     plt.show()
#     hist = np.histogram(group[1]['WindSpeed'])
#     mid = 0
#     score = 0
#     for i in range(hist[0].shape[0]):
#         mid = mid + hist[0][i] * ((hist[1][i] + hist[1][i + 1]) / 2)
#     print (mid / hist[0].sum())
#     for i in range(hist[0].shape[0]):
#         score = score + norm((mid / hist[0].sum()), 1).pdf((hist[1][i] + hist[1][i + 1]) / 2) * hist[0][i]
#     print score / hist[0].sum()
#     tot_scr = tot_scr + (score / hist[0].sum())
#     tot_ex = tot_ex + 1
#     plt.figure(figsize=(20, 10))
#     plt.scatter(work_data['WindSpeed'], work_data['trend_indexer'])
#     plt.scatter(group[1]['WindSpeed'], [group[0] for _ in range(group[1].shape[0])])
#     plt.show()

# print '-------------------------------------------------------------------------------------------'
# print tot_scr / tot_ex


# In[8]:


import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.cluster import *

def process_dbscan(data_set, dict_params_dbscan, scale_factor_wind_speed=4.0/3, scale_factor_active_power=1):

    # Only take the required data, only positive power points
    required_data = data_set[data_set['ActivePower'] > 0.08][['ActivePower', 'WindSpeed', 'Timestamp']].copy()

    # Get some params
    max_wind_speed = required_data['WindSpeed'].max()
    min_wind_speed = required_data['WindSpeed'].min()
    max_active_power = required_data['ActivePower'].max()
    min_active_power = required_data['ActivePower'].min()
    normalization_factor_wind_speed = max_wind_speed - min_wind_speed
    normalization_factor_active_power = max_active_power - min_active_power

    # Add some columns to required data
    required_data['ScaledWindSpeed'] = ((required_data['WindSpeed'] - min_wind_speed) * scale_factor_wind_speed)                                         / normalization_factor_wind_speed
    required_data['ScaledActivePower'] = ((required_data['ActivePower'] - min_active_power) * scale_factor_active_power)                                         / normalization_factor_active_power
    
    # Cluster default Number
    required_data['cluster_number'] = -2
    
    # Clustring Algorithm
    clustering_algorithm = DBSCAN(**dict_params_dbscan)
    
    # run the algorithm
    required_data['cluster_number'] = clustering_algorithm.fit_predict(required_data[['ScaledWindSpeed',                                                                                       'ScaledActivePower']])

    return required_data.copy()

def curve_filter(data_set, return_columns, push_down_value, curve_straighten_value, push_right_value):

    # Find the cluster with max number of elements
    max_elements_cluster_number = -2
    max_elements = 0
    for group in data_set.groupby('cluster_number'):
        if(group[1].shape[0] > max_elements):
            max_elements = group[1].shape[0]
            max_elements_cluster_number = group[0]

    
    # define limits
    scales_active_power_limits = (0.03, 0.97)
    
    # Apply some filters to get the required
    global_high_cluster = data_set[data_set['ScaledActivePower'] > 0.99]   
    max_elements_cluster = data_set[data_set['cluster_number'] == max_elements_cluster_number]
    filter_limits_active_power = max_elements_cluster[                                     (max_elements_cluster['ScaledActivePower'] > scales_active_power_limits[0])                                      & (max_elements_cluster['ScaledActivePower'] < scales_active_power_limits[1])]


    # Customised Sigmoid function for curve fitting
    def custom_sigmoid(x, a, b):
        return 1/(1 + np.exp(-a * x + b))
    
    optimize_on = pd.concat([filter_limits_active_power, global_high_cluster])
    
    params_optimal, params_covariance = curve_fit(custom_sigmoid, optimize_on['ScaledWindSpeed'],                                                   optimize_on['ScaledActivePower'])
    
    ################################### Filter elements below the Curve ############################
    filtered_data = data_set[data_set['ScaledActivePower'] > (custom_sigmoid(data_set['ScaledWindSpeed'],                                                                         params_optimal[0] + curve_straighten_value,                                                                         params_optimal[1] + push_right_value)                                                                         - push_down_value)]
    
    points_below_the_curve = filtered_data[return_columns].copy()
    
    return points_below_the_curve


# In[9]:


from datetime import timedelta, datetime


from os import listdir
from os.path import isfile, join

data_path = '/home/mytrah-pc/Mytrah_Adithya/data_turbine/'
for f in listdir(data_path):


    start = data_set['Timestamp'][0]
    time_obj = datetime.strptime(start, '%Y-%m-%dT%H:%M:%S') + timedelta(days=2)
    plus_day = time_obj.strftime('%Y-%m-%dT%H:%M:%S')

    def a_sig(x, a, b):
        return 1/(1 + np.exp(-a * x + b))

    plt.figure(figsize=(30, 15))

    data_set = pd.read_csv(data_path + f)

    testg = process_dbscan(
            data_set=data_set,
            dict_params_dbscan={
                'eps': 0.3/20,
                'min_samples': 13
            },
            scale_factor_wind_speed=4.0/3,
            scale_factor_active_power=1
        )
    max_wind_speed = testg['WindSpeed'].max()
    min_wind_speed = testg['WindSpeed'].min()
    max_active_power = testg['ActivePower'].max()
    min_active_power = testg['ActivePower'].min()
    normalization_factor_wind_speed = max_wind_speed - min_wind_speed
    normalization_factor_active_power = max_active_power - min_active_power

    testg['ScaledWindSpeed'] = ((testg['WindSpeed'] - min_wind_speed) * 1)                                             / normalization_factor_wind_speed
    testg['ScaledActivePower'] = ((testg['ActivePower'] - min_active_power) * 1)                                             / normalization_factor_active_power

    po = []
    while plus_day < testg['Timestamp'].iloc[-1]:
        test = testg[(testg['Timestamp'] > start) & (testg['Timestamp'] < plus_day)]

        start = plus_day
        time_obj = datetime.strptime(start, '%Y-%m-%dT%H:%M:%S') + timedelta(days=2)
        plus_day = time_obj.strftime('%Y-%m-%dT%H:%M:%S')

        if test.shape[0] < 30:
            continue
        else:
            pass


        params_optimal, params_covariance = curve_fit(a_sig, test['ScaledWindSpeed'],                                                       test['ScaledActivePower'])

        print params_optimal, params_covariance
        po.append(params_optimal)
        x = np.linspace(0, 1)
        y = a_sig(x, params_optimal[0], params_optimal[1])

        plt.plot(x, y)
        plt.scatter(test['ScaledWindSpeed'], test['ScaledActivePower'])

    print f
    mean_vals = np.mean(po, axis=0)
    std_vals = np.std(po, axis=0)

    cmp1 = mean_vals + 0.8 * std_vals
    cmp2 = mean_vals - 0.8 * std_vals
    new_pos = []
    
    for p in po:
        con1 = p < cmp1
        con2 = p > cmp2
        print con1, con2
        if con1[0] and con2[0] and con1[1] and con2[1]:
            new_pos.append(p)

    
    new_mean_vals = np.mean(new_pos, axis=0)
    new_std_vals = np.std(new_pos, axis=0)
    
    x = np.linspace(0, 1)
    y = a_sig(x, mean_vals[0], mean_vals[1] + 1.5 * std_vals[1])
    plt.plot(x, y, 'o', color='#000000')
    
    x = np.linspace(0, 1)
    y = a_sig(x, mean_vals[0], mean_vals[1] - 1.5 * std_vals[1])
    plt.plot(x, y, 'o', color='#000000')

    x = np.linspace(0, 1)
    y = a_sig(x, new_mean_vals[0], new_mean_vals[1] + 2 * new_std_vals[1])
    plt.plot(x, y, 'o', color='#FF0000')
    
    x = np.linspace(0, 1)
    y = a_sig(x, new_mean_vals[0], new_mean_vals[1] - 2 * new_std_vals[1])
    plt.plot(x, y, 'o', color='#FF0000')
    
    plt.show()


# In[ ]:




