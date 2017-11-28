
# coding: utf-8

# In[1]:


from datetime import timedelta, datetime

limiting_number = 5
start = data_set['Timestamp'].iloc[0]
time_obj = datetime.strptime(start, '%Y-%m-%dT%H:%M:%S') + timedelta(minutes=120)
plus_time = time_obj.strftime('%Y-%m-%dT%H:%M:%S')

curtailments_list = []

while plus_time < data_set['Timestamp'].iloc[-1]:
#     plt.figure(figsize=(30, 15))
    part_data = data_set[(data_set['Timestamp'] > start) & (data_set['Timestamp'] < plus_time)]
#     print part_data
#     print '--------------------------------------------------------'
#     plt.scatter(part_data['scaled_WindSpeed'], part_data['discrete_ActivePower'])
#     plt.show()
    time_obj = datetime.strptime(start, '%Y-%m-%dT%H:%M:%S') + timedelta(minutes=10)
    start = time_obj.strftime('%Y-%m-%dT%H:%M:%S')
    plus_time = (time_obj + timedelta(minutes=130)).strftime('%Y-%m-%dT%H:%M:%S')
    for group in part_data.groupby('discrete_ActivePower'):
        if group[1].shape[0] > limiting_number and group[0] > 10 and group[0] < 90:
            curtailments_list.append(group[1])

plt.figure(figsize=(30, 15))
plt.scatter(data_set['scaled_WindSpeed'], data_set['scaled_ActivePower'], s=np.pi*3*3)
curtailments_data_set = pd.concat(curtailments_list)
plt.scatter(curtailments_data_set['scaled_WindSpeed'], curtailments_data_set['scaled_ActivePower'], c='#000000', s=np.pi*3*3)

plt.show()

