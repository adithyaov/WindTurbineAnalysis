
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt_obj

fig_obj = plt_obj.figure(figsize = (25, 10))
axes_obj = fig_obj.add_subplot(111)

def my_plotter(axes_obj, x, y, param_dict):
    out = axes_obj.plot(x, y, **param_dict)
    return out

my_plotter(
    axes_obj,
    xr_data['TempBottomControlSection_AVG'].loc[:, 'MK014'].time_stamp,
    xr_data['TempBottomControlSection_AVG'].loc[:, 'MK014'],
    {
        'linestyle': ' ',
        'marker': '.'
    }
)
my_plotter(
    axes_obj,
    xr_data['TempBottomControlSection_AVG'].loc[:, 'MK015'].time_stamp,
    xr_data['TempBottomControlSection_AVG'].loc[:, 'MK015'],
    {
        'linestyle': ' ',
        'marker': '.'
    }
)

import matplotlib.dates as mdates
#import datetime as dt

axes_obj.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
axes_obj.set_xticks(pd.date_range(
    time_stamp_list[0],
    #time_stamp_list[-1] + dt.timedelta(minutes=10),
    time_stamp_list[-1],
    freq='10min'
), minor=False)
axes_obj.tick_params(
    axis='x',
    which='major',
    pad=5,
    labelsize=8,
    
)
axes_obj.minorticks_on
plt_obj.xticks(rotation=90)
plt_obj.grid(True)

plt_obj.show()


# In[ ]:




