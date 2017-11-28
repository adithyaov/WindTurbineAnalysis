
# coding: utf-8

# In[ ]:


# style_dict = {}

# for machine in machine_list:
#     style_dict[machine] = {}
#     style_dict[machine]['color'] = '#' + (str(randint(0,999999)) + '00000')[0:6]
#     style_dict[machine]['line_style'] = choice(['--', '-.', '-', ':'])

# x_range_list = range(num_sliced_files)

# plot_add = plot_graph(
#     plot_data = [
#         {
#             'y': list(get_data_by_machine(machine_name)['Amb.NacelleTemp_AVG']),
#             'x': x_range_list,
#             'color': style_dict[machine_name]['color'],
#             'line_style': style_dict[machine_name]['line_style']
#         }
#         for machine_name in machine_list
#     ],
#     grid = True,
#     y_label = 'Amb.NacelleTemp_AVG',
#     x_label = 'Ascending file names',
#     plot_title = 'Amb.NacelleTemp_AVG per machine for 90 minutes',
#     x_ticks = {
#         'ticks': sliced_file_list,
#         'rotation': 90
#     }
# )

#plot_add.show()

