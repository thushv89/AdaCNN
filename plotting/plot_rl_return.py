import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from math import ceil,floor
from operator import add


log_dir = 'plot_rl_return'

file_name = os.path.join(log_dir,'predicted_q_growth.log')
legends = []
data_x = []

n_layers = 13+3
data_y = [[] for _ in range(n_layers)]
selected_layer_ids = [3,4,5,9,12,14]
with open(file_name, 'r') as f:
    for l_id, line in enumerate(f):
        if l_id==0:
            legends = line.split(',')[1:]
        else:
            line_tokens = line.split(',')
            line_id = int(line_tokens[0])

            # rounding up the number to nearest 10
            data_x.append((ceil(line_id / 10.0) * 10)*50)

            for c_i in range(16):
                data_y[c_i].append(float(line_tokens[c_i + 1]))



for lyr_id in selected_layer_ids:
    print(lyr_id)
    print(len(data_x))
    print(len(data_y[lyr_id]))
    plt.plot(data_x,[0 for _ in data_x],c='gray',linestyle='--')
    plt.plot(data_x,data_y[lyr_id],label=legends[lyr_id],linewidth=2)
    plt.title('Long-Term Discounted Reward for Different Layers of VGG-13',fontsize=22)
    plt.xlabel('Iterations',fontsize=18)
    plt.ylabel('$Q(s_t,a_t,DQN)$',fontsize=18)

plt.legend(loc=3)
plt.show()