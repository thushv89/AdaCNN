import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from math import ceil,floor
from operator import add


log_dir = 'plot_train_test_accuracies'

file_names = [os.path.join(log_dir,'Error-cifar100-inc.log'),
              os.path.join(log_dir, 'Error-cifar100-rigid-pool.log')]
legends = [['Train ReAd-CNN','Test Read-CNN'],['Train Rigid-CNN-B','Test Rigid-CNN-B']]
y_labels = ['Train Accuracies (%)','Test Accuracies (%)']
data_x = []

columns_to_extract = [2,4]
data_y = [[[] for _ in range(len(columns_to_extract))] for _ in range(2)]

fig, axarr = plt.subplots(2,1)

plot_x_range = range(60000,70000,100)

for f_i, file_name in enumerate(file_names):
    with open(file_name, 'r') as f:
        for l_id, line in enumerate(f):
            if l_id==0:
                continue
            else:
                line_tokens = line.split(',')
                line_id = int(line_tokens[0])

                if line_id not in plot_x_range:
                    continue
                # no need to round numbers
                if f_i==0:
                    data_x.append(line_id)

                for c_i in range(len(line_tokens)):
                    if c_i in columns_to_extract:
                        val = float(line_tokens[c_i])
                        data_y[f_i][columns_to_extract.index(c_i)].append(val)

colors = ['r','b']
for f_i in range(len(file_names)):
    for col_i in range(len(columns_to_extract)):
        print(len(data_x))
        print(len(data_y[f_i][col_i]))

        #plt.plot(data_x,[0 for _ in data_x],c='gray',linestyle='--')
        axarr[col_i].plot(data_x,data_y[f_i][col_i],label=legends[f_i][col_i],linewidth=2,c=colors[f_i])
        axarr[col_i].fill_between(data_x,
                                  [np.mean(data_y[f_i][col_i])-np.std(data_y[f_i][col_i]) for _ in data_x],
                                  [np.mean(data_y[f_i][col_i])+np.std(data_y[f_i][col_i]) for _ in data_x],alpha=0.4,facecolor=colors[f_i])
        #axarr[col_i].title('Long-Term Discounted Reward for Different Layers of VGG-13',fontsize=22)
        axarr[1].set_xlabel('Iterations',fontsize=18)
        axarr[col_i].set_ylabel(y_labels[col_i],fontsize=18)
        axarr[col_i].legend(ncol=2, loc='upper center',bbox_to_anchor=(0.5,1.02))

axarr[0].set_ylim([20,100])
axarr[1].set_ylim([25,45])
plt.suptitle('Test and Train Accuracy behavior for CIFAR-100 Dataset',fontsize=24)
plt.show()