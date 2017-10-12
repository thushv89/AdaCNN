import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import os
from math import ceil,floor
from operator import add

# All the resource files needed should be in here
log_dir = 'plot_class_distributions'

# ======================================================================
#                       Data distribution
# ======================================================================

fig, axarr = plt.subplots(1,3)

file_names = [os.path.join(log_dir, 'class_distribution_cifar10.log'),
              os.path.join(log_dir, 'class_distribution_cifar100.log'),
              os.path.join(log_dir, 'class_distribution_imagenet250.log')]

num_classes = [10,100,250]
num_classes_plot = [10,100,50]

iterations_per_epoch = [5000,10000,10000]
num_tasks = [2,4,2]
tasks_read_so_far = [0,0,0]
begin_index_for_task = [0,0,0]

dist_y = [[[] for _ in range(num_classes_plot[f_i])] for f_i in range(3)]
dist_x = [[] for _ in range(3)]
ax_titles = ['CIFAR-10','CIFAR-100','Imagenet-250 (50 Classes)']
sup_title = 'Class Distribution of a Single Epoch for Different Datasets'
#dist_legends = ['Class %d'%i for i in range(num_classes)]


for f_i,file_name in enumerate(file_names):
    prev_line_id = 0
    with open(file_name,'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            else:
                line_tokens = line.split(',')
                line_id = int(line_tokens[0])

                if line_id<prev_line_id:
                    begin_index_for_task[f_i] += iterations_per_epoch[f_i]//num_tasks[f_i]
                    tasks_read_so_far[f_i] += 1

                # This will be true when you read data for a full epoch
                if tasks_read_so_far[f_i]==num_tasks[f_i]:
                    break

                # rounding up the number to nearest 10
                dist_x[f_i].append(begin_index_for_task[f_i]+ceil(line_id/10.0)*10)

                for c_i in range(num_classes_plot[f_i]):
                    dist_y[f_i][c_i].append(float(line_tokens[c_i+1]))

            prev_line_id = line_id

#normalize imagenet data

imagenet_dist = np.asarray(dist_y[2],dtype=np.float32)
sum_column = np.sum(imagenet_dist,axis=0).reshape(1,-1)
imagenet_dist = np.divide(imagenet_dist,sum_column)
print(imagenet_dist.shape)
dist_y[2] = imagenet_dist.tolist()

assert np.allclose(np.sum(np.asarray(dist_y[2]),axis=0),np.ones_like(sum_column))

for f_i in range(3):
    prev_dist_y = None
    cmap = matplotlib.cm.get_cmap('jet')
    used_bins = []
    for c_i in range(num_classes_plot[f_i]):
        rand_color_bins = np.arange(0,1.0,1.0/num_classes_plot[f_i])
        rand_color = np.random.choice(rand_color_bins)
        # Make sure we dont get same color for two lines
        while rand_color in used_bins:
            rand_color = np.random.choice(rand_color_bins)

        line_color = cmap(rand_color)
        fill_color = cmap(rand_color)
        if not prev_dist_y:
            axarr[f_i].plot(dist_x[f_i], dist_y[f_i][c_i],c=line_color)
            axarr[f_i].fill_between(dist_x[f_i], 0, dist_y[f_i][c_i], facecolor=fill_color)
            prev_dist_y = dist_y[f_i][c_i]
        else:
            axarr[f_i].plot(dist_x[f_i],map(add,dist_y[f_i][c_i],prev_dist_y),c=line_color)
            axarr[f_i].fill_between(dist_x[f_i], prev_dist_y, map(add,dist_y[f_i][c_i],prev_dist_y), facecolor=fill_color)
            prev_dist_y = map(add, dist_y[f_i][c_i], prev_dist_y)
        used_bins.append(rand_color)
    axarr[f_i].set_xlim([0, dist_x[f_i][-1]])
    axarr[f_i].set_ylim([0, 1.0])
    axarr[f_i].set_title(ax_titles[f_i])
    axarr[f_i].set_xlabel('Iterations',fontsize=18)
    axarr[0].set_ylabel('Class distribution \nover time (Stacked)',fontsize=18)
plt.suptitle(sup_title,fontsize=24)
#ax3.set_xlabel('Time ($t$)',fontsize=fontsize_label)
#ax3.set_ylabel('Proportion of instances of each class',fontsize=fontsize_label)
#ax3.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
#ax3.set_title('Class distribution over time ($t$)',fontsize=fontsize_title,y=1.19)
#ax3.legend(fontsize=fontsize_legend_small,loc=3,bbox_to_anchor=(0.1, 1.02),ncol=5)

fig.subplots_adjust(wspace=0.15,hspace=0.15,bottom=0.15,top=0.85,right=0.97,left=0.07)

plt.show()