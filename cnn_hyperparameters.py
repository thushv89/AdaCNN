import logging
import os

batch_size = 128  # number of datapoints in a single batch
# stationary 0.1 non-stationary 0.01
start_lr = 0.01
min_learning_rate = 0.0001
decay_learning_rate = True
decay_rate = 0.5
dropout_rate = 0.5
in_dropout_rate = 0.2
use_dropout = True
use_loc_res_norm = False
# keep beta small (0.2 is too much >0.002 seems to be fine)
include_l2_loss = True
beta = 1e-5
check_early_stopping_from = 5
accuracy_drop_cap = 3
iterations_per_batch = 1
epochs = 5
final_2d_width = None

lrn_radius = 5
lrn_alpha = 0.0001
lrn_beta = 0.75


research_parameters = {
    'save_train_test_images': False, # If true will save train and test images randomly (to make sure images are correctly read)
    'log_class_distribution': True, 'log_distribution_every': 128, # log distribution of data (useful for generating data distribution over time curves)
    'adapt_structure': True, # Enable AdaCNN behavior
    'hard_pool_acceptance_rate': 0.1, # Probability with which data is accepted in to the pool
    'replace_op_train_rate': 0.8,  # amount of batches from hard_pool selected to train
    'optimizer': 'Momentum', 'momentum': 0.9, 'pool_momentum': 0.0, # Two momentums one for data one for pool
    'use_custom_momentum_opt': True, # Use a custom implemented momentum (Tensorflow builtin optimizer doesnot support variable size tensors
    'remove_filters_by': 'Activation', # The criteria for removing filters (AdaCNN) set of minimum maximum mean activations
    'optimize_end_to_end': True, # if true functions such as add and finetune will optimize the network from starting layer to end (fulcon_out)
    'loss_diff_threshold': 0.02, # This is used to check if the loss reduction has stabalized
    'start_adapting_after': 500, # Acts as a warming up phase, adapting from the very begining can make CNNs unstable
    'debugging': True if logging_level == logging.DEBUG else False,
    'stop_training_at': 11000, # If needed to truncate training earlier
    'train_min_activation': False,
    'use_weighted_loss': True, # Weight the loss by the class distribution at the time
    'whiten_images': True, # Whiten images using batch mean and batch std for each batch
    'finetune_rate': 0.5, # amount of data from data pool used for finetuning
    'pool_randomize': True, # randomize the pool data when training with it
    'pool_randomize_rate': 0.25, # frequency the pool data randomized for
    'pooling_for_nonadapt': True,
    'hard_pool_max_threshold': 0.5, # when there's not much data use a higher pool accumulation rate
}


interval_parameters = {
    'history_dump_interval': 500,
    'policy_interval': 50,  # number of batches to process for each policy iteration
    'finetune_interval': 50,
    'test_interval': 100
}

# type of data training
datatype = 'imagenet-250'
behavior = 'stationary'

research_parameters['adapt_structure'] = False
research_parameters['pooling_for_nonadapt'] = True

if not (research_parameters['adapt_structure'] and research_parameters['pooling_for_nonadapt']):
    iterations_per_batch = 2

if research_parameters['adapt_structure']:
    epochs += 2  # for the trial one
    research_parameters['hard_pool_acceptance_rate'] *= 2.0

if behavior == 'non-stationary':
    include_l2_loss = False
    use_loc_res_norm = True
    lrn_radius = 5
    lrn_alpha = 0.0001
    lrn_beta = 0.75
    start_lr = 0.01
    decay_rate = 0.8
elif behavior == 'stationary':
    start_lr = 0.008
    include_l2_loss = True
    beta = 0.0005
    use_loc_res_norm = False
    decay_rate = 0.5
else:
    raise NotImplementedError

dataset_info = {'dataset_type': datatype, 'behavior': behavior}
dataset_filename, label_filename = None, None
test_dataset, test_labels = None, None

if datatype == 'cifar-10':
    image_size = 24
    num_labels = 10
    num_channels = 3  # rgb
    dataset_size = 50000
    use_warmup_epoch = True

    dataset_size = 1280000
    chunk_size = 51200

    interval_parameters['policy_interval'] = 24
    interval_parameters['finetune_interval'] = 24
    orig_finetune_interval = 50
    trial_phase_threshold = 1.0
    research_parameters['start_adapting_after'] = 1000
    pool_size = batch_size * 10 * num_labels
    test_size = 10000
    test_dataset_filename = 'data_non_station' + os.sep + 'cifar-10-test-dataset.pkl'
    test_label_filename = 'data_non_station' + os.sep + 'cifar-10-test-labels.pkl'

    if not research_parameters['adapt_structure']:
        # cnn_string = "C,3,1,128#C,3,1,128#C,3,1,128#P,3,2,0#C,3,1,256#Terminate,0,0,0"
        cnn_string = "C,3,1,96#C,3,1,96#C,3,1,96#P,3,2,0#C,3,1,192#C,3,1,192#C,3,1,192#PG,3,2,0#FC,2048,0,0#Terminate,0,0,0"

    else:
        cnn_string = "C,3,1,32#C,3,1,32#C,3,1,32#P,3,2,0#C,3,1,32#C,3,1,32#C,3,1,32#PG,3,2,0#FC,2048,0,0#Terminate,0,0,0"
        # cnn_string = "C,3,1,48#C,3,1,48#C,3,1,48#P,3,2,0#C,3,1,48#Terminate,0,0,0"
        filter_vector = [96, 96, 96, 0, 192, 192, 192]
        add_amount, remove_amount = 8, 4
        filter_min_threshold = 24

elif datatype == 'imagenet-250':
    image_size = 64
    num_labels = 250
    num_channels = 3  # rgb
    learning_rate = 0.001

    dataset_size = 1280000
    chunk_size = 51200

    interval_parameters['policy_interval'] = 24
    interval_parameters['finetune_interval'] = 24
    orig_finetune_interval = 50
    trial_phase_threshold = 1.0

    research_parameters['start_adapting_after'] = 2000
    research_parameters['hard_pool_max_threshold'] = 0.2

    pool_size = batch_size * 1 * num_labels
    test_size = 12500
    test_dataset_filename = 'data_non_station' + os.sep + 'imagenet-250-test-dataset.pkl'
    test_label_filename = 'data_non_station' + os.sep + 'imagenet-250-test-labels.pkl'

    if not research_parameters['adapt_structure']:
        cnn_string = "C,3,1,64#P,2,2,0#C,3,1,128#P,2,2,0#C,3,1,256#C,3,1,256#P,2,2,0#C,3,1,512#C,3,1,512#P,2,2,0#C,3,1,512#C,3,1,512#PG,2,2,0#FC,4096,0,0#Terminate,0,0,0"
    else:
        cnn_string = "C,3,1,32#P,2,2,0#C,3,1,32#P,2,2,0#C,3,1,32#C,3,1,32#P,2,2,0#C,3,1,32#C,3,1,32#P,2,2,0#C,3,1,32#C,3,1,32#PG,2,2,0#FC,4096,0,0#Terminate,0,0,0"
        filter_vector = [64, 0, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512]
        filter_min_threshold = 24
        add_amount, remove_amount = 16, 8

elif datatype == 'svhn-10':

    image_size = 32
    num_labels = 10
    num_channels = 3
    dataset_size = 128000
    use_warmup_epoch = False

    chunk_size = 25600
    dataset_size = 1280000

    dataset_filename = ''

    interval_parameters['policy_interval'] = 50
    interval_parameters['finetune_interval'] = 50
    orig_finetune_interval = 50
    trial_phase_threshold = 1.0

    research_parameters['start_adapting_after'] = 1000
    pool_size = batch_size * 10 * num_labels
    test_size = 26032
    test_dataset_filename = 'data_non_station' + os.sep + 'svhn-10-test-dataset.pkl'
    test_label_filename = 'data_non_station' + os.sep + 'svhn-10-test-labels.pkl'

    if not research_parameters['adapt_structure']:
        cnn_string = "C,5,1,128#P,3,2,0#C,5,1,128#P,3,2,0#C,3,1,128#PG,6,4,0#Terminate,0,0,0"
    else:
        cnn_string = "C,5,1,24#P,3,2,0#C,5,1,24#P,3,2,0#C,3,1,24#PG,6,4,0#Terminate,0,0,0"
        filter_vector = [128, 0, 128, 0, 128]
        add_amount, remove_amount = 4, 2
        filter_min_threshold = 12
