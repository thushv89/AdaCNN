import logging
import os



def get_research_hyperparameters(dataset_name, adapt, use_pooling,logging_level):
    '''
    These are various research hyperparameters used in AdaCNN.
    Some hyperparameters can be sent as arguments for convenience
    :param dataset_name:
    :param adapt: Whether use AdaCNN or Rigid-CNN
    :param use_pooling: Use Rigid-CNN-B or Rigid-CNN
    :return: Research hyperparamters as a dictionary
    '''
    research_parameters = {
        'save_train_test_images': False, # If true will save train and test images randomly (to make sure images are correctly read)
        'log_class_distribution': True, 'log_distribution_every': 24, # log distribution of data (useful for generating data distribution over time curves)
        'adapt_structure': adapt,  # Enable AdaCNN behavior
        'hard_pool_acceptance_rate': 0.1,  # Probability with which data is accepted in to the pool
        'optimizer': 'Momentum', 'momentum': 0.0, 'pool_momentum': 0.9,  # Two momentums one for data one for pool
        'use_custom_momentum_opt': True, # Use a custom implemented momentum (Tensorflow builtin optimizer doesnot support variable size tensors
        'remove_filters_by': 'Activation', # The criteria for removing filters (AdaCNN) set of minimum maximum mean activations
        'optimize_end_to_end': True, # if true functions such as add and finetune will optimize the network from starting layer to end (fulcon_out)
        'loss_diff_threshold': 0.02, # This is used to check if the loss reduction has stabalized
        'start_adapting_after': 500, # Acts as a warming up phase, adapting from the very begining can make CNNs unstable
        'debugging': True if logging_level == logging.DEBUG else False,
        'stop_training_at': 11000,  # If needed to truncate training earlier
        'train_min_activation': False,
        'use_weighted_loss': True,  # Weight the loss by the class distribution at the time
        'whiten_images': True,  # Whiten images using batch mean and batch std for each batch
        'finetune_rate': 0.5,  # amount of data from data pool used for finetuning
        'pool_randomize': True,  # randomize the pool data when training with it
        'pool_randomize_rate': 0.25,  # frequency the pool data randomized for
        'pooling_for_nonadapt': use_pooling,
        'hard_pool_max_threshold': 0.5,  # when there's not much data use a higher pool accumulation rate
    }

    if dataset_name == 'cifar-10':
        research_parameters['start_adapting_after'] = 1000
    elif dataset_name== 'imagenet-250':
        research_parameters['start_adapting_after'] = 2000
        research_parameters['hard_pool_max_threshold'] = 0.2
    elif dataset_name=='svhn-10':
        research_parameters['start_adapting_after'] = 1000

    if adapt:
        # quickly accumulate data at the beginning
        research_parameters['hard_pool_acceptance_rate'] *= 2.0

    return research_parameters


def get_interval_related_hyperparameters(dataset_name):

    interval_parameters = {
        'history_dump_interval': 500,
        'policy_interval': 0,  # number of batches to process for each policy iteration
        'finetune_interval': 0,
        'orig_finetune_interval':0,
        'test_interval': 100
    }

    if dataset_name == 'cifar-10':
        interval_parameters['policy_interval'] = 24
        interval_parameters['finetune_interval'] = 24
        interval_parameters['orig_finetune_interval'] = 50

    if dataset_name == 'cifar-100':
        interval_parameters['policy_interval'] = 24
        interval_parameters['finetune_interval'] = 24
        interval_parameters['orig_finetune_interval'] = 50

    elif dataset_name == 'imagenet-250':
        interval_parameters['policy_interval'] = 48
        interval_parameters['finetune_interval'] = 48
        interval_parameters['orig_finetune_interval'] = 48

    elif dataset_name == 'svhn-10':

        interval_parameters['policy_interval'] = 24
        interval_parameters['finetune_interval'] = 24
        interval_parameters['orig_finetune_interval'] = 50

    return interval_parameters


def get_model_specific_hyperparameters(dataset_name, dataset_behavior, adapt_structure, use_pooling, use_fse_capacity, num_labels):

    model_hyperparameters = {}
    model_hyperparameters['adapt_structure'] = adapt_structure
    model_hyperparameters['batch_size'] = 64  # number of datapoints in a single batch
    model_hyperparameters['start_lr'] = 0.0001
    model_hyperparameters['min_learning_rate'] = 0.000001
    model_hyperparameters['decay_learning_rate'] = True
    model_hyperparameters['decay_rate'] = 0.75
    model_hyperparameters['adapt_decay_rate'] = 0.9 # decay rate used for adaptation related optimziations
    if not use_fse_capacity:
        model_hyperparameters['dropout_rate'] = 0.5
        model_hyperparameters['in_dropout_rate'] = 0.2
    else:
        model_hyperparameters['dropout_rate'] = 0.1
        model_hyperparameters['in_dropout_rate'] = 0.0
    model_hyperparameters['use_dropout'] = True
    model_hyperparameters['check_early_stopping_from'] = 5
    model_hyperparameters['accuracy_drop_cap'] = 3
    model_hyperparameters['iterations_per_batch'] = 1

    model_hyperparameters['epochs'] = 10

    model_hyperparameters['n_iterations'] = 5000
    model_hyperparameters['start_eps'] = 0.5
    model_hyperparameters['eps_decay'] = 0.9
    model_hyperparameters['validation_set_accumulation_decay'] = 0.9
    model_hyperparameters['lrn_radius'] = 5
    model_hyperparameters['lrn_alpha'] = 0.0001
    model_hyperparameters['lrn_beta'] = 0.75

    if not (adapt_structure and use_pooling):
        model_hyperparameters['iterations_per_batch'] = 2

    model_hyperparameters['include_l2_loss'] = False
    model_hyperparameters['beta'] = 0.0005
    model_hyperparameters['use_loc_res_norm'] = False

    model_hyperparameters['top_k_accuracy'] = 1.0


    if dataset_name == 'cifar-10':

        pool_size = model_hyperparameters['batch_size'] * 10* num_labels

        if not adapt_structure:
            if not use_fse_capacity:
                cnn_string = "C,3,1,144#C,3,1,144#C,3,1,144#P,3,2,0" \
                             "#C,3,1,288#C,3,1,288#C,3,1,288" \
                             "#PG,3,2,0#Terminate,0,0,0"
            else:
                cnn_string = "C,3,1,68#C,3,1,68#C,3,1,68#P,3,2,0" \
                             "#C,3,1,136#C,3,1,136#C,3,1,136" \
                             "#PG,3,2,0#Terminate,0,0,0"
        else:
            cnn_string = "C,3,1,32#C,3,1,32#C,3,1,32#P,3,2,0" \
                         "#C,3,1,32#C,3,1,32#C,3,1,32" \
                         "#PG,3,2,0#Terminate,0,0,0"

            filter_vector = [144, 144, 144, 0, 288, 288, 288,0]
            add_amount, remove_amount, add_fulcon_amount = 8, 4, -1
            filter_min_threshold = 24
            fulcon_min_threshold = 64

        model_hyperparameters['n_tasks'] = 2
        model_hyperparameters['binned_data_dist_length'] = 10
        model_hyperparameters['prune_min_bound'] = 0.25
        model_hyperparameters['prune_max_bound'] = 0.75

    if dataset_name == 'cifar-100':

        pool_size = model_hyperparameters['batch_size'] * 2* num_labels

        if not adapt_structure:
            if not use_fse_capacity:
                cnn_string = "C,3,1,64#C,3,1,128#C,3,1,256#C,3,1,256" \
                             "#P,2,2,0#C,3,1,256#C,3,1,256#C,3,1,256#C,3,1,256" \
                             "#PG,3,2,0#FC,512,0,0#FC,512,0,0#FC,100,0,0#Terminate,0,0,0"
            else:
                cnn_string = "C,3,1,6#C,3,1,12#C,3,1,24#C,3,1,24" \
                             "#P,2,2,0#C,3,1,24#C,3,1,24#C,3,1,24#C,3,1,24" \
                             "#PG,3,2,0#FC,48,0,0#FC,48,0,0#FC,100,0,0#Terminate,0,0,0"
        else:
            cnn_string = "C,3,1,32#C,3,1,32#C,3,1,32#C,3,1,32" \
                         "#P,2,2,0#C,3,1,64#C,3,1,64#C,3,1,64#C,3,1,64" \
                         "#PG,2,2,0#FC,128,0,0#FC,128,0,0#FC,64,0,0#Terminate,0,0,0"

            start_filter_vector = [32, 32,   32,  32, 0,  64,  64,  64,  64, 0, 128, 128, 64]
            filter_vector =       [64, 128, 256, 256, 0, 256, 256, 256, 256, 0, 512, 512, 110]
            add_amount, remove_amount, add_fulcon_amount = 8, 4, 8
            filter_min_threshold = 24
            fulcon_min_threshold = 48

        model_hyperparameters['n_iterations'] = 10000
        model_hyperparameters['n_tasks'] = 4
        model_hyperparameters['binned_data_dist_length'] = 10
        model_hyperparameters['prune_min_bound'] = 0.75 # used to be 0.5
        model_hyperparameters['prune_max_bound'] = 0.9 # used to be 0.75

    elif dataset_name== 'imagenet-250':
        model_hyperparameters['top_k_accuracy'] = 5.0
        model_hyperparameters['n_iterations'] = 10000
        model_hyperparameters['epochs'] = 8
        pool_size = int(model_hyperparameters['batch_size'] * 1 * num_labels)

        if not adapt_structure:
            cnn_string = "C,3,1,64#C,3,1,64#P,2,2,0#C,3,1,128#C,3,1,128#P,2,2,0" \
                         "#C,3,1,256#C,3,1,256#P,2,2,0#C,3,1,256" \
                         "#C,3,1,256#P,2,2,0#C,3,1,256#C,3,1,256" \
                         "#PG,2,2,0#FC,2048,0,0#FC,2048,0,0#FC,250,0,0#Terminate,0,0,0"
        else:
            cnn_string = "C,3,1,48#C,3,1,48#P,2,2,0#C,3,1,48#C,3,1,48" \
                         "#P,2,2,0#C,3,1,48#C,3,1,48" \
                         "#P,2,2,0#C,3,1,48#C,3,1,48" \
                         "#P,2,2,0#C,3,1,48#C,3,1,48" \
                         "#PG,2,2,0#FC,128,0,0#FC,128,0,0#FC,125,0,0#Terminate,0,0,0"

            start_filter_vector = [48, 48, 0,  48,  48, 0,  48,  48, 0,  48,  48, 0,  48,  48, 0,  128,  128, 125]
            filter_vector =       [64, 64, 0, 128, 128, 0, 256, 256, 0, 256, 256, 0, 256, 256, 0, 2048, 2048, 250]
            filter_min_threshold = 32
            fulcon_min_threshold = 100
            add_amount, remove_amount, add_fulcon_amount = 8, 4, 24

        model_hyperparameters['n_tasks'] = 2
        model_hyperparameters['binned_data_dist_length'] = 25

        model_hyperparameters['prune_min_bound'] = 0.75
        model_hyperparameters['prune_max_bound'] = 0.9


    elif dataset_name=='svhn-10':
        pool_size = model_hyperparameters['batch_size'] * 10 * num_labels
        # test_size = 26032

        if not adapt_structure:
            cnn_string = "C,5,1,128#P,3,2,0#C,5,1,128#P,3,2,0#C,3,1,128#PG,6,4,0#Terminate,0,0,0"
        else:
            cnn_string = "C,5,1,24#P,3,2,0#C,5,1,24#P,3,2,0#C,3,1,24#PG,6,4,0#Terminate,0,0,0"
            filter_vector = [128, 0, 128, 0, 128]
            add_amount, remove_amount, add_fulcon_amount = 4, 2, 36
            filter_min_threshold = 12
        model_hyperparameters['n_tasks'] = 4
        model_hyperparameters['binned_data_dist_length'] = 10

    model_hyperparameters['cnn_string'] = cnn_string

    if adapt_structure or use_pooling:
        model_hyperparameters['pool_size'] = pool_size

    if adapt_structure:
        model_hyperparameters['filter_vector'] = filter_vector
        model_hyperparameters['start_filter_vector'] = start_filter_vector
        model_hyperparameters['add_amount'] = add_amount
        model_hyperparameters['add_fulcon_amount'] = add_fulcon_amount
        model_hyperparameters['remove_amount'] = remove_amount
        model_hyperparameters['filter_min_threshold'] = filter_min_threshold
        model_hyperparameters['fulcon_min_threshold'] = fulcon_min_threshold

    return model_hyperparameters


def get_data_specific_hyperparameters(dataset_name, dataset_behavior, dataset_dir):
    global research_parameters, interval_parameters
    data_hyperparameters,model_hyperparameters = {},{}

    resize_to = 0
    if dataset_name == 'cifar-10':
        image_size = 24
        num_labels = 10
        num_channels = 3  # rgb
        dataset_size = 50000
        test_size = 10000
        n_slices = 1
        fluctuation = 25

    elif dataset_name == 'cifar-100':

        image_size = 24
        num_labels = 100
        num_channels = 3  # rgb
        dataset_size = 50000
        test_size = 10000
        n_slices = 1
        fluctuation = 15

    elif dataset_name == 'imagenet-250':
        image_size = 128
        num_labels = 250
        num_channels = 3  # rgb
        dataset_size = 300000
        test_size = 12500
        n_slices = 10
        fluctuation = 5
        resize_to = 96

    elif dataset_name == 'svhn-10':

        image_size = 32
        num_labels = 10
        num_channels = 3
        dataset_size = 73257
        test_size = 26032
        n_slices = 1
        fluctuation = 25

    else:
        raise NotImplementedError

    data_hyperparameters['dataset_name'] = dataset_name
    data_hyperparameters['image_size'] = image_size
    data_hyperparameters['resize_to'] = resize_to
    data_hyperparameters['n_labels'] = num_labels
    data_hyperparameters['n_channels'] = num_channels
    data_hyperparameters['train_size'] = dataset_size
    data_hyperparameters['test_size'] = test_size
    data_hyperparameters['n_slices'] = n_slices
    data_hyperparameters['fluctuation_factor'] = fluctuation
    return data_hyperparameters