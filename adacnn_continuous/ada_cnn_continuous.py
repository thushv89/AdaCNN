__author__ = 'Thushan Ganegedara'

import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import numpy as np
import os
from math import ceil, floor
import logging
import sys
import ada_cnn_aac
from data_pool import Pool
from collections import Counter
from scipy.misc import imsave
import getopt
import time
import utils
import queue
from multiprocessing import Pool as MPPool
import copy
import constants
import cnn_hyperparameters_getter_start_big as cnn_hyperparameters_getter
import cnn_optimizer
import cnn_intializer_continuous
import ada_cnn_adapter
import data_generator
import label_sequence_generator
import h5py
import cnn_model_saver

logging_level = logging.INFO
logging_format = '[%(funcName)s] %(message)s'

datatype,behavior = None,None
dataset_dir,output_dir = None,None
adapt_structure = False
rigid_pooling = False
rigid_pool_type = None
rigid_naive = False

interval_parameters, research_parameters, model_hyperparameters, dataset_info = None, None, None, None
image_size, num_channels,resize_to = None,None,None
n_epochs, n_iterations, iterations_per_batch, num_labels, train_size, test_size, n_slices, data_fluctuation = None,None,None,None,None,None,None,None
cnn_string, filter_vector = None,None

batch_size = None
start_lr, decay_learning_rate, decay_rate, decay_steps = None,None,None,None
beta, include_l2_loss = None,None
use_dropout, in_dropout_rate, dropout_rate, current_adaptive_dropout = None, None, None,None
pool_size = None
use_loc_res_norm, lrn_radius, lrn_alpha, lrn_beta = None, None, None, None

# Constant Strings
TF_WEIGHTS = constants.TF_WEIGHTS
TF_BIAS = constants.TF_BIAS
TF_ACTIVAIONS_STR = constants.TF_ACTIVAIONS_STR
TF_FC_WEIGHT_IN_STR = constants.TF_FC_WEIGHT_IN_STR
TF_FC_WEIGHT_OUT_STR = constants.TF_FC_WEIGHT_OUT_STR
TF_LOSS_VEC_STR = constants.TF_LOSS_VEC_STR
TF_GLOBAL_SCOPE = constants.TF_GLOBAL_SCOPE
TOWER_NAME = constants.TOWER_NAME
TF_ADAPTATION_NAME_SCOPE = constants.TF_ADAPTATION_NAME_SCOPE
TF_SCOPE_DIVIDER = constants.TF_SCOPE_DIVIDER


start_eps = None
eps_decay = None
valid_acc_decay = None

n_tasks = None
prune_min_bound, prune_max_bound = None, None

act_decay = 0.99

def set_varialbes_with_input_arguments(dataset_name, dataset_behavior, adapt_structure, use_rigid_pooling, use_fse_capacity):
    global interval_parameters, model_hyperparameters, research_parameters, dataset_info, cnn_string, filter_vector
    global image_size, num_channels, resize_to
    global n_epochs, n_iterations, iterations_per_batch, num_labels, train_size, test_size, n_slices, data_fluctuation
    global start_lr, decay_learning_rate, decay_rate, decay_steps
    global batch_size, beta, include_l2_loss
    global use_dropout, in_dropout_rate, dropout_rate, current_adaptive_dropout
    global pool_size
    global use_loc_res_norm, lrn_radius, lrn_alpha, lrn_beta
    global start_eps,eps_decay,valid_acc_decay
    global n_tasks, prune_min_bound, prune_max_bound

    # Data specific parameters
    dataset_info = cnn_hyperparameters_getter.get_data_specific_hyperparameters(dataset_name, dataset_behavior,
                                                                                dataset_dir)
    image_size = dataset_info['image_size']
    resize_to = dataset_info['resize_to']
    num_channels = dataset_info['n_channels']

    # interval parameters
    interval_parameters = cnn_hyperparameters_getter.get_interval_related_hyperparameters(dataset_name)

    # Research parameters
    research_parameters = cnn_hyperparameters_getter.get_research_hyperparameters(dataset_name, adapt_structure, use_rigid_pooling, logging_level)

    # Model Hyperparameters
    model_hyperparameters = cnn_hyperparameters_getter.get_model_specific_hyperparameters(datatype, dataset_behavior, adapt_structure, rigid_pooling, use_fse_capacity, dataset_info['n_labels'])

    n_epochs = model_hyperparameters['epochs']
    n_iterations = model_hyperparameters['n_iterations']

    iterations_per_batch = model_hyperparameters['iterations_per_batch']

    num_labels = dataset_info['n_labels']
    train_size = dataset_info['train_size']
    test_size = dataset_info['test_size']
    n_slices = dataset_info['n_slices']
    data_fluctuation = dataset_info['fluctuation_factor']

    cnn_string = model_hyperparameters['cnn_string']
    if adapt_structure:
        filter_vector = model_hyperparameters['filter_vector']

    start_lr = model_hyperparameters['start_lr']
    decay_learning_rate = model_hyperparameters['decay_learning_rate']
    decay_rate = model_hyperparameters['decay_rate']
    decay_steps = 1

    batch_size = model_hyperparameters['batch_size']
    beta = model_hyperparameters['beta']
    include_l2_loss = model_hyperparameters['include_l2_loss']

    # Dropout
    use_dropout = model_hyperparameters['use_dropout']
    in_dropout_rate = model_hyperparameters['in_dropout_rate']
    dropout_rate = model_hyperparameters['dropout_rate']
    current_adaptive_dropout = dropout_rate

    # Local Response Normalization
    use_loc_res_norm = model_hyperparameters['use_loc_res_norm']
    lrn_radius = model_hyperparameters['lrn_radius']
    lrn_alpha = model_hyperparameters['lrn_alpha']
    lrn_beta = model_hyperparameters['lrn_beta']

    # pool parameters
    if adapt_structure or use_rigid_pooling:
        pool_size = model_hyperparameters['pool_size']

    if adapt_structure:
        start_eps = model_hyperparameters['start_eps']
        eps_decay = model_hyperparameters['eps_decay']
    valid_acc_decay = model_hyperparameters['validation_set_accumulation_decay']

    # Tasks
    n_tasks = model_hyperparameters['n_tasks']



cnn_ops, cnn_hyperparameters = None, None

state_action_history = []

cnn_ops, cnn_hyperparameters = None, None
tf_prune_cnn_hyperparameters = {}

num_gpus = -1

# Tensorflow Op / Variable related Python variables
# Optimizer Related
optimizer, custom_lr = None,None
tf_learning_rate = None
# Optimizer (Data) Related
tf_avg_grad_and_vars, apply_grads_op, concat_loss_vec_op, \
update_train_velocity_op, mean_loss_op = None,None,None,None,None

# Optimizer (Pool) Related
tf_pool_avg_gradvars, apply_pool_grads_op, update_pool_velocity_ops, mean_pool_loss = None, None, None, None

# Data related
tf_train_data_batch, tf_train_label_batch, tf_data_weights = [], [], []
tf_test_dataset, tf_test_labels = None, None
tf_valid_data_batch, tf_valid_label_batch = None, None

# Data (Pool) related
tf_pool_data_batch, tf_pool_label_batch = [], []
pool_pred, augmented_pool_data_batch, augmented_pool_label_batch = None, [],[]

# Logit related
tower_grads, tower_loss_vectors, tower_losses, tower_predictions = [], [], [], []
tower_pool_grads, tower_pool_losses = [], []
tower_logits = []
tf_dropout_rate = None

# Test/Valid related
valid_loss_op,valid_predictions_op, test_predicitons_op = None,None,None

# Adaptation related
tf_slice_optimize = {}
tf_slice_vel_update = {}
tf_add_filters_ops, tf_rm_filters_ops, tf_replace_ind_ops = {}, {}, {}
tf_indices, tf_indices_to_rm, tf_replicative_factor_vec, tf_indices_size = None,None,None,None
tf_update_hyp_ops = {}
tf_action_info = None
tf_weights_this,tf_bias_this = None, None
tf_weights_next,tf_wvelocity_this, tf_bvelocity_this, tf_wvelocity_next = None, None, None, None
tf_weight_shape,tf_in_size, tf_out_size = None, None, None
increment_global_step_op = None
tf_weight_mean_ops = None
tf_retain_id_placeholders = {}
tf_act_this = None

adapt_period = None

# Loggers
logger = None
perf_logger = None

# Reset CNN
tf_reset_cnn, tf_reset_cnn_custom = None, None
prune_reward_experience = []


def inference(dataset, tf_cnn_hyperparameters, training):
    global logger,cnn_ops

    first_fc = 'fulcon_out' if 'fulcon_0' not in cnn_ops else 'fulcon_0'

    last_conv_id = ''
    for op in cnn_ops:
        if 'conv' in op:
            last_conv_id = op

    logger.debug('Defining the logit calculation ...')
    logger.debug('\tCurrent set of operations: %s' % cnn_ops)
    activation_ops = []

    x = dataset
    '''if research_parameters['whiten_images']:
        mu, var = tf.nn.moments(x, axes=[1, 2, 3])
        tr_x = tf.transpose(x, [1, 2, 3, 0])
        tr_x = (tr_x - mu) / tf.maximum(tf.sqrt(var), 1.0 / (image_size * image_size * num_channels))
        x = tf.transpose(tr_x, [3, 0, 1, 2])'''

    if training and use_dropout:
        x = tf.nn.dropout(x, keep_prob=1.0 - in_dropout_rate, name='input_dropped')

    logger.debug('\tReceived data for X(%s)...' % x.get_shape().as_list())

    # need to calculate the output according to the layers we have
    for op in cnn_ops:
        if 'conv' in op:
            with tf.variable_scope(op, reuse=True) as scope:
                logger.debug('\tConvolving (%s) With Weights:%s Stride:%s' % (
                    op, cnn_hyperparameters[op]['weights'], cnn_hyperparameters[op]['stride']))
                logger.debug('\t\tWeights: %s', tf.shape(tf.get_variable(TF_WEIGHTS)).eval())
                w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)

                x = tf.nn.conv2d(x, w, cnn_hyperparameters[op]['stride'],
                                 padding=cnn_hyperparameters[op]['padding'])

                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                                     tf.assign(tf.get_variable(TF_ACTIVAIONS_STR),
                                               act_decay * tf.get_variable(TF_ACTIVAIONS_STR) + (
                                               1 - act_decay) * tf.reduce_mean(tf.abs(w), axis=[0, 1, 2])))

                x = utils.lrelu(x + b, name=scope.name + '/top')



        if 'pool' in op:
            logger.debug('\tPooling (%s) with Kernel:%s Stride:%s' % (
                op, cnn_hyperparameters[op]['kernel'], cnn_hyperparameters[op]['stride']))
            if cnn_hyperparameters[op]['type'] is 'max':
                x = tf.nn.max_pool(x, ksize=cnn_hyperparameters[op]['kernel'],
                                   strides=cnn_hyperparameters[op]['stride'],
                                   padding=cnn_hyperparameters[op]['padding'])
            elif cnn_hyperparameters[op]['type'] is 'avg':
                x = tf.nn.avg_pool(x, ksize=cnn_hyperparameters[op]['kernel'],
                                   strides=cnn_hyperparameters[op]['stride'],
                                   padding=cnn_hyperparameters[op]['padding'])
            if datatype!='imagenet-250' and training and use_dropout:
                x = tf.nn.dropout(x, keep_prob=1.0 - tf_dropout_rate, name='dropout')

            if use_loc_res_norm and 'pool_global' != op:
                x = tf.nn.local_response_normalization(x, depth_radius=lrn_radius, alpha=lrn_alpha, beta=lrn_beta)

        if 'fulcon' in op:
            with tf.variable_scope(op, reuse=True) as scope:
                w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)

                if first_fc == op:
                    # we need to reshape the output of last subsampling layer to
                    # convert 4D output to a 2D input to the hidden layer
                    # e.g subsample layer output [batch_size,width,height,depth] -> [batch_size,width*height*depth]

                    logger.debug('Input size of fulcon_out : %d', cnn_hyperparameters[op]['in'])
                    # Transpose x (b,h,w,d) to (b,d,w,h)
                    # This help us to do adaptations more easily
                    x = tf.transpose(x, [0, 3, 1, 2])
                    x = tf.reshape(x, [-1, tf_cnn_hyperparameters[op][TF_FC_WEIGHT_IN_STR]])

                    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                                         tf.assign(tf.get_variable(TF_ACTIVAIONS_STR),
                                                   act_decay * tf.get_variable(TF_ACTIVAIONS_STR) + (
                                                       1 - act_decay) * tf.reduce_mean(tf.abs(w), axis=[0])))

                    x = utils.lrelu(tf.matmul(x, w) + b, name=scope.name + '/top')
                    if training and use_dropout:
                        x = tf.nn.dropout(x, keep_prob=1.0 - tf_dropout_rate, name='dropout')

                elif 'fulcon_out' == op:
                    x = tf.matmul(x, w) + b

                else:
                    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                                         tf.assign(tf.get_variable(TF_ACTIVAIONS_STR),
                                                   act_decay * tf.get_variable(TF_ACTIVAIONS_STR) + (
                                                       1 - act_decay) * tf.reduce_mean(tf.abs(w), axis=[0])))

                    x = utils.lrelu(tf.matmul(x, w) + b, name=scope.name + '/top')
                    if training and use_dropout:
                        x = tf.nn.dropout(x, keep_prob=1.0 - tf_dropout_rate, name='dropout')

    return x


def get_weights_mean_for_pruning():
    weight_mean_ops = {}

    for op in cnn_ops:
        if 'conv' in op:
            with tf.variable_scope(op, reuse=True) as scope:
                w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)
                weight_mean_ops[op] = tf.assign(tf.get_variable(TF_ACTIVAIONS_STR), tf.reduce_mean(w, [0, 1, 2]),
                              validate_shape=False)

        if 'fulcon' in op:
            with tf.variable_scope(op, reuse=True) as scope:
                w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)
                weight_mean_ops[op] = tf.assign(tf.get_variable(TF_ACTIVAIONS_STR), tf.reduce_mean(w, [1]),
                              validate_shape=False)

    return weight_mean_ops


def tower_loss(dataset, labels, weighted, tf_data_weights, tf_cnn_hyperparameters):
    global cnn_ops
    logits = inference(dataset, tf_cnn_hyperparameters, True)
    # use weighted loss
    if weighted:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels) * tf_data_weights)
    else:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    if include_l2_loss:
        layers_for_l2_loss = []
        l2_weights = []
        for op in cnn_ops:

            if 'fulcon' in op or 'conv' in op:
                with tf.variable_scope(op):
                    l2_weights.append(tf.get_variable(TF_WEIGHTS))

        loss = tf.reduce_sum([loss, beta * tf.reduce_sum([tf.nn.l2_loss(w) for w in l2_weights])])

    total_loss = loss

    return total_loss


def calc_loss_vector(scope, dataset, labels, tf_cnn_hyperparameters):
    logits = inference(dataset, tf_cnn_hyperparameters, True)
    return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name=TF_LOSS_VEC_STR)


def average_gradients(tower_grads):
    # tower_grads => [((grads0gpu0,var0gpu0),...,(grads0gpuN,var0gpuN)),((grads1gpu0,var1gpu0),...,(grads1gpuN,var1gpuN))]
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, v in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, axis=0)
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def concat_loss_vector_towers(tower_loss_vectors):
    concat_loss_vec = None
    for loss_vec in tower_loss_vectors:
        if concat_loss_vec is None:
            concat_loss_vec = tf.identity(loss_vec)
        else:
            concat_loss_vec = tf.concat(axis=1, values=loss_vec)

    return concat_loss_vec


def mean_tower_activations(tower_activations):
    mean_activations = []
    if len(tower_activations)>1:
        for act_towers in zip(*tower_activations):
            stacked_activations = None
            for a in act_towers:
                if stacked_activations is None:
                    stacked_activations = tf.identity(a)
                else:
                    stacked_activations = tf.stack([stacked_activations, a], axis=0)

            mean_activations.append(tf.reduce_mean(stacked_activations, [0]))
    else:
        for act_single_tower in tower_activations[0]:
            mean_activations.append(act_single_tower)

    return mean_activations


def predict_with_logits(logits):
    # Predictions for the training, validation, and test data.
    prediction = tf.nn.softmax(logits)
    return prediction


def predict_with_dataset(dataset, tf_cnn_hyperparameters):
    logits = inference(dataset, tf_cnn_hyperparameters, False)
    prediction = tf.nn.softmax(logits)
    return prediction


def accuracy(predictions, labels):
    assert predictions.shape[0] == labels.shape[0]
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def setup_loggers(adapt_structure):
    '''
    Setting up loggers
    logger: Main Logger
    error_logger: Log Train loss, Validation Accuracy, Test Accuracy
    perf_logger: Logging time
    hyp_logger: Log hyperparameters
    :param adapt_structure:
    :return:
    '''


    main_logger = logging.getLogger('main_ada_cnn_logger')
    main_logger.setLevel(logging_level)
    main_logger.propagate=False
    # File handler for writing to file
    main_file_handler = logging.FileHandler(output_dir + os.sep + 'ada_cnn_main.log', mode='w')
    main_file_handler.setLevel(logging.DEBUG)
    main_file_handler.setFormatter(logging.Formatter('%(message)s'))
    main_logger.addHandler(main_file_handler)
    # Console handler for writing to console
    main_console = logging.StreamHandler(sys.stdout)
    main_console.setFormatter(logging.Formatter(logging_format))
    #main_console.setLevel(logging_level)
    main_logger.addHandler(main_console)

    error_logger = logging.getLogger('error_logger')
    error_logger.propagate = False
    error_logger.setLevel(logging.INFO)
    errHandler = logging.FileHandler(output_dir + os.sep + 'Error.log', mode='w')
    errHandler.setFormatter(logging.Formatter('%(message)s'))
    error_logger.addHandler(errHandler)
    error_logger.info('#Batch_ID,Loss(Train),Train Accuracy, Valid(Unseen),Test Accuracy')

    perf_logger = logging.getLogger('time_logger')
    perf_logger.propagate = False
    perf_logger.setLevel(logging.INFO)
    perf_handler = logging.FileHandler(output_dir + os.sep + 'time.log', mode='w')
    perf_handler.setFormatter(logging.Formatter('%(message)s'))
    perf_logger.addHandler(perf_handler)
    perf_logger.info('#Batch_ID,Time(Full),Time(Train),Op count, Var count')

    hyp_logger = logging.getLogger('hyperparameter_logger')
    hyp_logger.propagate = False
    hyp_logger.setLevel(logging.INFO)
    hyp_handler = logging.FileHandler(output_dir + os.sep + 'Hyperparameter.log', mode='w')
    hyp_handler.setFormatter(logging.Formatter('%(message)s'))
    hyp_logger.addHandler(hyp_handler)

    cnn_structure_logger, q_logger, prune_logger = None, None, None
    if adapt_structure:
        cnn_structure_logger = logging.getLogger('cnn_structure_logger')
        main_logger.propagate = False
        cnn_structure_logger.setLevel(logging.INFO)
        structHandler = logging.FileHandler(output_dir + os.sep + 'cnn_structure.log', mode='w')
        structHandler.setFormatter(logging.Formatter('%(message)s'))
        cnn_structure_logger.addHandler(structHandler)
        cnn_structure_logger.info('#batch_id:state:action:reward:#layer_1_hyperparameters#layer_2_hyperparameters#...')

        q_logger = logging.getLogger('q_eval_rand_logger')
        main_logger.propagate = False
        q_logger.setLevel(logging.INFO)
        q_handler = logging.FileHandler(output_dir + os.sep + 'QMetric.log', mode='w')
        q_handler.setFormatter(logging.Formatter('%(message)s'))
        q_logger.addHandler(q_handler)
        q_logger.info('#batch_id,q_metric')

        prune_logger = logging.getLogger('prune_reward_logger')
        prune_logger.propagate = False
        prune_logger.setLevel(logging.INFO)
        prune_handler = logging.FileHandler(output_dir + os.sep + 'PruneRewardExperience.log', mode='w')
        prune_handler.setFormatter(logging.Formatter('%(message)s'))
        prune_logger.addHandler(prune_handler)
        prune_logger.info('#task_id,prune_factor,acc_after, acc_before, acc_gain,reward, infer_type')

    class_dist_logger = logging.getLogger('class_dist_logger')
    class_dist_logger.propagate = False
    class_dist_logger.setLevel(logging.INFO)
    class_dist_handler = logging.FileHandler(output_dir + os.sep + 'class_distribution.log', mode='w')
    class_dist_handler.setFormatter(logging.Formatter('%(message)s'))
    class_dist_logger.addHandler(class_dist_handler)

    pool_dist_logger = logging.getLogger('pool_distribution_logger')
    pool_dist_logger.propagate = False
    pool_dist_logger.setLevel(logging.INFO)
    pool_handler = logging.FileHandler(output_dir + os.sep + 'pool_valid_distribution.log', mode='w')
    pool_handler.setFormatter(logging.Formatter('%(message)s'))
    pool_dist_logger.addHandler(pool_handler)
    pool_dist_logger.info('#Class distribution')

    pool_ft_dist_logger = logging.getLogger('pool_distribution_ft_logger')
    pool_ft_dist_logger.propagate = False
    pool_ft_dist_logger.setLevel(logging.INFO)
    pool_ft_handler = logging.FileHandler(output_dir + os.sep + 'pool_ft_distribution.log', mode='w')
    pool_ft_handler.setFormatter(logging.Formatter('%(message)s'))
    pool_ft_dist_logger.addHandler(pool_ft_handler)
    pool_ft_dist_logger.info('#Class distribution')

    return main_logger, perf_logger, \
           cnn_structure_logger, q_logger, class_dist_logger, \
           pool_dist_logger, pool_ft_dist_logger, hyp_logger, error_logger, prune_logger


def get_activation_dictionary(activation_list, cnn_ops, conv_op_ids):
    current_activations = {}
    for act_i, layer_act in enumerate(activation_list):
        current_activations[cnn_ops[conv_op_ids[act_i]]] = layer_act
    return current_activations


def define_tf_ops(global_step, tf_cnn_hyperparameters, init_cnn_hyperparameters):
    global optimizer
    global tf_train_data_batch, tf_train_label_batch, tf_data_weights, tf_reset_cnn, tf_reset_cnn_custom, tf_prune_cnn_hyperparameters
    global tf_test_dataset,tf_test_labels
    global tf_pool_data_batch, tf_pool_label_batch
    global tower_grads, tower_loss_vectors, tower_losses, tower_predictions
    global tower_pool_grads, tower_pool_losses, tower_logits,tf_dropout_rate
    global tf_add_filters_ops, tf_rm_filters_ops, tf_replace_ind_ops, tf_slice_optimize, tf_slice_vel_update
    global tf_indices, tf_indices_to_rm, tf_replicative_factor_vec, tf_indices_size, tf_weight_mean_ops, tf_retain_id_placeholders
    global tf_avg_grad_and_vars, apply_grads_op, concat_loss_vec_op, update_train_velocity_op, mean_loss_op
    global tf_pool_avg_gradvars, apply_pool_grads_op, update_pool_velocity_ops, mean_pool_loss
    global valid_loss_op,valid_predictions_op, test_predicitons_op
    global tf_valid_data_batch,tf_valid_label_batch
    global pool_pred, augmented_pool_data_batch, augmented_pool_label_batch
    global tf_update_hyp_ops, tf_action_info
    global tf_weights_this,tf_bias_this, tf_weights_next,tf_wvelocity_this, tf_bvelocity_this, tf_wvelocity_next
    global tf_weight_shape,tf_in_size, tf_out_size
    global increment_global_step_op,tf_learning_rate
    global tf_act_this
    global logger

    # custom momentum optimizing we calculate momentum manually
    logger.info('Defining Optimizer')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
    increment_global_step_op = tf.assign(global_step, global_step + 1)
    if not adapt_structure:
        tf_learning_rate = tf.train.exponential_decay(start_lr, global_step, decay_steps=decay_steps,
                                   decay_rate=decay_rate, staircase=True)
    else:
        tf_learning_rate = tf.placeholder(dtype=tf.float32,name='start_learning_rate_placeholder')

    tf_dropout_rate = tf.Variable(dropout_rate,trainable=False,dtype=tf.float32,name='tf_dropout_rate')
    # Test data (Global)
    logger.info('Defining Test data placeholders')
    if datatype != 'imagenet-250':
        tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels),
                                         name='TestDataset')
    else:
        tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size, resize_to, resize_to, num_channels),
                                         name='TestDataset')
    tf_test_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='TestLabels')

    # Tower-Like Calculations
    # Calculate logits (train and pool),
    # Calculate gradients (train and pool)
    # Tower_grads will contain
    # [[(grad0gpu0,var0gpu0),...,(gradNgpu0,varNgpu0)],...,[(grad0gpuD,var0gpuD),...,(gradNgpuD,varNgpuD)]]

    for gpu_id in range(num_gpus):
        logger.info('Defining TF operations for GPU ID: %d', gpu_id)
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('%s_%d' % (TOWER_NAME, gpu_id)) as scope:
                tf.get_variable_scope().reuse_variables()
                # Input train data
                logger.info('\tDefning Training Data placeholders and weights')
                if datatype != 'imagenet-250':
                    tf_train_data_batch.append(tf.placeholder(tf.float32,
                                                              shape=(
                                                                  batch_size, image_size, image_size, num_channels),
                                                              name='TrainDataset'))
                else:
                    tf_train_data_batch.append(tf.placeholder(tf.float32,
                                                              shape=(
                                                                  batch_size, resize_to, resize_to, num_channels),
                                                              name='TrainDataset'))

                tf_train_label_batch.append(
                    tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='TrainLabels'))


                tf_data_weights.append(tf.placeholder(tf.float32, shape=(batch_size), name='TrainWeights'))

                # Training data opearations
                logger.info('\tDefining logit operations')
                tower_logit_op = inference(tf_train_data_batch[-1],
                                                                    tf_cnn_hyperparameters, True)
                tower_logits.append(tower_logit_op)


                logger.info('\tDefine Loss for each tower')
                tf_tower_loss = tower_loss(tf_train_data_batch[-1], tf_train_label_batch[-1], True,
                                           tf_data_weights[-1], tf_cnn_hyperparameters)

                tower_losses.append(tf_tower_loss)
                tf_tower_loss_vec = calc_loss_vector(scope, tf_train_data_batch[-1], tf_train_label_batch[-1],
                                                     tf_cnn_hyperparameters)
                tower_loss_vectors.append(tf_tower_loss_vec)

                logger.info('\tGradient calculation opeartions for tower')
                tower_grad = cnn_optimizer.gradients(optimizer, tf_tower_loss, global_step,tf_learning_rate)
                tower_grads.append(tower_grad)

                logger.info('\tPrediction operations for tower')
                tower_pred = predict_with_dataset(tf_train_data_batch[-1], tf_cnn_hyperparameters)
                tower_predictions.append(tower_pred)

                # Pooling data operations
                logger.info('\tPool related operations')
                if datatype!='imagenet-250':
                    tf_pool_data_batch.append(tf.placeholder(tf.float32,
                                                             shape=(
                                                                 None, image_size, image_size, num_channels),
                                                             name='PoolDataset'))
                else:
                    tf_pool_data_batch.append(tf.placeholder(tf.float32,
                                                             shape=(
                                                                 None, resize_to, resize_to, num_channels),
                                                             name='PoolDataset'))
                tf_pool_label_batch.append(
                    tf.placeholder(tf.float32, shape=(None, num_labels), name='PoolLabels'))

                # Used for augmenting the pool data

                augmented_pool_data_batch.append(tf_augment_data_with(tf_pool_data_batch[-1],datatype))
                augmented_pool_label_batch.append(tf_pool_label_batch[-1])

                with tf.name_scope('pool') as scope:
                    #single_pool_logit_op = inference(tf_pool_data_batch[-1],tf_cnn_hyperparameters, True)

                    single_pool_loss = tower_loss(augmented_pool_data_batch[-1], augmented_pool_label_batch[-1], False, None,
                                                  tf_cnn_hyperparameters)
                    tower_pool_losses.append(single_pool_loss)
                    single_pool_grad = cnn_optimizer.gradients(optimizer, single_pool_loss, global_step, tf_learning_rate)
                    tower_pool_grads.append(single_pool_grad)

    logger.info('GLOBAL_VARIABLES (all)')
    logger.info('\t%s\n', [v.name for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])

    with tf.device('/gpu:0'):
        logger.info('Tower averaging for Gradients for Training data')
        # Train data operations
        # avg_grad_and_vars = [(avggrad0,var0),(avggrad1,var1),...]
        tf_avg_grad_and_vars = average_gradients(tower_grads)
        apply_grads_op, update_train_velocity_op = cnn_optimizer.apply_gradient_with_rmsprop(optimizer, tf_learning_rate,
                                                                                             global_step,
                                                                                             tf_avg_grad_and_vars)
        concat_loss_vec_op = concat_loss_vector_towers(tower_loss_vectors)
        mean_loss_op = tf.reduce_mean(tower_losses)

        logger.info('Tower averaging for Gradients for Pool data')
        # Pool data operations
        tf_pool_avg_gradvars = average_gradients(tower_pool_grads)
        apply_pool_grads_op, update_pool_velocity_ops = cnn_optimizer.apply_pool_gradient_with_rmsprop(
            optimizer,tf_learning_rate,global_step,tf_pool_avg_gradvars)

        mean_pool_loss = tf.reduce_mean(tower_pool_losses)

        # Weight mean calculation for pruning
        tf_weight_mean_ops = get_weights_mean_for_pruning()

    with tf.device('/gpu:0'):

        increment_global_step_op = tf.assign(global_step, global_step + 1)

        # GLOBAL: Tensorflow operations for hard_pool
        with tf.name_scope('pool') as scope:
            tf.get_variable_scope().reuse_variables()
            print(len(tf_pool_label_batch))
            pool_pred = predict_with_dataset(tf_pool_data_batch[0], tf_cnn_hyperparameters)

        # GLOBAL: Tensorflow operations for test data
        # Valid data (Next train batch) Unseen
        logger.info('Validation data placeholders, losses and predictions')
        if datatype != 'imagenet-250':
            tf_valid_data_batch = tf.placeholder(tf.float32,
                                                 shape=(batch_size, image_size, image_size, num_channels),
                                                 name='ValidDataset')
        else:
            tf_valid_data_batch = tf.placeholder(tf.float32,
                                                 shape=(batch_size, resize_to, resize_to, num_channels),
                                                 name='ValidDataset')

        tf_valid_label_batch = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='ValidLabels')
        # Tensorflow operations for validation data
        valid_loss_op = tower_loss(tf_valid_data_batch, tf_valid_label_batch, False, None, tf_cnn_hyperparameters)
        valid_predictions_op = predict_with_dataset(tf_valid_data_batch, tf_cnn_hyperparameters)

        test_predicitons_op = predict_with_dataset(tf_test_dataset, tf_cnn_hyperparameters)

        # GLOBAL: Structure adaptation
        with tf.name_scope(TF_ADAPTATION_NAME_SCOPE):
            if research_parameters['adapt_structure']:
                # Tensorflow operations that are defined one for each convolution operation
                tf_indices = tf.placeholder(dtype=tf.int32, shape=(None,), name='optimize_indices')
                tf_indices_size = tf.placeholder(tf.int32)
                tf_indices_to_rm = tf.placeholder(dtype=tf.int32, shape=[None], name = 'indices_to_rm')

                tf_action_info = tf.placeholder(shape=[3], dtype=tf.int32,
                                                name='tf_action')  # [op_id,action_id,amount] (action_id 0 - add, 1 -remove)

                logger.info('Defining placeholders for newly added parameters')
                # Adaptation placeholders for convolution layers
                tf_weights_this = tf.placeholder(shape=[None, None, None, None], dtype=tf.float32,
                                                 name='new_weights_current')
                tf_bias_this = tf.placeholder(shape=(None,), dtype=tf.float32, name='new_bias_current')
                tf_weights_next = tf.placeholder(shape=[None, None, None, None], dtype=tf.float32,
                                                 name='new_weights_next')

                tf_wvelocity_this = tf.placeholder(shape=[None, None, None, None], dtype=tf.float32,
                                                   name='new_weights_velocity_current')
                tf_bvelocity_this = tf.placeholder(shape=(None,), dtype=tf.float32,
                                                   name='new_bias_velocity_current')
                tf_wvelocity_next = tf.placeholder(shape=[None, None, None, None], dtype=tf.float32,
                                                   name='new_weights_velocity_next')

                tf_act_this = tf.placeholder(shape=(None,), dtype=tf.float32, name='new_activations_current')

                logger.info('Defining weight shapes for setting shapes after adaptations')
                tf_weight_shape = tf.placeholder(shape=[4], dtype=tf.int32, name='weight_shape')
                tf_in_size = tf.placeholder(dtype=tf.int32, name='fulcon_input_size')
                tf_out_size = tf.placeholder(dtype=tf.int32, name='fulcon_output_size')
                tf_replicative_factor_vec = tf.placeholder(dtype=tf.float32, shape=[None], name='tf_replicative_factor')

                tf_reset_cnn = cnn_intializer_continuous.reset_cnn(init_cnn_hyperparameters, cnn_ops)

                for op in cnn_ops:
                    if 'pool' in op:
                        continue
                    else:
                        tf_retain_id_placeholders[op] = {'in':None, 'out':None}
                        tf_retain_id_placeholders[op]['in'] = tf.placeholder(dtype=tf.int32,shape=[None],name='weight_mean_placeholder_in_'+op)
                        tf_retain_id_placeholders[op]['out'] = tf.placeholder(dtype=tf.int32, shape=[None],
                                                                            name='weight_mean_placeholder_out_' + op)
                #tf_reset_cnn_custom = cnn_intializer.reset_cnn_preserve_weights_custom(tf_prune_cnn_hyperparameters,cnn_ops,tf_retain_id_placeholders)

                for tmp_op in cnn_ops:
                    # Convolution related adaptation operations
                    if 'conv' in tmp_op:
                        tf_update_hyp_ops[tmp_op] = ada_cnn_adapter.update_tf_hyperparameters(tmp_op, tf_weight_shape, tf_in_size, tf_out_size)
                        tf_add_filters_ops[tmp_op] = ada_cnn_adapter.add_with_action(tmp_op, tf_action_info, tf_weights_this,
                                                                     tf_bias_this, tf_weights_next,
                                                                     tf_wvelocity_this, tf_bvelocity_this,
                                                                     tf_wvelocity_next,tf_replicative_factor_vec, tf_act_this)

                        tf_slice_optimize[tmp_op], tf_slice_vel_update[tmp_op] = cnn_optimizer.optimize_masked_momentum_gradient(
                            optimizer, tf_indices,
                            tmp_op, tf_pool_avg_gradvars, tf_cnn_hyperparameters,
                            tf_learning_rate, global_step
                        )

                        tf_rm_filters_ops[tmp_op] = ada_cnn_adapter.remove_with_action(tmp_op, tf_action_info, tf_cnn_hyperparameters, tf_indices_to_rm)

                    # Fully connected realted adaptation operations
                    elif 'fulcon' in tmp_op:
                        # Update hyp operation is required for fulcon_out
                        tf_update_hyp_ops[tmp_op] = ada_cnn_adapter.update_tf_hyperparameters(tmp_op, tf_weight_shape, tf_in_size, tf_out_size)

                        if tmp_op!='fulcon_out':
                            tf_add_filters_ops[tmp_op] = ada_cnn_adapter.add_to_fulcon_with_action(
                                tmp_op, tf_action_info,tf_weights_this, tf_bias_this, tf_weights_next,
                                tf_wvelocity_this,tf_bvelocity_this, tf_wvelocity_next, tf_replicative_factor_vec, tf_act_this
                            )

                            tf_slice_optimize[tmp_op], tf_slice_vel_update[
                                tmp_op] = cnn_optimizer.optimize_masked_momentum_gradient_for_fulcon(
                                optimizer, tf_indices,
                                tmp_op, tf_pool_avg_gradvars, tf_cnn_hyperparameters,
                                tf_learning_rate, global_step
                            )

                            tf_rm_filters_ops[tmp_op] = ada_cnn_adapter.remove_from_fulcon(tmp_op, tf_action_info,
                                                                                           tf_cnn_hyperparameters,
                                                                                           tf_indices_to_rm)

def check_several_conditions_with_assert(num_gpus):
    batches_in_chunk = model_hyperparameters['chunk_size']//model_hyperparameters['batch_size']
    assert batches_in_chunk % num_gpus == 0
    assert num_gpus > 0


def distort_img(img):
    if np.random.random()<0.4:
        img = np.fliplr(img)
    if np.random.random()<0.4:
        brightness = np.random.random()*1.5 - 0.6
        img += brightness
    if np.random.random()<0.4:
        contrast = np.random.random()*0.8 + 0.4
        img *= contrast

    return img

def tf_augment_data_with(tf_pool_batch, dataset_type):

    if dataset_type != 'svhn-10':
        tf_aug_pool_batch = tf.map_fn(lambda data: tf.image.random_flip_left_right(data), tf_pool_batch)

    tf_aug_pool_batch = tf.image.random_brightness(tf_aug_pool_batch,0.5)
    tf_aug_pool_batch = tf.image.random_contrast(tf_aug_pool_batch,0.5,1.5)

    # Not necessary they are already normalized
    #tf_image_batch = tf.map_fn(lambda img: tf.image.per_image_standardization(img), tf_image_batch)

    return tf_aug_pool_batch


def get_pool_valid_accuracy(hard_pool_valid):
    global pool_dataset, pool_labels, pool_pred
    tmp_pool_accuracy = []
    pool_dataset, pool_labels = hard_pool_valid.get_pool_data(False)
    for pool_id in range(0,(hard_pool_valid.get_size() // batch_size)):
        pbatch_data = pool_dataset[pool_id * batch_size:(pool_id + 1) * batch_size, :, :, :]
        pbatch_labels = pool_labels[pool_id * batch_size:(pool_id + 1) * batch_size, :]
        pool_feed_dict = {tf_pool_data_batch[0]: pbatch_data,
                          tf_pool_label_batch[0]: pbatch_labels}

        p_predictions = session.run(pool_pred, feed_dict=pool_feed_dict)
        tmp_pool_accuracy.append(accuracy(p_predictions, pbatch_labels))

    return np.mean(tmp_pool_accuracy)


def get_adaptive_dropout():
    global cnn_hyperparameters,cnn_ops, dropout_rate,filter_vector
    current_conv_depth_total, current_fulcon_depth_total = 0, 0
    conv_filter_vec, fulcon_filter_vec = [],[]
    is_fulcon_exist = False
    for op_i, op in enumerate(cnn_ops):
        if 'conv' in op:
            conv_filter_vec.append(filter_vector[op_i])
        elif 'fulcon_out'!= op and 'fulcon' in op:
            fulcon_filter_vec.append(filter_vector[op_i])
            is_fulcon_exist = True

    for scope in cnn_ops:
        if 'conv' in scope:
            current_conv_depth_total += cnn_hyperparameters[scope]['weights'][3]
        if scope!='fulcon_out' and 'fulcon' in scope:
            current_fulcon_depth_total += cnn_hyperparameters[scope]['out']

    if is_fulcon_exist:
        dropout_factor =  ((current_conv_depth_total*1.0/sum(conv_filter_vec)) + (current_fulcon_depth_total*1.0/sum(fulcon_filter_vec)))/2.0
    else:
        dropout_factor = (current_conv_depth_total * 1.0 / sum(conv_filter_vec))

    return dropout_rate*(dropout_factor**2)



def get_new_distorted_weights(new_curr_weights,curr_weight_shape):
    if np.random.random() < 0.25:
        new_curr_weights = np.flip(new_curr_weights, axis=np.random.choice([0, 1]))
    if np.random.random() < 0.25:
        new_curr_weights = np.swapaxes(new_curr_weights, 0, 1)
    if np.random.random() < 0.8:
        if np.random.random() < 0.5:
            translate_amout = np.random.choice([1, 2, 3])
            new_curr_weights = np.pad(new_curr_weights, ((translate_amout, 0), (0, 0), (0, 0), (0, 0)), 'mean')
            new_curr_weights = new_curr_weights[:curr_weight_shape[0], :, :, :]
        if np.random.random() < 0.5:
            translate_amout = np.random.choice([1, 2, 3])
            new_curr_weights = np.pad(new_curr_weights, ((0, translate_amout), (0, 0), (0, 0), (0, 0)), 'mean')
            new_curr_weights = new_curr_weights[translate_amout:, :, :, :]
        if np.random.random() < 0.5:
            translate_amout = np.random.choice([1, 2, 3])
            new_curr_weights = np.pad(new_curr_weights, ((0, 0), (translate_amout, 0), (0, 0), (0, 0)), 'mean')
            new_curr_weights = new_curr_weights[:, :curr_weight_shape[1], :, :]
        if np.random.random() < 0.5:
            translate_amout = np.random.choice([1, 2, 3])
            new_curr_weights = np.pad(new_curr_weights, ((0, 0), (0, translate_amout), (0, 0), (0, 0)), 'mean')
            new_curr_weights = new_curr_weights[:, translate_amout:, :, :]

    return new_curr_weights

def run_actual_add_operation(session, current_op, li, last_conv_id, hard_pool_ft,amount_to_add, epoch, random_add):
    '''
    Run the add operation using the given Session
    :param session:
    :param current_op:
    :param li:
    :param last_conv_id:
    :param hard_pool_ft:
    :return:
    '''
    global current_adaptive_dropout,cnn_hyperparameters

    logger.info('Running add operation for %s by adding %d from %d(total)',current_op,amount_to_add,cnn_hyperparameters[current_op]['weights'][3])
    if current_op != last_conv_id:
        next_conv_op = \
            [tmp_op for tmp_op in cnn_ops[cnn_ops.index(current_op) + 1:] if 'conv' in tmp_op][0]


    with tf.variable_scope(TF_GLOBAL_SCOPE, reuse=True):

        with tf.variable_scope(current_op,reuse=True):
            curr_weights = tf.get_variable(TF_WEIGHTS).eval()
            logger.debug('\tCurrent weight size: %s',curr_weights.shape)
            with tf.variable_scope(TF_WEIGHTS, reuse=True):
                w_vel = tf.get_variable(constants.TF_TRAIN_MOMENTUM).eval()
                logger.debug('\tCurrent weight vel shape: %s',w_vel.shape)
            curr_bias = tf.get_variable(TF_BIAS).eval()
            logger.debug('\tCurrent bias size: %s', curr_bias.shape)
            curr_act = tf.get_variable(TF_ACTIVAIONS_STR).eval()
            logger.debug('\tCurrent mean activation size: %s', curr_act.shape)

        if current_op != last_conv_id:
            with tf.variable_scope(next_conv_op, reuse=True):
                next_weights = tf.get_variable(TF_WEIGHTS).eval()
                logger.debug('\tNext weight size: %s', next_weights.shape)
        else:
            with tf.variable_scope(first_fc, reuse=True):
                next_weights = tf.get_variable(TF_WEIGHTS).eval()
                logger.debug('\tNext weight size: %s', next_weights.shape)

        curr_weight_shape = curr_weights.shape
        next_weights_shape = next_weights.shape

        #Net2Net type initialization
        if random_add<0.25:
            print('Net2Net Initialization')
            rand_indices_1 = np.random.choice(np.arange(curr_weights.shape[3]).tolist(),size=amount_to_add,replace=True)
            rand_indices_2 = np.random.choice(np.arange(curr_weights.shape[3]).tolist(), size=amount_to_add, replace=True)

            #all_indices_plus_rand = np.concatenate([np.arange(0,curr_weights.shape[3]).ravel(), np.asarray(rand_indices_1).ravel(), np.asarray(rand_indices_2).ravel()])

            #ind_counter = Counter(all_indices_plus_rand.tolist())
            #sorted_keys = sorted(ind_counter.keys())
            #count_vec = np.asarray([ind_counter[k] for k in sorted_keys ])
            #count_vec = np.concatenate([count_vec,(count_vec[rand_indices_1]+count_vec[rand_indices_2])/2.0])

            count_vec = np.ones((curr_weight_shape[3] + amount_to_add), dtype=np.float32)
            print('count vec',count_vec.shape)
            print(count_vec)
            new_curr_weights = (curr_weights[:,:,:,rand_indices_1] + curr_weights[:,:,:,rand_indices_2])/2.0
            new_curr_weights = get_new_distorted_weights(new_curr_weights,curr_weight_shape)

            new_curr_bias = (curr_bias[rand_indices_1] + curr_bias[rand_indices_2])/2.0

            new_act_this = (curr_act[rand_indices_1] + curr_act[rand_indices_2]) / 2.0

            if last_conv_id != current_op:
                new_next_weights = (next_weights[:,:,rand_indices_1,:] + next_weights[:,:,rand_indices_2,:])/2.0
                #new_next_weights = next_weights[:, :, rand_indices, :]
            else:
                low_bound_1 = (rand_indices_1*final_2d_width*final_2d_width).tolist()
                upper_bound_1 = ((rand_indices_1+1) * final_2d_width * final_2d_width).tolist()
                low_up_bounds_1 = list(zip(low_bound_1,upper_bound_1))
                all_indices_1 = [np.arange(l,u) for (l,u) in low_up_bounds_1]
                all_indices_1 = np.stack(all_indices_1).ravel()

                low_bound_2 = (rand_indices_2 * final_2d_width * final_2d_width).tolist()
                upper_bound_2 = ((rand_indices_2 + 1) * final_2d_width * final_2d_width).tolist()
                low_up_bounds_2 = list(zip(low_bound_2, upper_bound_2))
                all_indices_2 = [np.arange(l, u) for (l, u) in low_up_bounds_2]
                all_indices_2 = np.stack(all_indices_2).ravel()

                new_next_weights = (next_weights[all_indices_1,:] + next_weights[all_indices_2,:])/2.0
                new_next_weights = np.expand_dims(np.expand_dims(new_next_weights,-1),-1)

        # Random initialization
        else:
            print('Random Initialization')
            rand_scale = 0.001
            new_curr_weights = np.random.uniform(low=-rand_scale, high=rand_scale, size=(
            curr_weight_shape[0], curr_weight_shape[1], curr_weight_shape[2], amount_to_add))
            new_curr_bias = np.random.uniform(low=-rand_scale, high=rand_scale, size=(amount_to_add))

            new_act_this = np.zeros(shape=(amount_to_add), dtype=np.float32)

            if last_conv_id != current_op:

                new_next_weights = np.random.uniform(low=-rand_scale, high=rand_scale,
                                                     size=(next_weights_shape[0],next_weights_shape[1],amount_to_add, next_weights_shape[3]))
            else:
                new_next_weights = np.random.uniform(low=-rand_scale, high=rand_scale,
                                                     size=(amount_to_add *final_2d_width *final_2d_width, next_weights_shape[1],1,1))
            count_vec = np.ones((curr_weight_shape[3] + amount_to_add), dtype=np.float32)


    _ = session.run(tf_add_filters_ops[current_op],
                    feed_dict={
                        tf_action_info: np.asarray([li, 1, amount_to_add]),
                        tf_weights_this: new_curr_weights,
                        tf_bias_this: new_curr_bias,

                        tf_weights_next: new_next_weights,
                        tf_replicative_factor_vec: count_vec,
                        tf_wvelocity_this: np.zeros(shape=(
                            cnn_hyperparameters[current_op]['weights'][0],
                            cnn_hyperparameters[current_op]['weights'][1],
                            cnn_hyperparameters[current_op]['weights'][2], amount_to_add),dtype=np.float32),
                        tf_bvelocity_this: np.zeros(shape=(amount_to_add,),dtype=np.float32),
                        tf_wvelocity_next: np.zeros(shape=(
                            cnn_hyperparameters[next_conv_op]['weights'][0],
                            cnn_hyperparameters[next_conv_op]['weights'][1],
                            amount_to_add, cnn_hyperparameters[next_conv_op]['weights'][3]),dtype=np.float32) if last_conv_id != current_op else
                        np.zeros(shape=(final_2d_width * final_2d_width * amount_to_add,
                                        cnn_hyperparameters[first_fc]['out'], 1, 1),dtype=np.float32),
                        tf_act_this: new_act_this
                    })

    # change both weights and biase in the current op
    logger.debug('\tAdding %d new weights', amount_to_add)

    with tf.variable_scope(TF_GLOBAL_SCOPE + TF_SCOPE_DIVIDER + current_op,
                           reuse=True) as scope:
        current_op_weights = tf.get_variable(TF_WEIGHTS)

    if research_parameters['debugging']:
        logger.debug('\tSummary of changes to weights of %s ...', current_op)
        logger.debug('\t\tNew Weights: %s', str(tf.shape(current_op_weights).eval()))

    # change out hyperparameter of op
    cnn_hyperparameters[current_op]['weights'][3] += amount_to_add
    if research_parameters['debugging']:
        assert cnn_hyperparameters[current_op]['weights'][2] == \
               tf.shape(current_op_weights).eval()[2]

    session.run(tf_update_hyp_ops[current_op], feed_dict={
        tf_weight_shape: cnn_hyperparameters[current_op]['weights']
    })

    if current_op == last_conv_id:

        with tf.variable_scope(TF_GLOBAL_SCOPE + TF_SCOPE_DIVIDER + first_fc, reuse=True):
            first_fc_weights = tf.get_variable(TF_WEIGHTS)
        cnn_hyperparameters[first_fc]['in'] += final_2d_width * final_2d_width * amount_to_add

        if research_parameters['debugging']:
            logger.debug('\tNew %s in: %d', first_fc, cnn_hyperparameters[first_fc]['in'])
            logger.debug('\tSummary of changes to weights of %s', first_fc)
            logger.debug('\t\tNew Weights: %s', str(tf.shape(first_fc_weights).eval()))

        session.run(tf_update_hyp_ops[first_fc], feed_dict={
            tf_in_size: cnn_hyperparameters[first_fc]['in'],
            tf_out_size: cnn_hyperparameters[first_fc]['out']
        })

    else:

        next_conv_op = \
            [tmp_op for tmp_op in cnn_ops[cnn_ops.index(current_op) + 1:] if 'conv' in tmp_op][
                0]
        assert current_op != next_conv_op

        with tf.variable_scope(TF_GLOBAL_SCOPE + TF_SCOPE_DIVIDER + next_conv_op, reuse=True):
            next_conv_op_weights = tf.get_variable(TF_WEIGHTS)

        if research_parameters['debugging']:
            logger.debug('\tSummary of changes to weights of %s', next_conv_op)
            logger.debug('\t\tCurrent Weights: %s', str(tf.shape(next_conv_op_weights).eval()))

        cnn_hyperparameters[next_conv_op]['weights'][2] += amount_to_add

        if research_parameters['debugging']:
            assert cnn_hyperparameters[next_conv_op]['weights'][2] == \
                   tf.shape(next_conv_op_weights).eval()[2]

        session.run(tf_update_hyp_ops[next_conv_op], feed_dict={
            tf_weight_shape: cnn_hyperparameters[next_conv_op]['weights']
        })

    # optimize the newly added fiterls only
    pool_dataset, pool_labels = hard_pool_ft.get_pool_data(True)


    current_adaptive_dropout = get_adaptive_dropout()
    # This is a pretty important step
    # Unless you run this onces, the sizes of weights do not change
    train_feed_dict.update({tf_dropout_rate:current_adaptive_dropout})
    _ = session.run([tower_logits], feed_dict=train_feed_dict)
    pbatch_train_count = 0

    # Train only with half of the batch
    for pool_id in range(0, (hard_pool_ft.get_size() // batch_size) - 1, num_gpus):
        if np.random.random() < research_parameters['finetune_rate']:
            pbatch_data, pbatch_labels = [], []

            pool_feed_dict = {
                tf_indices: np.arange(cnn_hyperparameters[current_op]['weights'][3] - amount_to_add,
                                      cnn_hyperparameters[current_op]['weights'][3])}
            pool_feed_dict.update({tf_dropout_rate:current_adaptive_dropout})

            for gpu_id in range(num_gpus):
                pbatch_data.append(pool_dataset[(pool_id + gpu_id) * batch_size:(
                                                                                    pool_id + gpu_id + 1) * batch_size,
                                   :, :, :])
                pbatch_labels.append(pool_labels[(pool_id + gpu_id) * batch_size:(
                                                                                     pool_id + gpu_id + 1) * batch_size,
                                     :])
                pool_feed_dict.update({tf_pool_data_batch[gpu_id]: pbatch_data[-1],
                                       tf_pool_label_batch[gpu_id]: pbatch_labels[-1]})

            pool_feed_dict.update({tf_learning_rate:model_hyperparameters['start_lr']})
            _, _ = session.run([tf_slice_optimize[current_op], tf_slice_vel_update[current_op]],
                               feed_dict=pool_feed_dict)

    _ = session.run([tower_logits], feed_dict=train_feed_dict)


def run_actual_add_operation_for_fulcon(session, current_op, li, last_conv_id, hard_pool_ft,amount_to_add,epoch, random_add):
    '''
    Run the add operation using the given Session
    :param session:
    :param current_op:
    :param li:
    :param last_conv_id:
    :param hard_pool_ft:
    :return:
    '''
    global current_adaptive_dropout

    logger.info('Running add operation (fulcon) for %s by adding %d from %d', current_op, amount_to_add,cnn_hyperparameters[current_op]['out'])
    if current_op != last_conv_id:
        next_fulcon_op = \
            [tmp_op for tmp_op in cnn_ops[cnn_ops.index(current_op) + 1:]][0]


    with tf.variable_scope(TF_GLOBAL_SCOPE, reuse=True):

        with tf.variable_scope(current_op,reuse=True):
            curr_weights = tf.get_variable(TF_WEIGHTS).eval()
            logger.debug('\tCurrent weights shape: %s',curr_weights.shape)
            with tf.variable_scope(TF_WEIGHTS, reuse=True):
                w_vel = tf.get_variable(constants.TF_TRAIN_MOMENTUM).eval()
                logger.debug('\tCurrent weight vel shape: %s',w_vel.shape)
            curr_bias = tf.get_variable(TF_BIAS).eval()
            logger.debug('\tCurrent bias shape: %s', curr_bias.shape)
            curr_act = tf.get_variable(TF_ACTIVAIONS_STR).eval()
            logger.debug('\tCurrent mean activation shape: %s', curr_act.shape)

        with tf.variable_scope(next_fulcon_op, reuse=True):
            next_weights = tf.get_variable(TF_WEIGHTS).eval()

        curr_weight_shape = curr_weights.shape
        next_weights_shape = next_weights.shape

        # Net2Net Initialization
        if random_add<0.25:
            rand_indices_1 = np.random.choice(np.arange(curr_weights.shape[1]).tolist(),size=amount_to_add,replace=True)
            rand_indices_2 = np.random.choice(np.arange(curr_weights.shape[1]).tolist(), size=amount_to_add, replace=True)

            #all_indices_plus_rand = np.concatenate([np.arange(0,curr_weights.shape[1]).ravel(), np.asarray(rand_indices_1).ravel(), np.asarray(rand_indices_2).ravel()])
            #print('allindices plus rand',all_indices_plus_rand.shape)
            #ind_counter = Counter(all_indices_plus_rand.tolist())
            #sorted_keys = np.asarray(sorted(ind_counter.keys()))
            #count_vec = np.asarray([ind_counter[k] for k in sorted_keys])
            #count_vec = np.concatenate([count_vec,(count_vec[rand_indices_1]+count_vec[rand_indices_2])/2.0])
            count_vec = np.ones((curr_weight_shape[1] + amount_to_add), dtype=np.float32)

            print('count vec',count_vec.shape)
            print(count_vec)
            new_curr_weights = np.expand_dims(np.expand_dims((curr_weights[:,rand_indices_1]+curr_weights[:,rand_indices_2])/2.0,-1),-1)
            new_curr_bias = (curr_bias[rand_indices_1] + curr_bias[rand_indices_2])/2.0

            new_curr_act = (curr_act[rand_indices_1] + curr_act[rand_indices_2]) / 2.0

            new_next_weights = (next_weights[rand_indices_1,:] + next_weights[rand_indices_2,:])/2.0
            new_next_weights = np.expand_dims(np.expand_dims(new_next_weights,-1),-1)

        else:
            rand_scale = 0.001
            new_curr_weights = np.random.uniform(low=-rand_scale,high=rand_scale,size=(curr_weight_shape[0],amount_to_add,1,1))
            new_curr_bias = np.random.uniform(low=-rand_scale,high=rand_scale,size=(amount_to_add))

            new_curr_act = np.zeros(shape=(amount_to_add), dtype=np.float32)

            new_next_weights = np.random.uniform(low=-rand_scale,high=rand_scale,size=(amount_to_add,next_weights_shape[1],1,1))
            count_vec = np.ones((curr_weight_shape[1]+amount_to_add),dtype=np.float32)

    _ = session.run(tf_add_filters_ops[current_op],
                    feed_dict={
                        tf_action_info: np.asarray([li, 1, amount_to_add]),
                        tf_weights_this: new_curr_weights,
                        tf_bias_this: new_curr_bias,
                        tf_replicative_factor_vec: count_vec,
                        tf_weights_next: new_next_weights,
                        tf_wvelocity_this: np.zeros(
                            shape=(curr_weight_shape[0],amount_to_add,1,1),
                            dtype=np.float32),
                        tf_bvelocity_this: np.zeros(shape=(amount_to_add),dtype=np.float32),
                        tf_wvelocity_next: np.zeros(
                            shape=(amount_to_add, next_weights_shape[1],1,1),
                            dtype=np.float32),
                        tf_act_this:new_curr_act
                    })

    # change both weights and biase in the current op
    logger.debug('\tAdding %d new weights', amount_to_add)

    with tf.variable_scope(TF_GLOBAL_SCOPE,reuse=True) as scope:
        with tf.variable_scope(current_op, reuse=True):
            current_op_weights = tf.get_variable(TF_WEIGHTS).eval()
            with tf.variable_scope(TF_WEIGHTS, reuse=True):
                current_w_vel = tf.get_variable(constants.TF_TRAIN_MOMENTUM).eval()

            assert current_op_weights.shape[1] == current_w_vel.shape[1]

        with tf.variable_scope(next_fulcon_op, reuse=True):
            next_fulcon_op_weights = tf.get_variable(TF_WEIGHTS).eval()
            with tf.variable_scope(TF_WEIGHTS, reuse=True):
                next_w_vel = tf.get_variable(constants.TF_TRAIN_MOMENTUM).eval()

            assert next_fulcon_op_weights.shape[0] == next_w_vel.shape[0]

    if research_parameters['debugging']:
        logger.debug('\tSummary of changes to weights of %s ...', current_op)
        logger.debug('\t\tNew Weights: %s', current_op_weights.shape)
        logger.debug('\t\tNew Weights vel: %s', current_w_vel.shape)
        logger.debug('\t\tNext weights: %s',next_fulcon_op_weights.shape)
        logger.debug('\t\tNext Weights vel: %s',next_w_vel.shape)

    # change out hyperparameter of op
    cnn_hyperparameters[current_op]['out'] += amount_to_add

    session.run(tf_update_hyp_ops[current_op], feed_dict={
        tf_in_size: cnn_hyperparameters[current_op]['in'],
        tf_out_size: cnn_hyperparameters[current_op]['out']
    })

    next_fulcon_op = cnn_ops[cnn_ops.index(current_op) + 1]
    assert current_op != next_fulcon_op

    if research_parameters['debugging']:
        logger.debug('\tSummary of changes to weights of %s', next_fulcon_op)
        logger.debug('\t\tCurrent Weights: %s', str(tf.shape(next_fulcon_op_weights).eval()))

    cnn_hyperparameters[next_fulcon_op]['in'] += amount_to_add

    if research_parameters['debugging']:
        assert cnn_hyperparameters[next_fulcon_op]['in'] == \
               tf.shape(next_fulcon_op_weights).eval()[0]

    session.run(tf_update_hyp_ops[next_fulcon_op], feed_dict={
        tf_in_size: cnn_hyperparameters[next_fulcon_op]['in'],
        tf_out_size: cnn_hyperparameters[next_fulcon_op]['out']
    })

    # optimize the newly added fiterls only
    pool_dataset, pool_labels = hard_pool_ft.get_pool_data(True)

    current_adaptive_dropout = get_adaptive_dropout()
    # This is a pretty important step
    # Unless you run this onces, the sizes of weights do not change
    train_feed_dict.update({tf_dropout_rate:current_adaptive_dropout})
    _ = session.run([tower_logits], feed_dict=train_feed_dict)
    pbatch_train_count = 0

    # Train only newly added parameters
    for pool_id in range(0, (hard_pool_ft.get_size() // batch_size) - 1, num_gpus):
        if np.random.random() < research_parameters['finetune_rate']:
            pbatch_data, pbatch_labels = [], []

            pool_feed_dict = {
                tf_indices: np.arange(cnn_hyperparameters[current_op]['out'] - amount_to_add,
                                      cnn_hyperparameters[current_op]['out'])}
            pool_feed_dict.update({tf_dropout_rate:current_adaptive_dropout})

            for gpu_id in range(num_gpus):
                pbatch_data.append(pool_dataset[(pool_id + gpu_id) * batch_size:(
                                                                                    pool_id + gpu_id + 1) * batch_size,
                                   :, :, :])
                pbatch_labels.append(pool_labels[(pool_id + gpu_id) * batch_size:(
                                                                                     pool_id + gpu_id + 1) * batch_size,
                                     :])
                pool_feed_dict.update({tf_pool_data_batch[gpu_id]: pbatch_data[-1],
                                       tf_pool_label_batch[gpu_id]: pbatch_labels[-1]})

            pool_feed_dict.update({tf_learning_rate: model_hyperparameters['start_lr']})
            _, _ = session.run([tf_slice_optimize[current_op], tf_slice_vel_update[current_op]],feed_dict=pool_feed_dict)

            pbatch_train_count += 1


    _ = session.run([tower_logits], feed_dict=train_feed_dict)


def run_actual_remove_operation(session, current_op, li, last_conv_id, hard_pool_ft, amount_to_rmv):
    global current_adaptive_dropout

    rm_indices = np.random.choice(list(range(cnn_hyperparameters[current_op]['weights'][3])),replace=False,size=(amount_to_rmv))
    logger.info('Running remove operation for %s by removing %d filetrs from %d (total)',
                current_op, amount_to_rmv,cnn_hyperparameters[current_op]['weights'][3])
    logger.debug('Removing indices')
    logger.debug('\t%s', rm_indices)

    _ = session.run(tf_rm_filters_ops[current_op],
                    feed_dict={
                        tf_action_info: np.asarray([li, 1, amount_to_rmv]),
                        tf_indices_to_rm: rm_indices
                    })

    with tf.variable_scope(TF_GLOBAL_SCOPE, reuse=True):
        with tf.variable_scope(current_op, reuse=True):
            current_op_weights = tf.get_variable(TF_WEIGHTS).eval()
            logger.debug('\tCurrent weights shape: %s',current_op_weights.shape)
            with tf.variable_scope(TF_WEIGHTS, reuse=True):
                w_vel = tf.get_variable(constants.TF_TRAIN_MOMENTUM).eval()
                logger.debug('\tCurrent weight vel shape: %s',w_vel.shape)
            current_op_bias = tf.get_variable(TF_BIAS).eval()
            logger.debug('\tCurrent bias shape: %s', current_op_bias.shape)

    cnn_hyperparameters[current_op]['weights'][3] -= amount_to_rmv
    if research_parameters['debugging']:
        logger.debug('\tSize after feature map reduction: %s,%s', current_op,
                     current_op_weights.shape)
        assert current_op_weights.shape[3] == \
               cnn_hyperparameters[current_op]['weights'][3], \
            'Shapes of weights (%s) did not match (Actual: %s),(Maintained: %s'\
            %(current_op,current_op_weights.shape,cnn_hyperparameters[current_op]['weights'])

    session.run(tf_update_hyp_ops[current_op], feed_dict={
        tf_weight_shape: cnn_hyperparameters[current_op]['weights']
    })

    if current_op == last_conv_id:
        with tf.variable_scope(TF_GLOBAL_SCOPE + TF_SCOPE_DIVIDER + first_fc, reuse=True):
            first_fc_weights = tf.get_variable(TF_WEIGHTS)
            logger.debug('\tNext weights shape: %s', first_fc_weights.eval().shape)

        cnn_hyperparameters[first_fc]['in'] -= final_2d_width * final_2d_width * amount_to_rmv
        if research_parameters['debugging']:
            logger.debug('\tSize after feature map reduction: %s,%s',
                         first_fc, tf.shape(first_fc_weights).eval())

        session.run(tf_update_hyp_ops[first_fc], feed_dict={
            tf_in_size: cnn_hyperparameters[first_fc]['in'],
            tf_out_size: cnn_hyperparameters[first_fc]['out']
        })

    else:
        next_conv_op = \
            [tmp_op for tmp_op in cnn_ops[cnn_ops.index(current_op) + 1:] if 'conv' in tmp_op][0]
        assert current_op != next_conv_op

        with tf.variable_scope(TF_GLOBAL_SCOPE + TF_SCOPE_DIVIDER + next_conv_op, reuse=True):
            next_conv_op_weights = tf.get_variable(TF_WEIGHTS)
            logger.debug('\tNext weights shape: %s', next_conv_op_weights.eval().shape)
        cnn_hyperparameters[next_conv_op]['weights'][2] -= amount_to_rmv

        logger.debug('\tSize after feature map reduction: %s,%s', next_conv_op,
                     str(tf.shape(next_conv_op_weights).eval()))
        assert tf.shape(next_conv_op_weights).eval()[2] == \
               cnn_hyperparameters[next_conv_op]['weights'][2]

        session.run(tf_update_hyp_ops[next_conv_op], feed_dict={
            tf_weight_shape: cnn_hyperparameters[next_conv_op]['weights']
        })

    current_adaptive_dropout = get_adaptive_dropout()
    # This is a pretty important step
    # Unless you run this onces, the sizes of weights do not change

    _ = session.run([tower_logits], feed_dict=train_feed_dict)


def run_actual_remove_operation_for_fulcon(session, current_op, li, last_conv_id, hard_pool_ft, amount_to_rmv):

    global current_adaptive_dropout
    rm_indices = np.random.choice(list(range(cnn_hyperparameters[current_op]['out'])), replace=False, size=(amount_to_rmv))

    logger.info('Running remove (fulcon) operation for %s by removing %d from %d',
                current_op, amount_to_rmv, cnn_hyperparameters[current_op]['out'])
    logger.debug('Removing indices')
    logger.debug('\t%s',rm_indices)

    _ = session.run(tf_rm_filters_ops[current_op],
                    feed_dict={
                        tf_action_info: np.asarray([li, 1, amount_to_rmv]),
                        tf_indices_to_rm: rm_indices
                    })

    with tf.variable_scope(TF_GLOBAL_SCOPE + TF_SCOPE_DIVIDER + current_op, reuse=True):
        current_op_weights = tf.get_variable(TF_WEIGHTS)
        logger.debug('\tCurrent weights shape: %s', current_op_weights.eval().shape)
        with tf.variable_scope(TF_WEIGHTS, reuse=True):
            w_vel = tf.get_variable(constants.TF_TRAIN_MOMENTUM)
            logger.debug('\tCurrent weight vel shape: %s', w_vel.eval().shape)

    cnn_hyperparameters[current_op]['out'] -= amount_to_rmv


    logger.debug('\tSize after feature map reduction: %s,%s', current_op,
                 tf.shape(current_op_weights).eval())
    assert tf.shape(current_op_weights).eval()[1] == \
           cnn_hyperparameters[current_op]['out']

    session.run(tf_update_hyp_ops[current_op], feed_dict={
        tf_in_size: cnn_hyperparameters[current_op]['in'],
        tf_out_size: cnn_hyperparameters[current_op]['out']
    })

    next_fulcon_op = \
        [tmp_op for tmp_op in cnn_ops[cnn_ops.index(current_op) + 1:] if 'fulcon' in tmp_op][0]
    assert current_op != next_fulcon_op

    with tf.variable_scope(TF_GLOBAL_SCOPE + TF_SCOPE_DIVIDER + next_fulcon_op, reuse=True):
        next_fulcon_op_weights = tf.get_variable(TF_WEIGHTS)
        with tf.variable_scope(TF_WEIGHTS, reuse = True):
            next_w_vel = tf.get_variable(constants.TF_TRAIN_MOMENTUM)

    cnn_hyperparameters[next_fulcon_op]['in'] -= amount_to_rmv

    if research_parameters['debugging']:
        logger.debug('\tSummary of changes to weights of %s ...', current_op)
        logger.debug('\t\tNew Weights: %s', current_op_weights.eval().shape)
        logger.debug('\t\tNew Weights vel: %s', w_vel.eval().shape)
        logger.debug('\t\tNext weights: %s', next_fulcon_op_weights.eval().shape)
        logger.debug('\t\tNext Weights vel: %s', next_w_vel.eval().shape)

    logger.debug('\tSize after feature map reduction: %s,%s', next_fulcon_op,
                 tf.shape(next_fulcon_op_weights).eval())
    assert tf.shape(next_fulcon_op_weights).eval()[0] == \
           cnn_hyperparameters[next_fulcon_op]['in']

    session.run(tf_update_hyp_ops[next_fulcon_op], feed_dict={
        tf_in_size: cnn_hyperparameters[next_fulcon_op]['in'],
        tf_out_size: cnn_hyperparameters[next_fulcon_op]['out']
    })

    current_adaptive_dropout = get_adaptive_dropout()
    # This is a pretty important step
    # Unless you run this onces, the sizes of weights do not change
    _ = session.run([tower_logits], feed_dict=train_feed_dict)


def run_actual_finetune_operation(hard_pool_ft,rate):
    '''
    Run the finetune opeartion in the default session
    :param hard_pool_ft:
    :return:
    '''
    global current_adaptive_dropout

    pool_dataset, pool_labels = hard_pool_ft.get_pool_data(True)

    # without if can give problems in exploratory stage because of no data in the pool
    if hard_pool_ft.get_size() > batch_size:
        # Train with latter half of the data


        for pool_id in range(0, (hard_pool_ft.get_size() // batch_size) - 1, num_gpus):
            if np.random.random() < rate:
                #print('fintuning network (pool_id: ',pool_id,')')
                pool_feed_dict = {}
                pool_feed_dict.update({tf_dropout_rate:current_adaptive_dropout})
                for gpu_id in range(num_gpus):
                    pbatch_data = pool_dataset[(pool_id + gpu_id) * batch_size:(
                                                                                   pool_id + gpu_id + 1) * batch_size,
                                  :, :, :]
                    pbatch_labels = pool_labels[(pool_id + gpu_id) * batch_size:(
                                                                                    pool_id + gpu_id + 1) * batch_size,
                                    :]
                    pool_feed_dict.update({tf_pool_data_batch[gpu_id]: pbatch_data,
                                           tf_pool_label_batch[gpu_id]: pbatch_labels})

                if adapt_structure:
                    pool_feed_dict.update({tf_learning_rate: current_data_lr * model_hyperparameters['start_lr']})

                _, _ = session.run([apply_pool_grads_op, update_pool_velocity_ops],
                                   feed_dict=pool_feed_dict)


def top_n_accuracy(predictions,labels,n):
    '''
    Gives the top-n accuracy instead of top-1 accuracy
    Useful for large datasets
    :param predictions:
    :param labels:
    :param n:
    :return:
    '''
    assert predictions.shape[0] == labels.shape[0]
    correct_total = 0
    for pred_item, lbl_item in zip(predictions,labels):
        lbl_idx = int(np.argmax(lbl_item))
        top_preds = list(np.argsort(pred_item).flatten()[-n:])
        if lbl_idx in top_preds:
            correct_total += 1
    return (100.0 * correct_total)/predictions.shape[0]


def logging_hyperparameters(hyp_logger, cnn_hyperparameters, research_hyperparameters,
                            model_hyperparameters, interval_hyperparameters, dataset_info):

    hyp_logger.info('#Various hyperparameters')
    hyp_logger.info('# Initial CNN architecture related hyperparameters')
    hyp_logger.info(cnn_hyperparameters)
    hyp_logger.info('# Dataset info')
    hyp_logger.info(dataset_info)
    hyp_logger.info('# Research parameters')
    hyp_logger.info(research_hyperparameters)
    hyp_logger.info('# Interval parameters')
    hyp_logger.info(interval_hyperparameters)
    hyp_logger.info('# Model parameters')
    hyp_logger.info(model_hyperparameters)


def change_data_prior_to_introduce_new_labels_over_time(data_prior,n_tasks,n_iterations,labels_of_each_task,n_labels):
    '''
    We consider a group of labels as one task
    :param data_prior: probabilities for each class
    :param n_slices:
    :param n_iterations:
    :return:
    '''
    assert len(data_prior)==n_iterations
    iterations_per_task = n_iterations//n_tasks

    # Calculates the starting and ending positions of each task in the total iterations
    iterations_start_index_of_task = []
    for s_index in range(n_tasks):
        iterations_start_index_of_task.append(s_index*iterations_per_task)
    iterations_start_index_of_task.append(n_iterations) # last task
    assert n_tasks+1 == len(iterations_start_index_of_task)
    logger.info('The perids allocated for each slice')
    logger.info(iterations_start_index_of_task)

    # Creates a new prior according the the specified tasks
    new_data_prior = np.zeros(shape=(n_iterations,n_labels),dtype=np.float32)
    for task_id in range(n_tasks):
        print_i = 0
        start_idx, end_idx = iterations_start_index_of_task[task_id], iterations_start_index_of_task[task_id+1]
        logger.info('Preparing data from index %d to index %d',start_idx,end_idx)

        for dist_i in range(start_idx,end_idx):
            new_data_prior[dist_i,labels_of_each_task[task_id]] = data_prior[dist_i]
            if print_i < 2:
                logger.info('Sample label sequence')
                logger.info(new_data_prior[dist_i])
            print_i += 1

    assert new_data_prior.shape[0]==len(data_prior)
    del data_prior
    return new_data_prior


def init_tf_prune_cnn_hyperparameters():
    '''
    Initialize prune CNN hyperparameters with placeholders for each weight shape
    :return:
    '''
    global tf_prune_cnn_hyperparameters

    for op in cnn_ops:
        if 'conv' in op:
            tf_prune_cnn_hyperparameters[op] = {'weights':
                                   tf.placeholder(shape=[4],dtype=tf.int32)
                               }

        if 'fulcon' in op:

            tf_prune_cnn_hyperparameters[op] = {'in': tf.placeholder(shape=None, dtype=tf.int32),
                               'out': tf.placeholder(shape=None, dtype=tf.int32)
                                   }


def get_pruned_cnn_hyperparameters(current_cnn_hyperparams,prune_factor):
    '''
    Get pruned hyperparameters by pruning the current architecture with a factor
    Should look like below
    pruned_cnn_hyps = {conv_0:{'weights':[x,y,z,w]},'fulcon_out':{'in':x,'out':y}}
    :param current_cnn_hyperparams: Current CNN weight shapes
    :param prune_factor: How much of the network is to prune
    :return:
    '''
    global cnn_ops,first_fc,final_2d_width,model_hyperparameters
    pruned_cnn_hyps = {}
    prev_op = None
    prev_fulcon_op = None
    for op in cnn_ops:
        if 'pool' in op:
            pruned_cnn_hyps[op] = dict(current_cnn_hyperparams[op])

        if 'conv' in op:

            if op=='conv_0':
                pruned_cnn_hyps[op] = {'weights': list(current_cnn_hyperparams[op]['weights']),
                                       'stride': list(current_cnn_hyperparams[op]['stride']),
                                       'padding': 'SAME'}
                pruned_cnn_hyps[op]['weights'][3] = max([model_hyperparameters['filter_min_threshold'],
                                              int(pruned_cnn_hyps[op]['weights'][3]*prune_factor)])

            else:
                pruned_cnn_hyps[op] = {'weights': list(current_cnn_hyperparams[op]['weights']),
                                       'stride': list(current_cnn_hyperparams[op]['stride']),
                                       'padding': 'SAME'}
                pruned_cnn_hyps[op]['weights'][2] = pruned_cnn_hyps[prev_op]['weights'][3]
                pruned_cnn_hyps[op]['weights'][3] = max([model_hyperparameters['filter_min_threshold'],
                                                         int(pruned_cnn_hyps[op]['weights'][3] * prune_factor)])
            prev_op = op

        if 'fulcon' in op:

            # if first fc is fulcon_out
            if op==first_fc and op=='fulcon_out':
                pruned_cnn_hyps[op] = {'in':current_cnn_hyperparams[op]['in'],
                                       'out': current_cnn_hyperparams[op]['out']}
                pruned_cnn_hyps[op]['in']=final_2d_width*final_2d_width*pruned_cnn_hyps[prev_op]['weights'][3]
            # if first_fc is not fulcon_out
            elif op==first_fc:
                pruned_cnn_hyps[op] = {'in': current_cnn_hyperparams[op]['in'],
                                       'out': max([model_hyperparameters['fulcon_min_threshold'],
                                                   int(current_cnn_hyperparams[op]['out'] * prune_factor)])}
                pruned_cnn_hyps[op]['in'] = final_2d_width * final_2d_width * pruned_cnn_hyps[prev_op]['weights'][3]
            # for all layers not first_fc and fulcon_out
            elif op != 'fulcon_out':
                pruned_cnn_hyps[op] = {'in': pruned_cnn_hyps[prev_fulcon_op]['out'],
                                       'out': max([model_hyperparameters['fulcon_min_threshold'],
                                                   int(current_cnn_hyperparams[op]['out'] * prune_factor)])}
            # for fulcon_out if fulcon_out is not first_fc
            else:
                pruned_cnn_hyps[op] = {'in': pruned_cnn_hyps[prev_fulcon_op]['out'],
                                       'out': current_cnn_hyperparams[op]['out']}
            prev_fulcon_op = op

    return pruned_cnn_hyps


def get_pruned_ids_feed_dict(cnn_hyps,pruned_cnn_hyps,weight_means):
    global tf_retain_id_placeholders, cnn_ops,num_channels,final_2d_width,first_fc

    retain_ids_feed_dict = {}
    prev_op = None
    for op in cnn_ops:
        logger.info('Calculating prune ids for %s (op) %s (prev_op)',op,prev_op)
        if 'pool' in op:
            continue
        elif 'conv' in op:

            # Out pruning
            amount_to_retain_out_ch = pruned_cnn_hyps[op]['weights'][3]
            logger.info('\tAmount of filers to retain for %s: %d (out)',op,amount_to_retain_out_ch)
            op_weight_means = weight_means[op]
            op_retain_ids_out = np.argsort(op_weight_means).ravel()[-amount_to_retain_out_ch:]
            retain_ids_feed_dict.update({tf_retain_id_placeholders[op]['out']: op_retain_ids_out})

            # In pruning
            if prev_op is not None:
                assert pruned_cnn_hyps[op]['weights'][2] == pruned_cnn_hyps[prev_op]['weights'][3]
                amount_to_retain_in_ch = pruned_cnn_hyps[op]['weights'][2]
                prev_op_weight_means = weight_means[prev_op]
                op_retain_ids_in = np.argsort(prev_op_weight_means).ravel()[-amount_to_retain_in_ch:]
                retain_ids_feed_dict.update({tf_retain_id_placeholders[op]['in']: op_retain_ids_in})
                logger.info('\tAmount of filers to retain for %s: %d (in)', op, amount_to_retain_in_ch)
            else:
                op_retain_ids_in = np.arange(pruned_cnn_hyps[cnn_ops[0]]['weights'][2])
                retain_ids_feed_dict.update({tf_retain_id_placeholders[op]['in']: op_retain_ids_in})

            prev_op = op

        elif 'fulcon' in op:
            # Out pruning
            amount_to_retain_out_ch = pruned_cnn_hyps[op]['out']
            logger.info('\tAmount of filers to retain for %s: %d (out)', op, amount_to_retain_out_ch)
            op_weight_means = weight_means[op]
            op_retain_ids_out = np.argsort(op_weight_means).ravel()[-amount_to_retain_out_ch:]
            retain_ids_feed_dict.update({tf_retain_id_placeholders[op]['out']:op_retain_ids_out})

            # In Pruning
            if op==first_fc:
                amount_to_retain_in_ch_slices = pruned_cnn_hyps[prev_op]['weights'][3]
                logger.info('\tAmount of filers to retain for %s: %d (in slices)', op, amount_to_retain_in_ch_slices)
                prev_op_weight_means = weight_means[prev_op]
                op_retain_slice_ids = np.argsort(prev_op_weight_means).ravel()[-amount_to_retain_in_ch_slices:]

                op_retain_ids_in = []
                for slice_id in op_retain_slice_ids:
                    op_retain_ids_in.extend(list(range(slice_id*(final_2d_width*final_2d_width),(slice_id+1)*(final_2d_width*final_2d_width))))

                logger.info('\tAmount of filers to retain for %s: %d (in ids)', op, len(op_retain_ids_in))
                retain_ids_feed_dict.update({tf_retain_id_placeholders[op]['in']: np.asarray(op_retain_ids_in)})

            else:
                amount_to_retain_in_ch = pruned_cnn_hyps[op]['in']
                logger.info('\tAmount of filers to retain for %s: %d (in ids)', op, amount_to_retain_in_ch)
                prev_op_weight_means = weight_means[prev_op]
                op_retain_ids_in = np.argsort(prev_op_weight_means).ravel()[-amount_to_retain_in_ch:]
                retain_ids_feed_dict.update({tf_retain_id_placeholders[op]['in']: op_retain_ids_in})

            prev_op = op

    return retain_ids_feed_dict


def get_pruned_cnn_hyp_feed_dict(prune_hyps):
    '''
    Prepare the feed dict to be fed when running the pruning op
    :param prune_hyps:
    :return:
    '''
    global tf_prune_cnn_hyperparameters,cnn_ops
    feed_dict = {}
    for op in cnn_ops:
        if 'conv' in op:
            feed_dict.update(
                {tf_prune_cnn_hyperparameters[op]['weights']:
                     prune_hyps[op]['weights']}
            )
        if 'fulcon' in op:
            feed_dict.update(
                {tf_prune_cnn_hyperparameters[op]['in']:
                 prune_hyps[op]['in']}
            )
            feed_dict.update(
                {tf_prune_cnn_hyperparameters[op]['out']:
                     prune_hyps[op]['out']}
            )

    return feed_dict


def calculate_pool_accuracy(hard_pool):
    '''
    Calculates the mini-batch wise accuracy for all the data points in the pool
    :param hard_pool:
    :return:
    '''
    global batch_size, tf_pool_data_batch, tf_pool_label_batch
    pool_accuracy = []
    pool_dataset, pool_labels = hard_pool.get_pool_data(False)
    for pool_id in range(hard_pool.get_size() // batch_size):
        pbatch_data = pool_dataset[pool_id * batch_size:(pool_id + 1) * batch_size, :, :, :]
        pbatch_labels = pool_labels[pool_id * batch_size:(pool_id + 1) * batch_size, :]
        pool_feed_dict = {tf_pool_data_batch[0]: pbatch_data,
                          tf_pool_label_batch[0]: pbatch_labels}
        p_predictions = session.run(pool_pred, feed_dict=pool_feed_dict)
        if num_labels <= 25:
            pool_accuracy.append(accuracy(p_predictions, pbatch_labels))
        else:
            pool_accuracy.append(top_n_accuracy(p_predictions, pbatch_labels, 5))

    return np.mean(pool_accuracy) if len(pool_accuracy) > 0 else 0



if __name__ == '__main__':

    # Various run-time arguments specified
    #
    allow_growth = False
    fake_tasks = False
    noise_label_rate = None
    noise_image_rate = None
    adapt_randomly = False
    use_fse_capacity = False
    inter_op_threads, intra_op_threads = 0, 0

    try:
        opts, args = getopt.getopt(
            sys.argv[1:], "",
            ["output_dir=", "num_gpus=", "memory=", "intra_op_threads=", "inter_op_threads=", 'allow_growth=',
             'dataset_type=', 'dataset_behavior=',
             'adapt_structure=', 'rigid_pooling=', 'rigid_pool_type=',
             'all_labels_included=', 'noise_labels=', 'noise_images=', 'adapt_randomly=', 'use_fse_capacity='])
    except getopt.GetoptError as err:
        print(err.with_traceback())
        print('<filename>.py --output_dir= --num_gpus= --memory=')

    if len(opts) != 0:
        for opt, arg in opts:
            if opt == '--output_dir':
                output_dir = arg
            if opt == '--num_gpus':
                num_gpus = int(arg)
            if opt == '--memory':
                mem_frac = float(arg)
            if opt == '--allow_growth':
                allow_growth = bool(int(arg))
            if opt == '--dataset_type':
                datatype = str(arg)
            if opt == '--dataset_behavior':
                behavior = str(arg)
            if opt == '--adapt_structure':
                adapt_structure = bool(int(arg))
            if opt == '--rigid_pooling':
                rigid_pooling = bool(int(arg))
            if opt == '--rigid_pool_type':
                rigid_pool_type = str(arg)
            if opt == '--all_labels_included':
                fake_tasks = bool(int(arg))
            if opt == '--noise_labels':
                noise_label_rate = float(arg)
            if opt == '--noise_images':
                noise_image_rate = float(arg)
            if opt == '--adapt_randomly':
                adapt_randomly = bool(int(arg))
            if opt == '--use_fse_capacity':
                use_fse_capacity = bool(int(arg))
            if opt == '--intra_op_threads':
                intra_op_threads = int(arg)
            if opt == '--inter_op_threads':
                inter_op_threads = int(arg)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not (adapt_structure or rigid_pooling):
        rigid_naive = True

    # Setting up loggers
    logger, perf_logger, cnn_structure_logger, \
    q_logger, class_dist_logger, pool_dist_logger, pool_dist_ft_logger, \
    hyp_logger, error_logger, prune_logger = setup_loggers(adapt_structure)

    logger.info('Created loggers')

    logger.info('Created Output directory: %s', output_dir)
    logger.info('Received all the required user arguments at Runtime')
    logger.debug('Output DIR: %s', output_dir)
    logger.info('Number of GPUs: %d', num_gpus)
    logger.info('Memory fraction per GPU: %.3f', mem_frac)
    logger.info('Dataset Name: %s', datatype)
    logger.info('Data Behavior: %s', behavior)
    logger.info('Use AdaCNN: %d', adapt_structure)
    logger.info('Use rigid pooling: %d', rigid_pooling)
    logger.info('Use rigid naive: %d',rigid_naive)
    logger.info('Adapt Randomly: %d',adapt_randomly)

    # =====================================================================
    # VARIOS SETTING UPS
    # SET FROM MAIN FUNCTIONS OF OTHER CLASSES
    cnn_model_saver.set_from_main(output_dir)

    set_varialbes_with_input_arguments(datatype, behavior, adapt_structure,rigid_pooling, use_fse_capacity)
    cnn_intializer_continuous.set_from_main(research_parameters, logging_level, logging_format)

    logger.info('Creating CNN hyperparameters and operations in the correct format')
    # Getting hyperparameters
    cnn_ops, cnn_hyperparameters, final_2d_width = utils.get_ops_hyps_from_string(dataset_info, cnn_string)
    init_cnn_ops, init_cnn_hyperparameters, final_2d_width = utils.get_ops_hyps_from_string(dataset_info, cnn_string)

    logger.info('Created CNN hyperparameters and operations in the correct format successfully\n')

    ada_cnn_adapter.set_from_main(research_parameters, final_2d_width,cnn_ops, cnn_hyperparameters, logging_level, logging_format)
    cnn_optimizer.set_from_main(research_parameters,model_hyperparameters,logging_level,logging_format,cnn_ops,final_2d_width)

    logger.info('Creating loggers\n')
    logger.info('=' * 80 + '\n')
    logger.info('Recognized following convolution operations')
    logger.info(cnn_ops)
    logger.info('With following Hyperparameters')
    logger.info(cnn_hyperparameters)
    logger.info(('=' * 80) + '\n')
    logging_hyperparameters(
        hyp_logger,cnn_hyperparameters,research_parameters,
        model_hyperparameters,interval_parameters,dataset_info
    )

    logging.info('Reading data from HDF5 File')
    # Datasets
    if datatype=='cifar-10':
        dataset_file= h5py.File("data"+os.sep+"cifar_10_dataset.hdf5", "r")
        train_dataset, train_labels = dataset_file['/train/images'], dataset_file['/train/labels']
        test_dataset, test_labels = dataset_file['/test/images'], dataset_file['/test/labels']

    elif datatype=='cifar-100':
        dataset_file = h5py.File("data" + os.sep + "cifar_100_dataset.hdf5", "r")
        train_dataset, train_labels = dataset_file['/train/images'], dataset_file['/train/labels']
        test_dataset, test_labels = dataset_file['/test/images'], dataset_file['/test/labels']

    elif datatype=='imagenet-250':
        dataset_file = h5py.File(".."+os.sep+"PreprocessingBenchmarkImageDatasets"+ os.sep + "imagenet_small_test" + os.sep + 'imagenet_250_dataset.hdf5', 'r')
        train_dataset, train_labels = dataset_file['/train/images'], dataset_file['/train/labels']
        test_dataset, test_labels = dataset_file['/valid/images'], dataset_file['/valid/labels']

    logging.info('Reading data from HDF5 File sucessful.\n')

    # Setting up graph parameters
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = False
    if mem_frac is not None:
        config.gpu_options.per_process_gpu_memory_fraction = mem_frac
    else:
        config.gpu_options.per_process_gpu_memory_fraction = 0.9

    session = tf.InteractiveSession(config=config)

    home_dir = '.' + os.sep + output_dir + os.sep + "ada_cnn_tensorboard_data"
    if not os.path.exists(home_dir):
        dirs_in_tensorboard = []
    else:
        print('file exist')
        dirs_in_tensorboard = [int(o) for o in os.listdir(home_dir + os.sep + "main")
                               if os.path.isdir(os.path.join(home_dir + os.sep + "main", o))]
    if len(dirs_in_tensorboard) == 0:
        summary_writer = tf.summary.FileWriter(
            output_dir + os.sep + "ada_cnn_tensorboard_data" + os.sep + "main" + os.sep + '0')
    else:
        summary_writer = tf.summary.FileWriter(
            output_dir + os.sep + "ada_cnn_tensorboard_data" +
            os.sep + "main" + os.sep + '%d' % (max(dirs_in_tensorboard) + 1))

    # Defining pool
    if adapt_structure or rigid_pooling:
        logger.info('Defining pools of data (validation and finetuning)')
        hardness = 0.5
        if datatype != 'imagenet-250':
            hard_pool_valid = Pool(size=pool_size//2, batch_size=batch_size, image_size=image_size,
                                   num_channels=num_channels, num_labels=num_labels, assert_test=False)
            hard_pool_ft = Pool(size=pool_size//2, batch_size=batch_size, image_size=image_size,
                                   num_channels=num_channels, num_labels=num_labels, assert_test=False)
        else:
            hard_pool_valid = Pool(size=pool_size // 2, batch_size=batch_size, image_size=resize_to,
                                   num_channels=num_channels, num_labels=num_labels, assert_test=False)
            hard_pool_ft = Pool(size=pool_size // 2, batch_size=batch_size, image_size=resize_to,
                                num_channels=num_channels, num_labels=num_labels, assert_test=False)
        logger.info('Defined pools of data successfully\n')

    first_fc = 'fulcon_out' if 'fulcon_0' not in cnn_ops else 'fulcon_0'
    # -1 is because we don't want to count fulcon_out
    layer_count = len(cnn_ops) - 1

    # ids of the convolution ops
    convolution_op_ids = []
    fulcon_op_ids = []
    for op_i, op in enumerate(cnn_ops):
        if 'conv' in op:
            convolution_op_ids.append(op_i)
        # Do not put the fulcon_out in the fulcon_ids we do not adapt the output layer
        # it already has the total number  of classes
        if 'fulcon' in op and op!='fulcon_out':
            fulcon_op_ids.append(op_i)

    logger.info('Found all convolution opeartion ids')

    # Defining initial hyperparameters as TF variables (so we can change them later during adaptation)
    # Define global set (used for learning rate decay)
    # Define CNN variables
    # Define CNN variable momentums
    # Define all the required operations
    with tf.variable_scope(TF_GLOBAL_SCOPE,reuse=False) as scope:
        global_step = tf.get_variable(initializer=0, dtype=tf.int32, trainable=False, name='global_step')
        logger.info('Defining TF Hyperparameters')
        tf_cnn_hyperparameters = cnn_intializer_continuous.init_tf_hyperparameters(cnn_ops, cnn_hyperparameters)
        logger.info('Defining Weights and Bias for CNN operations')
        _ = cnn_intializer_continuous.initialize_cnn_with_ops(cnn_ops, cnn_hyperparameters)
        logger.info('Following parameters defined')
        logger.info([v.name for v in tf.trainable_variables()])
        logger.info('='*80)
        logger.info('Defining Velocities for Weights and Bias for CNN operations')
        _ = cnn_intializer_continuous.define_velocity_vectors(scope, cnn_ops, cnn_hyperparameters)
        logger.info('Following parameters defined')
        logger.info([v.name for v in tf.global_variables()])
        logger.info('=' * 80)
    logger.info(('=' * 80) + '\n')


    if research_parameters['adapt_structure']:

        # Pruning hyperparameter initialization

        print(tf_prune_cnn_hyperparameters)

        # Adapting Policy Learner
        current_adaptive_dropout = get_adaptive_dropout()
        state_history_length = 2
        growth_adapter = ada_cnn_aac.AdaCNNAdaptingAdvantageActorCritic(
            discount_rate=0.75,
            filter_vector=filter_vector,
            conv_ids=convolution_op_ids, fulcon_ids=fulcon_op_ids, net_depth=layer_count,
            n_conv=len(convolution_op_ids), n_fulcon=len(fulcon_op_ids),
            epsilon=0.5,
            batch_size=32, persist_dir=output_dir,
            session=session,
            state_history_length=state_history_length,
            hidden_layers=[128, 64, 32], momentum=0.9, learning_rate=0.001,
            rand_state_length=32, adapt_max_amount=model_hyperparameters['add_amount'],
            adapt_fulcon_max_amount=model_hyperparameters['add_fulcon_amount'],
            num_classes=num_labels, filter_min_threshold=model_hyperparameters['filter_min_threshold'],
            fulcon_min_threshold=model_hyperparameters['fulcon_min_threshold'],
            trial_phase_threshold=1.0, binned_data_dist_length=model_hyperparameters['binned_data_dist_length'],
            top_k_accuracy=model_hyperparameters['top_k_accuracy']
        )

    # Running initialization opeartion
    logger.info('Running global variable initializer')
    init_op = tf.global_variables_initializer()
    _ = session.run(init_op)
    logger.info('Variable initialization successful\n')

    # Defining all Tensorflow ops required for
    # calculating logits, loss, predictions
    logger.info('Defining all the required Tensorflow operations')
    with tf.variable_scope(TF_GLOBAL_SCOPE, reuse=True) as scope:
        define_tf_ops(global_step,tf_cnn_hyperparameters,init_cnn_hyperparameters)
    logger.info('Defined all TF operations successfully\n')

    # Generating data distribution over time for n_iterations
    # Repeated every epoch
    data_gen = data_generator.DataGenerator(batch_size,num_labels,dataset_info['train_size'],dataset_info['n_slices'],
                                            image_size, dataset_info['n_channels'], dataset_info['resize_to'],
                                            dataset_info['dataset_name'], session)

    labels_per_task = num_labels // n_tasks
    if datatype=='cifar-10':
        labels_of_each_task = [list(range(i*labels_per_task,(i+1)*labels_per_task)) for i in range(n_tasks)]
    elif datatype=='cifar-100':
        n_tasks = 4
        labels_per_task = 50
        labels_of_each_task = [list(range(0,50)),list(range(25,75)),list(range(50,100)),list(range(75,100))+list(range(0,25))]
    elif datatype=='imagenet-250':
        # override settings
        n_tasks = 2
        labels_per_task = 250
        labels_of_each_task = [list(range(0, 250)),list(range(0, 250))]

    if behavior=='non-stationary':
        data_prior = label_sequence_generator.create_prior(n_iterations,behavior,labels_per_task,data_fluctuation)
    elif behavior=='stationary':
        data_prior = np.ones((n_iterations, labels_per_task)) * (1.0 / labels_per_task)
    else:
        raise NotImplementedError

    data_prior = change_data_prior_to_introduce_new_labels_over_time(data_prior,n_tasks,n_iterations,labels_of_each_task,num_labels)

    logger.debug('CNN_HYPERPARAMETERS')
    logger.debug('\t%s\n', tf_cnn_hyperparameters)

    logger.debug('TRAINABLE_VARIABLES')
    logger.debug('\t%s\n', [v.name for v in tf.trainable_variables()])

    logger.info('Variables initialized...')

    train_losses = []
    mean_train_loss = 0
    prev_test_accuracy = 0  # used to calculate test accuracy drop

    rolling_ativation_means = {}
    for op in cnn_ops:
        if 'conv' in op:
            logger.debug('\tDefining rolling activation mean for %s', op)
            rolling_ativation_means[op] = np.zeros([cnn_hyperparameters[op]['weights'][3]])
    act_decay = 0.9
    current_state, current_action,curr_adaptation_status, current_unscaled_action = None, None,None,None
    prev_unseen_valid_accuracy = 0
    pool_acc_before_adapt_queue, pool_acc_after_adapt_queue = [],[]
    valid_acc_queue = []

    current_q_learn_op_id = 0
    logger.info('Convolutional Op IDs: %s', convolution_op_ids)

    logger.info('Starting Training Phase')

    # Check if loss is stabilized (for starting adaptations)
    previous_loss = 1e5  # used for the check to start adapting
    current_data_lr = 1.0
    finetune_action = 1.0
    # Reward for Q-Learner
    prev_pool_accuracy = 0
    max_pool_accuracy = 0

    unseen_valid_accuracy = 0
    unseen_valid_after_accuracy = 0


    # Stop and start adaptations when necessary
    start_adapting = False
    stop_adapting = False
    adapt_period = 'both'
    # need to have a starting value because if the algorithm choose to add the data to validation set very first step
    train_accuracy = 0

    n_iter_per_task = n_iterations//n_tasks

    prev_binned_data_dist_vector = None
    running_binned_data_dist_vector = np.zeros((model_hyperparameters['binned_data_dist_length']),dtype=np.float32)
    tf_summary_data_dist = tf.placeholder(shape=[model_hyperparameters['binned_data_dist_length']],dtype=tf.float32,name='tf_summary_data_dist')
    tf_summary_accuracy = tf.placeholder(shape=None, dtype=tf.float32, name='tf_summary_accuracy' )

    with tf.name_scope(TF_GLOBAL_SCOPE):
        with tf.name_scope(constants.TF_SUMMARY_SCOPE):
            tf_summary_data_dist_writer = tf.summary.histogram('data_ditribution', tf_summary_data_dist)
            tf_summary_accuracy_writer = tf.summary.scalar('test_accuracy',tf_summary_accuracy)

    # Merge all summary writing ops to a single op so we don't ahve to run them individually
    summary_writer.add_graph(session.graph)

    binned_data_dist_decay = 0.5
    current_train_acc, prev_train_acc = 0, 0

    for epoch in range(n_epochs):

        if adapt_structure:
            adapter = growth_adapter

        for task in range(n_tasks):

            # We set the max pool accuracy to zero at the begining of each task.
            # Because in my opinion, pool accuracy treats each task differently
            # max_pool_accuracy = 0.0

            research_parameters['momentum']=0.9
            research_parameters['pool_momentum']=0.9

            cnn_optimizer.update_hyperparameters(research_parameters)

            # we stop 'num_gpus' items before the ideal number of training batches
            # because we always keep num_gpus items for valid data in memory
            for batch_id in range(0, n_iter_per_task - num_gpus, num_gpus):

                global_batch_id = (n_iterations * epoch) + (task*n_iter_per_task) + batch_id
                global_trial_phase = (global_batch_id * 1.0 / n_iterations)
                local_trial_phase = batch_id * 1.0/n_iter_per_task

                t0 = time.clock()  # starting time for a batch

                # We load 1 extra batch (chunk_size+1) because we always make the valid batch the batch_id+1

                # Feed dicitonary with placeholders for each tower
                batch_data, batch_labels, batch_weights = [], [], []
                train_feed_dict = {}
                train_feed_dict.update({tf_dropout_rate:current_adaptive_dropout})
                # ========================================================
                # Creating data batchs for the towers
                for gpu_id in range(num_gpus):
                    label_seq = label_sequence_generator.sample_label_sequence_for_batch(n_iterations, data_prior,
                                                                                         batch_size, num_labels)

                    # Get current batch of data nd labels (training)
                    b_d, b_l = data_gen.generate_data_with_label_sequence(train_dataset, train_labels, label_seq, dataset_info)
                    batch_data.append(b_d)
                    batch_labels.append(b_l)


                    if batch_id == 0:
                        logger.info('\tDataset shape: %s', b_d.shape)
                        logger.info('\tLabels shape: %s', b_l.shape)

                    # log class distribution
                    if (batch_id + gpu_id) % research_parameters['log_distribution_every'] == 0:
                        cnt = Counter(label_seq)
                        dist_str = ''
                        for li in range(num_labels):
                            dist_str += str(cnt[li] / len(label_seq)) + ',' if li in cnt else str(0) + ','
                        class_dist_logger.info('%d,%s', batch_id, dist_str)

                    # calculate binned data distribution and running average of that for RL state
                    cnt = Counter(np.argmax(batch_labels[-1], axis=1))
                    label_count_sorted = np.asarray([cnt[ci] if ci in cnt.keys() else 0.0 for ci in range(num_labels)])*1.0/batch_size
                    if model_hyperparameters['binned_data_dist_length'] != num_labels:
                        split_label_count_sorted = np.split(label_count_sorted,model_hyperparameters['binned_data_dist_length'])
                        label_count_sorted = np.asarray([np.asscalar(np.sum(lbl_cnt)) for lbl_cnt in split_label_count_sorted])

                    running_binned_data_dist_vector = binned_data_dist_decay * np.asarray(label_count_sorted) + (1.0-binned_data_dist_decay) * running_binned_data_dist_vector
                    # Do this only in the first epoch as the data is repeated there onwards
                    if epoch==0:
                        if batch_id%10==0:
                            data_dist_summ = session.run(tf_summary_data_dist_writer, feed_dict={tf_summary_data_dist:np.asarray(label_count_sorted)})
                            summary_writer.add_summary(data_dist_summ, global_step=global_batch_id)

                    if behavior == 'non-stationary':
                        batch_w = np.zeros((batch_size,))
                        batch_labels_int = np.argmax(batch_labels[-1], axis=1)

                        for li in range(num_labels):
                            batch_w[np.where(batch_labels_int == li)[0]] = max(1.0 - (cnt[li] * 1.0 / batch_size),
                                                                               1.0 / num_labels)
                        batch_weights.append(batch_w)

                    elif behavior == 'stationary':
                        batch_weights.append(np.ones((batch_size,)))
                    else:
                        raise NotImplementedError

                    train_feed_dict.update({
                        tf_train_data_batch[gpu_id]: batch_data[-1], tf_train_label_batch[gpu_id]: batch_labels[-1],
                        tf_data_weights[gpu_id]: batch_weights[-1]
                    })
                # =========================================================

                t0_train = time.clock()

                # =========================================================
                # Training Phase (Calculate loss and predictions)

                l, super_loss_vec, train_predictions = session.run(
                    [mean_loss_op, concat_loss_vec_op,
                     tower_predictions], feed_dict=train_feed_dict
                )

                train_accuracy = np.mean(
                    [accuracy(train_predictions[gid], batch_labels[gid]) for gid in range(num_gpus)]) / 100.0
                current_train_acc = train_accuracy
                # =========================================================

                # ==========================================================
                # Updating Pools of data

                # This if condition get triggered stochastically for AdaCNN
                # Never for Rigid-Pooling or Naive-Training
                if adapt_structure and np.random.random()<0.05*(valid_acc_decay**(epoch)):
                        #print('Only collecting valid data')
                        # Concatenate current 'num_gpus' batches to a single matrix
                        single_iteration_batch_data, single_iteration_batch_labels = None, None
                        for gpu_id in range(num_gpus):

                            if single_iteration_batch_data is None and single_iteration_batch_labels is None:
                                single_iteration_batch_data, single_iteration_batch_labels = batch_data[gpu_id], \
                                                                                             batch_labels[gpu_id]
                            else:
                                single_iteration_batch_data = np.append(single_iteration_batch_data, batch_data[gpu_id],
                                                                        axis=0)
                                single_iteration_batch_labels = np.append(single_iteration_batch_labels,
                                                                          batch_labels[gpu_id], axis=0)

                        hard_pool_valid.add_hard_examples(single_iteration_batch_data, single_iteration_batch_labels,
                                                          super_loss_vec,
                                                          max(0.05, prev_train_acc - current_train_acc))

                        logger.debug('Pooling data summary')
                        logger.debug('\tData batch size %d', single_iteration_batch_data.shape[0])
                        logger.debug('\tAccuracy %.3f', train_accuracy)
                        logger.debug('\tPool size (Valid): %d', hard_pool_valid.get_size())

                else:

                    # Concatenate current 'num_gpus' batches to a single matrix
                    single_iteration_batch_data, single_iteration_batch_labels = None, None
                    for gpu_id in range(num_gpus):

                        if single_iteration_batch_data is None and single_iteration_batch_labels is None:
                            single_iteration_batch_data, single_iteration_batch_labels = batch_data[gpu_id], \
                                                                                         batch_labels[gpu_id]
                        else:
                            single_iteration_batch_data = np.append(single_iteration_batch_data, batch_data[gpu_id],
                                                                    axis=0)
                            single_iteration_batch_labels = np.append(single_iteration_batch_labels,
                                                                      batch_labels[gpu_id], axis=0)

                    # Higer rates of accumulating data causes the pool to lose uniformity
                    if np.random.random()<0.05:
                        if adapt_structure or (rigid_pooling and rigid_pool_type == 'smart'):
                            #print('add examples to hard pool ft')
                            hard_pool_ft.add_hard_examples(single_iteration_batch_data, single_iteration_batch_labels,
                                                           super_loss_vec, max(0.05,prev_train_acc-current_train_acc))

                            logger.debug('\tPool size (FT): %d', hard_pool_ft.get_size())
                        elif (not adapt_structure) and rigid_pooling and rigid_pool_type=='naive':
                            #print('add examples to rigid pooling naive')
                            hard_pool_ft.add_hard_examples(single_iteration_batch_data, single_iteration_batch_labels,
                                                              super_loss_vec,1.0)

                            logger.debug('\tPool size (FT): %d', hard_pool_ft.get_size())

                    if adapt_structure:
                        train_feed_dict.update({tf_learning_rate:current_data_lr*model_hyperparameters['start_lr']})
                    # =========================================================
                    # # Training Phase (Optimization)
                    for _ in range(iterations_per_batch):
                        #print('training on current batch (action type: ', current_action_type, ')')
                        with tf.control_dependencies(update_train_velocity_op):
                            _ = session.run(
                                apply_grads_op, feed_dict=train_feed_dict
                            )


                t1_train = time.clock()

                if np.isnan(l):
                    logger.critical('Diverged (NaN detected) (batchID) %d (last Cost) %.3f', batch_id,
                                    train_losses[-1])
                assert not np.isnan(l)

                train_losses.append(l)
                prev_train_acc = current_train_acc


                # =============================================================
                # log hard_pool distribution over time
                if (adapt_structure or rigid_pooling) and \
                        (batch_id > 0 and batch_id % interval_parameters['history_dump_interval'] == 0):

                    # with open(output_dir + os.sep + 'Q_' + str(epoch) + "_" + str(batch_id)+'.pickle', 'wb') as f:
                    #    pickle.dump(adapter.get_Q(), f, pickle.HIGHEST_PROTOCOL)

                    pool_dist_string = ''
                    for val in hard_pool_valid.get_class_distribution():
                        pool_dist_string += str(val) + ','

                    pool_dist_logger.info('%s%d', pool_dist_string, hard_pool_valid.get_size())

                    pool_dist_string = ''
                    for val in hard_pool_ft.get_class_distribution():
                        pool_dist_string += str(val) + ','

                    pool_dist_ft_logger.info('%s%d', pool_dist_string, hard_pool_ft.get_size())
                # ==============================================================


                # ================================================================
                # Things done if one of below scenarios
                # For AdaCNN if adaptations stopped
                # For rigid pool CNNs from the beginning
                if ((research_parameters['pooling_for_nonadapt'] and
                         not research_parameters['adapt_structure']) or stop_adapting) and \
                        (batch_id > 0 and batch_id % interval_parameters['finetune_interval'] == 0):

                    logger.info('Pooling for non-adaptive CNN')
                    logger.info('Using dropout rate of: %.3f', dropout_rate)

                    # ===============================================================
                    # Finetune with data in hard_pool_ft (AdaCNN)
                    run_actual_finetune_operation(hard_pool_ft, finetune_action)

                    # =================================================================
                    # Calculate pool accuracy (hard_pool_valid)
                    if adapt_structure:
                        mean_pool_accuracy = get_pool_valid_accuracy(hard_pool_valid)
                    elif rigid_pooling:
                        mean_pool_accuracy = get_pool_valid_accuracy(hard_pool_ft)

                    logger.info('\tPool accuracy (hard_pool_valid): %.5f', mean_pool_accuracy)
                    # ==================================================================

                # ==============================================================
                # Testing Phase
                if batch_id % interval_parameters['test_interval'] == 0:
                    mean_train_loss = np.mean(train_losses)
                    logger.info('=' * 60)
                    logger.info('\tBatch ID: %d' % batch_id)
                    if decay_learning_rate:
                        if adapt_structure:
                            logger.info('\tLearning rate: %.5f' %(current_data_lr*model_hyperparameters['start_lr']))
                        else:
                            logger.info('\tLearning rate: %.5f' % session.run(tf_learning_rate))
                    else:
                        logger.info('\tLearning rate: %.5f' % start_lr)

                    logger.info('\tMinibatch Mean Loss: %.3f' % mean_train_loss)
                    logger.info('\tTrain Accuracy: %.3f'%(train_accuracy*100))
                    logger.info('\tValidation Accuracy (Unseen): %.3f' % unseen_valid_accuracy)
                    logger.info('\tValidation accumulation rate %.3f', 0.5*(valid_acc_decay**epoch))
                    logger.info('\tCurrent adaptive Dropout: %.3f', current_adaptive_dropout)
                    logger.info('\tTrial phase: %.3f (Local) %.3f (Global)', local_trial_phase, global_trial_phase)

                    test_accuracies = []
                    test_accuracies_top_5 = []
                    for test_batch_id in range(test_size // batch_size):
                        batch_test_data = test_dataset[test_batch_id * batch_size:(test_batch_id + 1) * batch_size, :, :, :]
                        batch_test_labels = test_labels[test_batch_id * batch_size:(test_batch_id + 1) * batch_size, :]

                        if datatype=='imagenet-250':
                            batch_test_data = batch_test_data[:,
                                              (image_size-resize_to)//2:((image_size-resize_to)//2)+resize_to,
                                              (image_size - resize_to) // 2:((image_size - resize_to) // 2) + resize_to,
                                              :]

                        batch_ohe_test_labels = np.zeros((batch_size,num_labels),dtype=np.float32)
                        batch_ohe_test_labels[np.arange(batch_size),batch_test_labels[:,0]] = 1.0
                        feed_test_dict = {tf_test_dataset: batch_test_data, tf_test_labels: batch_ohe_test_labels}
                        test_predictions = session.run(test_predicitons_op, feed_dict=feed_test_dict)
                        test_accuracies.append(accuracy(test_predictions, batch_ohe_test_labels))
                        if datatype=='imagenet-250':
                            test_accuracies_top_5.append(top_n_accuracy(test_predictions,batch_ohe_test_labels,5))

                        if test_batch_id < 10:

                            logger.debug('=' * 80)
                            logger.debug('Actual Test Labels %d', test_batch_id)
                            logger.debug(batch_test_labels.flatten()[:25])
                            logger.debug('Predicted Test Labels %d', test_batch_id)
                            logger.debug(np.argmax(test_predictions, axis=1).flatten()[:25])
                            logger.debug('Test: %d, %.3f', test_batch_id, accuracy(test_predictions, batch_test_labels))
                            logger.debug('=' * 80)

                    current_test_accuracy = np.mean(test_accuracies)
                    current_test_accuracy_top_5 = np.mean(test_accuracies_top_5)
                    if datatype!='imagenet-250':
                        logger.info('\tTest Accuracy: %.3f' % current_test_accuracy)
                    else:
                        logger.info('\tTest Accuracy: %.3f (Top-1) %.3f (Top-5)' %(current_test_accuracy,current_test_accuracy_top_5))

                    logger.info('=' * 60)
                    logger.info('')

                    # Logging error
                    prev_test_accuracy = current_test_accuracy
                    error_logger.info('%d,%.3f,%.3f,%.3f,%.3f',
                                      global_batch_id, mean_train_loss, train_accuracy*100.0,
                                      unseen_valid_accuracy, np.mean(test_accuracies)
                                      )

                    accuracy_summ = session.run(tf_summary_accuracy_writer,
                                                 feed_dict={tf_summary_accuracy: current_test_accuracy})
                    summary_writer.add_summary(accuracy_summ, global_step=global_batch_id)

                # ====================================================================

                    if research_parameters['adapt_structure'] and \
                            not start_adapting and \
                            (previous_loss - mean_train_loss < research_parameters['loss_diff_threshold'] or batch_id >
                                research_parameters['start_adapting_after']):
                        start_adapting = True
                        logger.info('=' * 80)
                        logger.info('Loss Stabilized: Starting structural adaptations...')
                        logger.info('Hardpool acceptance rate: %.2f', research_parameters['hard_pool_acceptance_rate'])
                        logger.info('=' * 80)

                    previous_loss = mean_train_loss

                    # reset variables
                    mean_train_loss = 0.0
                    train_losses = []

                # =======================================================================
                # Adaptations Phase of AdaCNN
                if research_parameters['adapt_structure']:

                    # ==============================================================
                    # Before starting the adaptations
                    if not start_adapting and batch_id > 0 and batch_id % (interval_parameters['finetune_interval']) == 0:

                        logger.info('Finetuning before starting adaptations. (To gain a reasonable accuracy to start with)')

                        pool_dataset, pool_labels = hard_pool_ft.get_pool_data(True)

                        # without if can give problems in exploratory stage because of no data in the pool
                        if hard_pool_ft.get_size() > batch_size:
                            # Train with latter half of the data
                            run_actual_finetune_operation(hard_pool_ft,finetune_action)
                    # ==================================================================

                    # ==================================================================
                    # Actual Adaptations
                    if (start_adapting and not stop_adapting) and batch_id > 0 and \
                                            batch_id % interval_parameters['policy_interval'] == 4:

                        # ==================================================================
                        # Calculating pool accuracy (Before Adaptation)
                        p_accuracy = calculate_pool_accuracy(hard_pool_ft)
                        pool_acc_before_adapt_queue.append(p_accuracy)
                        if len(pool_acc_before_adapt_queue) > 20:
                            del pool_acc_before_adapt_queue[0]
                        # ==================================================================

                        # ==================================================================
                        # Policy Update (Update policy only when we take actions actually using the qlearner)
                        # (Not just outputting finetune action)
                        # ==================================================================
                        if (not adapt_randomly) and current_state:

                            layer_specific_actions, finetune_action = current_action[:-1], current_action[-1]
                            assert len(layer_specific_actions)==len(convolution_op_ids)+len(fulcon_op_ids),'Number of layer specific ations did not match actual conv and fulcon layer count'
                            # don't use current state as the next state, current state is for a different layer

                            next_state = [0 for _ in range(len(filter_vector))]
                            affected_layer_index = 0
                            for li, la in enumerate(layer_specific_actions):

                                layer_id_for_action = None
                                if li < len(convolution_op_ids):
                                    layer_id_for_action = convolution_op_ids[li]
                                elif li < len(convolution_op_ids + fulcon_op_ids):
                                    layer_id_for_action = fulcon_op_ids[li - len(convolution_op_ids)]

                                next_state[layer_id_for_action]=current_state[layer_id_for_action] + la


                            next_state = list(next_state) + list(prev_binned_data_dist_vector)
                            assert len(current_state)==len(next_state),'Lengths of the current and next state donot match'
                            logger.info(('=' * 40) + ' Update Summary ' + ('=' * 40))
                            logger.info('\tState (prev): %s', str(current_state))
                            logger.info('\tAction (prev): %s', str(current_action))
                            logger.info('\tState (next): %s', str(next_state))
                            logger.info('\tPool Accuracy: %.3f', p_accuracy)
                            logger.info('\tValid accuracy (Before Adapt): %.3f', unseen_valid_accuracy)
                            logger.info('\tValid accuracy (After Adapt): %.3f', unseen_valid_after_accuracy)
                            logger.info('\tPrev pool Accuracy: %.3f\n', prev_pool_accuracy)
                            logger.info(('=' * 80) + '\n')
                            assert not np.isnan(p_accuracy)

                            # ================================================================================
                            # Actual updating of the policy
                            adapter.train_actor_critic({'prev_state': current_state, 'prev_action': current_unscaled_action,
                                                   'curr_state': next_state,
                                                   'next_accuracy': None,
                                                   'pool_accuracy_before_adapt_queue': pool_acc_before_adapt_queue,
                                                   'pool_accuracy_after_adapt_queue': pool_acc_after_adapt_queue,
                                                   'unseen_valid_before': unseen_valid_accuracy,
                                                   'unseen_valid_after': unseen_valid_after_accuracy,
                                                   'batch_id': global_batch_id})
                            # ===================================================================================

                            cnn_structure_logger.info(
                                '%d:%s:%s:%.5f:%s', global_batch_id, current_state,
                                current_action, p_accuracy,
                                utils.get_cnn_string_from_ops(cnn_ops, cnn_hyperparameters)
                            )

                            # The growth adapter has the random states for Q_eval^rand
                            q_logger.info('%d,%.5f', global_batch_id, growth_adapter.get_average_Q())

                            logger.debug('Resetting both data distribution means')

                            max_pool_accuracy = max(max(pool_acc_before_adapt_queue), p_accuracy)
                            prev_pool_accuracy = p_accuracy

                        # =============================================================
                        # Validation Phase (Use single tower) (Before adaptations)
                        v_label_seq = label_sequence_generator.sample_label_sequence_for_batch(
                            n_iterations, data_prior, batch_size, num_labels, freeze_index_increment=True
                        )
                        batch_valid_data, batch_valid_labels = data_gen.generate_data_with_label_sequence(
                            train_dataset, train_labels, v_label_seq, dataset_info
                        )

                        feed_valid_dict = {tf_valid_data_batch: batch_valid_data,
                                           tf_valid_label_batch: batch_valid_labels}
                        unseen_valid_predictions = session.run(valid_predictions_op, feed_dict=feed_valid_dict)
                        if num_labels>=25:
                            unseen_valid_accuracy = top_n_accuracy(unseen_valid_predictions, batch_valid_labels,5)
                        else:
                            unseen_valid_accuracy = accuracy(unseen_valid_predictions, batch_valid_labels)


                        # =============================================================

                        # ===================================================================================
                        # Execute action according to the policy
                        # ==================================================================================
                        filter_dict, filter_list = {}, []
                        for op_i, op in enumerate(cnn_ops):
                            if 'conv' in op:
                                filter_dict[op_i] = cnn_hyperparameters[op]['weights'][3]
                                filter_list.append(cnn_hyperparameters[op]['weights'][3])
                            elif 'pool' in op:
                                filter_dict[op_i] = 0
                                filter_list.append(0)
                            elif 'fulcon' in op and op!='fulcon_out':
                                filter_dict[op_i] = cnn_hyperparameters[op]['out']
                                filter_list.append(cnn_hyperparameters[op]['out'])

                        # For epoch 0 and 1
                        # Epoch 0: Randomly grow the network
                        # Epoch 1: Deterministically grow the network
                        if not adapt_randomly:
                            data = {'filter_counts': filter_dict, 'filter_counts_list': filter_list,
                                    'binned_data_dist': running_binned_data_dist_vector.tolist()}
                            if np.random.random()<start_eps:
                                current_state, current_action, current_unscaled_action = adapter.sample_action_stochastic_from_actor(data)
                            else:
                                current_state, current_action, current_unscaled_action = adapter.sample_action_deterministic_from_actor(data)
                        else:
                            raise NotImplementedError

                        layer_specific_actions, finetune_action = current_action[:-1], current_action[-1]

                        logger.info('Finetune rate: %.5f', finetune_action)
                        logger.info('Data train rate: %.5f', current_data_lr)

                        # reset the binned data distribution
                        prev_binned_data_dist_vector = running_binned_data_dist_vector
                        running_binned_data_dist_vector = np.zeros(
                            (model_hyperparameters['binned_data_dist_length']), dtype=np.float32)

                        for li, la in enumerate(layer_specific_actions):

                            layer_id_for_action = None
                            if li < len(convolution_op_ids):
                                layer_id_for_action = convolution_op_ids[li]
                            elif li < len(convolution_op_ids+fulcon_op_ids):
                                layer_id_for_action = fulcon_op_ids[li-len(convolution_op_ids)]
                            # where all magic happens (adding and removing filters)
                            si, ai = current_state, la
                            current_op = cnn_ops[layer_id_for_action]

                            for tmp_op in reversed(cnn_ops):
                                if 'conv' in tmp_op:
                                    last_conv_id = tmp_op
                                    break

                            if ai>0.0:
                                random_add = np.random.random()
                                if 'conv' in current_op:
                                    run_actual_add_operation(session,current_op,layer_id_for_action,last_conv_id,hard_pool_ft,ai, epoch, random_add)
                                elif 'fulcon' in current_op:
                                    run_actual_add_operation_for_fulcon(session,current_op,layer_id_for_action, last_conv_id, hard_pool_ft,ai, epoch, random_add)

                            elif ai <0.0:
                                if 'conv' in current_op:
                                    run_actual_remove_operation(session,current_op,layer_id_for_action,last_conv_id,hard_pool_ft,abs(ai))
                                elif 'fulcon' in current_op:
                                    run_actual_remove_operation_for_fulcon(session,current_op,layer_id_for_action, last_conv_id,hard_pool_ft,abs(ai))

                        if finetune_action>0.0:
                            # pooling takes place here
                            run_actual_finetune_operation(hard_pool_ft,finetune_action)

                        # ==================================================================
                        # Calculating pool accuracy (After Adaptation)
                        p_accuracy = calculate_pool_accuracy(hard_pool_ft)
                        pool_acc_after_adapt_queue.append(p_accuracy)
                        if len(pool_acc_after_adapt_queue) > 20:
                            del pool_acc_after_adapt_queue[0]
                        # ==================================================================

                        unseen_valid_after_predictions = session.run(valid_predictions_op, feed_dict=feed_valid_dict)
                        if num_labels>25:
                            unseen_valid_after_accuracy = top_n_accuracy(unseen_valid_after_predictions, batch_valid_labels,5)
                        else:
                            unseen_valid_after_accuracy = accuracy(unseen_valid_after_predictions, batch_valid_labels)

                        # =============================================================



                # ==============================================================
                # Logging time / tensorflow information
                t1 = time.clock()
                op_count = len(tf.get_default_graph().get_operations())
                var_count = len(tf.global_variables()) + len(tf.local_variables()) + len(tf.model_variables())
                perf_logger.info('%d,%.5f,%.5f,%d,%d', global_batch_id, t1 - t0,
                                 (t1_train - t0_train) / num_gpus, op_count, var_count)
                # ================================================================

            # ==============================================================
            # Epoch ending condition check (Given by RL)
            '''if epoch < model_hyperparameters['rl_epochs']-1 and adapter.check_pool_accuracy_saturation():

                adapter.reset_max_pool_accuracy()
                logger.info('='*80)
                logger.info('End of %d episolde', epoch)
                logger.info('Epsilon: %.5f',start_eps)
                session.run(tf_reset_cnn)

                _, init_cnn_hyperparameters, _ = utils.get_ops_hyps_from_string(dataset_info, cnn_string)
                cnn_hyperparameters = dict(init_cnn_hyperparameters)
                print(cnn_hyperparameters)

                print(init_cnn_hyperparameters)
                for op in cnn_ops:
                    if 'conv' in op:
                        session.run(tf_update_hyp_ops[op],
                                    feed_dict={tf_weight_shape: init_cnn_hyperparameters[op]['weights']})
                    elif 'fulcon' in op:
                        session.run(tf_update_hyp_ops[op], feed_dict={tf_in_size: init_cnn_hyperparameters[op]['in'],
                                                                      tf_out_size: init_cnn_hyperparameters[op]['out']})
                print(session.run(tf_cnn_hyperparameters))

                hard_pool_ft.reset_pool()
                hard_pool_valid.reset_pool()
                start_adapting = False
                session.run(tower_logits, feed_dict=train_feed_dict)
                previous_loss = 1e5
                break
                logger.info('=' * 80)'''
            # ==============================================================

        # Reset the model every rl epoch except the last one
        if epoch < model_hyperparameters['rl_epochs'] - 1:

            adapter.reset_max_pool_accuracy()
            logger.info('=' * 80)
            logger.info('End of %d episolde (full epoch)',epoch)
            logger.info('Epsilon: %.5f', start_eps)
            session.run(tf_reset_cnn)

            _, init_cnn_hyperparameters, _ = utils.get_ops_hyps_from_string(dataset_info, cnn_string)
            cnn_hyperparameters = dict(init_cnn_hyperparameters)
            print(cnn_hyperparameters)

            print(init_cnn_hyperparameters)
            for op in cnn_ops:
                if 'conv' in op:
                    session.run(tf_update_hyp_ops[op],
                                feed_dict={tf_weight_shape: init_cnn_hyperparameters[op]['weights']})
                elif 'fulcon' in op:
                    session.run(tf_update_hyp_ops[op], feed_dict={tf_in_size: init_cnn_hyperparameters[op]['in'],
                                                                  tf_out_size: init_cnn_hyperparameters[op]['out']})
            print(session.run(tf_cnn_hyperparameters))

            session.run(tower_logits, feed_dict=train_feed_dict)
            hard_pool_ft.reset_pool()
            hard_pool_valid.reset_pool()

            start_adapting = False
            logger.info('=' * 80)
            previous_loss = 1e5

        # We use whatever the model found by the last RL episode
        if epoch == model_hyperparameters['rl_epochs']-1:
            stop_adapting = True
            current_data_lr = 1.0

        # =======================================================
        # Decay learning rate (if set) Every 2 epochs
        if research_parameters['adapt_structure']:
            if decay_learning_rate and epoch>model_hyperparameters['rl_epochs'] and epoch%2==1:
                session.run(increment_global_step_op)
                current_data_lr *= model_hyperparameters['decay_rate']

        else:
            if decay_learning_rate and epoch>0 and epoch%2==1:
                session.run(increment_global_step_op)
        # ======================================================

        # AdaCNN Algorithm
        if research_parameters['adapt_structure']:

            start_eps = max([start_eps*eps_decay,0.1])
            # At the moment not stopping adaptations for any reason
            # stop_adapting = adapter.check_if_should_stop_adapting()

        cnn_model_saver.save_cnn_hyperparameters(cnn_ops,final_2d_width,cnn_hyperparameters,'cnn-hyperparameters-%d.pickle'%epoch)
        cnn_model_saver.save_cnn_weights(cnn_ops,session,'cnn-model-%d.ckpt'%epoch)