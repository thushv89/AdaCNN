__author__ = 'Thushan Ganegedara'

import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import numpy as np
import os
from math import ceil, floor
import logging
import sys
import ada_cnn_qlearner
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
import cnn_hyperparameters_getter
import cnn_optimizer
import cnn_intializer
import ada_cnn_adapter
import data_generator
import label_sequence_generator
import h5py

logging_level = logging.INFO
logging_format = '[%(funcName)s] %(message)s'

datatype,behavior = None,None
dataset_dir,output_dir = None,None
adapt_structure = False
rigid_pooling = False

interval_parameters, research_parameters, model_hyperparameters, dataset_info = None, None, None, None
image_size, num_channels = None,None
n_epochs, iterations_per_batch, num_labels, train_size, test_size, n_slices, data_fluctuation = None,None,None,None,None,None,None
cnn_string, filter_vector = None,None

batch_size = None
start_lr, decay_learning_rate, decay_rate, decay_steps = None,None,None,None
beta, include_l2_loss = None,None
use_dropout, in_dropout_rate, dropout_rate = None, None, None
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
def set_varialbes_with_input_arguments(dataset_name, dataset_behavior, adapt_structure, use_rigid_pooling):
    global interval_parameters, model_hyperparameters, research_parameters, dataset_info, cnn_string, filter_vector
    global image_size, num_channels
    global n_epochs, iterations_per_batch, num_labels, train_size, test_size, n_slices, data_fluctuation
    global start_lr, decay_learning_rate, decay_rate, decay_steps
    global batch_size, beta, include_l2_loss
    global use_dropout, in_dropout_rate, dropout_rate
    global pool_size
    global use_loc_res_norm, lrn_radius, lrn_alpha, lrn_beta
    global start_eps,eps_decay,valid_acc_decay
    global n_tasks

    # Data specific parameters
    dataset_info = cnn_hyperparameters_getter.get_data_specific_hyperparameters(dataset_name, dataset_behavior,
                                                                                dataset_dir)
    image_size = dataset_info['image_size']
    num_channels = dataset_info['n_channels']

    # interval parameters
    interval_parameters = cnn_hyperparameters_getter.get_interval_related_hyperparameters(dataset_name)

    # Research parameters
    research_parameters = cnn_hyperparameters_getter.get_research_hyperparameters(dataset_name, adapt_structure, use_rigid_pooling, logging_level)

    # Model Hyperparameters
    model_hyperparameters = cnn_hyperparameters_getter.get_model_specific_hyperparameters(datatype, dataset_behavior, adapt_structure, rigid_pooling, dataset_info['n_labels'])

    n_epochs = model_hyperparameters['epochs']

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

    # Local Response Normalization
    use_loc_res_norm = model_hyperparameters['use_loc_res_norm']
    lrn_radius = model_hyperparameters['lrn_radius']
    lrn_alpha = model_hyperparameters['lrn_alpha']
    lrn_beta = model_hyperparameters['lrn_beta']

    # pool parameters
    pool_size = model_hyperparameters['pool_size']

    if adapt_structure:
        start_eps = model_hyperparameters['start_eps']
        eps_decay = model_hyperparameters['eps_decay']
    valid_acc_decay = model_hyperparameters['validation_set_accumulation_decay']

    # Tasks
    n_tasks = model_hyperparameters['n_tasks']

n_iterations = 10000
cnn_ops, cnn_hyperparameters = None, None

state_action_history = []

cnn_ops, cnn_hyperparameters = None, None
num_gpus = -1

# Tensorflow Op / Variable related Python variables
# Optimizer Related
optimizer, custom_lr = None,None
tf_learning_rate = None
# Optimizer (Data) Related
tf_avg_grad_and_vars, apply_grads_op, concat_loss_vec_op, \
update_train_velocity_op, tf_mean_activation, mean_loss_op = None,None,None,None,None,None

# Optimizer (Pool) Related
tf_pool_avg_gradvars, apply_pool_grads_op, update_pool_velocity_ops, tf_mean_pool_activations, mean_pool_loss = None, None, None, None, None

# Data related
tf_train_data_batch, tf_train_label_batch, tf_data_weights = [], [], []
tf_test_dataset, tf_test_labels = None, None
tf_valid_data_batch, tf_valid_label_batch = None, None

# Data (Pool) related
tf_pool_data_batch, tf_pool_label_batch = [], []
pool_pred = None

# Logit related
tower_grads, tower_loss_vectors, tower_losses, tower_activation_update_ops, tower_predictions = [], [], [], [], []
tower_pool_grads, tower_pool_losses, tower_pool_activation_update_ops = [], [], []
tower_logits = []

# Test/Valid related
valid_loss_op,valid_predictions_op, test_predicitons_op = None,None,None

# Adaptation related
tf_slice_optimize = {}
tf_slice_vel_update = {}
tf_add_filters_ops, tf_rm_filters_ops, tf_replace_ind_ops = {}, {}, {}
tf_indices, tf_indices_size = None,None
tf_update_hyp_ops = {}
tf_action_info, tf_running_activations = None, None
tf_weights_this,tf_bias_this = None, None
tf_weights_next,tf_wvelocity_this, tf_bvelocity_this, tf_wvelocity_next = None, None, None, None
tf_weight_shape,tf_in_size = None, None
increment_global_step_op = None

adapt_period = None

# Loggers
logger = None
perf_logger = None


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
    if research_parameters['whiten_images']:
        mu, var = tf.nn.moments(x, axes=[1, 2, 3])
        tr_x = tf.transpose(x, [1, 2, 3, 0])
        tr_x = (tr_x - mu) / tf.maximum(tf.sqrt(var), 1.0 / (image_size * image_size * num_channels))
        x = tf.transpose(tr_x, [3, 0, 1, 2])

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
                x = utils.lrelu(x + b, name=scope.name + '/top')

                activation_ops.append(
                    tf.assign(tf.get_variable(TF_ACTIVAIONS_STR), tf.reduce_mean(x, [0, 1, 2]), validate_shape=False))

                if use_loc_res_norm and op == last_conv_id:
                    x = tf.nn.local_response_normalization(x, depth_radius=lrn_radius, alpha=lrn_alpha,
                                                           beta=lrn_beta)  # hyperparameters from tensorflow cifar10 tutorial

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
            if training and use_dropout:
                x = tf.nn.dropout(x, keep_prob=1.0 - dropout_rate, name='dropout')

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
                    x = tf.reshape(x, [batch_size, tf_cnn_hyperparameters[op][TF_FC_WEIGHT_IN_STR]])
                    x = utils.lrelu(tf.matmul(x, w) + b, name=scope.name + '/top')
                    if training and use_dropout:
                        x = tf.nn.dropout(x, keep_prob=1.0 - dropout_rate, name='dropout')

                elif 'fulcon_out' == op:
                    x = tf.matmul(x, w) + b

                else:
                    x = utils.lrelu(tf.matmul(x, w) + b, name=scope.name + '/top')
                    if training and use_dropout:
                        x = tf.nn.dropout(x, keep_prob=1.0 - dropout_rate, name='dropout')

    return x, activation_ops


def tower_loss(dataset, labels, weighted, tf_data_weights, tf_cnn_hyperparameters):
    global cnn_ops
    logits, _ = inference(dataset, tf_cnn_hyperparameters, True)
    # use weighted loss
    if weighted:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels) * tf_data_weights)
    else:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    if include_l2_loss:
        fulcons = []
        for op in cnn_ops:
            if 'fulcon' in op and op != 'fulcon_out':
                fulcons.append(op)
        fc_weights = []
        for op in fulcons:
            with tf.variable_scope(op):
                fc_weights.append(tf.get_variable(TF_WEIGHTS))

        loss = tf.reduce_sum([loss, beta * tf.reduce_sum([tf.nn.l2_loss(w) for w in fc_weights])])

    total_loss = loss

    return total_loss


def calc_loss_vector(scope, dataset, labels, tf_cnn_hyperparameters):
    logits, _ = inference(dataset, tf_cnn_hyperparameters, True)
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
    for a_i in zip(*tower_activations):
        stacked_activations = None
        for a in a_i:
            if stacked_activations is None:
                stacked_activations = tf.identity(a)
            else:
                stacked_activations = tf.stack([stacked_activations, a], axis=0)

        mean_activations.append(tf.reduce_mean(stacked_activations, [0]))
    return mean_activations


def predict_with_logits(logits):
    # Predictions for the training, validation, and test data.
    prediction = tf.nn.softmax(logits)
    return prediction


def predict_with_dataset(dataset, tf_cnn_hyperparameters):
    logits, _ = inference(dataset, tf_cnn_hyperparameters, False)
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
    error_logger.info('#Batch_ID,Loss(Train),Valid(Unseen),Test Accuracy')

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

    cnn_structure_logger, q_logger = None, None
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

    class_dist_logger = logging.getLogger('class_dist_logger')
    class_dist_logger.propagate = False
    class_dist_logger.setLevel(logging.INFO)
    class_dist_handler = logging.FileHandler(output_dir + os.sep + 'class_distribution.log', mode='w')
    class_dist_handler.setFormatter(logging.Formatter('%(message)s'))
    class_dist_logger.addHandler(class_dist_handler)

    pool_dist_logger = logging.getLogger('pool_distribution_logger')
    pool_dist_logger.propagate = False
    pool_dist_logger.setLevel(logging.INFO)
    pool_handler = logging.FileHandler(output_dir + os.sep + 'pool_distribution.log', mode='w')
    pool_handler.setFormatter(logging.Formatter('%(message)s'))
    pool_dist_logger.addHandler(pool_handler)
    pool_dist_logger.info('#Class distribution')

    return main_logger, perf_logger, \
           cnn_structure_logger, q_logger, class_dist_logger, \
           pool_dist_logger, hyp_logger, error_logger


def get_activation_dictionary(activation_list, cnn_ops, conv_op_ids):
    current_activations = {}
    for act_i, layer_act in enumerate(activation_list):
        current_activations[cnn_ops[conv_op_ids[act_i]]] = layer_act
    return current_activations


def define_tf_ops(global_step, tf_cnn_hyperparameters, init_cnn_hyperparameters):
    global optimizer
    global tf_train_data_batch, tf_train_label_batch, tf_data_weights
    global tf_test_dataset,tf_test_labels
    global tf_pool_data_batch, tf_pool_label_batch
    global tower_grads, tower_loss_vectors, tower_losses, tower_activation_update_ops, tower_predictions
    global tower_pool_grads, tower_pool_losses, tower_pool_activation_update_ops, tower_logits
    global tf_add_filters_ops, tf_rm_filters_ops, tf_replace_ind_ops, tf_slice_optimize, tf_slice_vel_update
    global tf_indices, tf_indices_size
    global tf_avg_grad_and_vars, apply_grads_op, concat_loss_vec_op, update_train_velocity_op, tf_mean_activation, mean_loss_op
    global tf_pool_avg_gradvars, apply_pool_grads_op, update_pool_velocity_ops, tf_mean_pool_activations, mean_pool_loss
    global valid_loss_op,valid_predictions_op, test_predicitons_op
    global tf_valid_data_batch,tf_valid_label_batch
    global pool_pred
    global tf_update_hyp_ops, tf_action_info, tf_running_activations
    global tf_weights_this,tf_bias_this, tf_weights_next,tf_wvelocity_this, tf_bvelocity_this, tf_wvelocity_next
    global tf_weight_shape,tf_in_size
    global increment_global_step_op,tf_learning_rate
    global logger

    # custom momentum optimizing we calculate momentum manually
    logger.info('Defining Optimizer')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
    increment_global_step_op = tf.assign(global_step, global_step + 1)
    tf_learning_rate = tf.train.exponential_decay(start_lr, global_step, decay_steps=decay_steps,
                               decay_rate=decay_rate, staircase=True)

    # Test data (Global)
    logger.info('Defining Test data placeholders')
    tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels),
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
                tf_train_data_batch.append(tf.placeholder(tf.float32,
                                                          shape=(
                                                              batch_size, image_size, image_size, num_channels),
                                                          name='TrainDataset'))
                tf_train_label_batch.append(
                    tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='TrainLabels'))
                tf_data_weights.append(tf.placeholder(tf.float32, shape=(batch_size), name='TrainWeights'))

                # Training data opearations
                logger.info('\tDefining logit operations')
                tower_logit_op, tower_tf_activation_ops = inference(tf_train_data_batch[-1],
                                                                    tf_cnn_hyperparameters, True)
                tower_logits.append(tower_logit_op)
                tower_activation_update_ops.append(tower_tf_activation_ops)

                logger.info('\tDefine Loss for each tower')
                tf_tower_loss = tower_loss(tf_train_data_batch[-1], tf_train_label_batch[-1], True,
                                           tf_data_weights[-1], tf_cnn_hyperparameters)

                tower_losses.append(tf_tower_loss)
                tf_tower_loss_vec = calc_loss_vector(scope, tf_train_data_batch[-1], tf_train_label_batch[-1],
                                                     tf_cnn_hyperparameters)
                tower_loss_vectors.append(tf_tower_loss_vec)

                logger.info('\tGradient calculation opeartions for tower')
                tower_grad = cnn_optimizer.gradients(optimizer, tf_tower_loss, global_step,
                                       tf.constant(start_lr, dtype=tf.float32))
                tower_grads.append(tower_grad)

                logger.info('\tPrediction operations for tower')
                tower_pred = predict_with_dataset(tf_train_data_batch[-1], tf_cnn_hyperparameters)
                tower_predictions.append(tower_pred)

                # Pooling data operations
                logger.info('\tPool related operations')
                tf_pool_data_batch.append(tf.placeholder(tf.float32,
                                                         shape=(
                                                             batch_size, image_size, image_size, num_channels),
                                                         name='PoolDataset'))
                tf_pool_label_batch.append(
                    tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='PoolLabels'))

                with tf.name_scope('pool') as scope:
                    single_pool_logit_op, single_activation_update_op = inference(tf_pool_data_batch[-1],
                                                                                  tf_cnn_hyperparameters, True)
                    tower_pool_activation_update_ops.append(single_activation_update_op)

                    single_pool_loss = tower_loss(tf_pool_data_batch[-1], tf_pool_label_batch[-1], False, None,
                                                  tf_cnn_hyperparameters)
                    tower_pool_losses.append(single_pool_loss)
                    single_pool_grad = cnn_optimizer.gradients(optimizer, single_pool_loss, global_step, start_lr)
                    tower_pool_grads.append(single_pool_grad)

    logger.info('GLOBAL_VARIABLES (all)')
    logger.info('\t%s\n', [v.name for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])

    with tf.device('/gpu:0'):
        logger.info('Tower averaging for Gradients for Training data')
        # Train data operations
        # avg_grad_and_vars = [(avggrad0,var0),(avggrad1,var1),...]
        tf_avg_grad_and_vars = average_gradients(tower_grads)
        apply_grads_op = cnn_optimizer.apply_gradient_with_momentum(optimizer, start_lr, global_step)
        concat_loss_vec_op = concat_loss_vector_towers(tower_loss_vectors)
        update_train_velocity_op = cnn_optimizer.update_train_momentum_velocity(tf_avg_grad_and_vars)
        tf_mean_activation = mean_tower_activations(tower_activation_update_ops)
        mean_loss_op = tf.reduce_mean(tower_losses)

        logger.info('Tower averaging for Gradients for Pool data')
        # Pool data operations
        tf_pool_avg_gradvars = average_gradients(tower_pool_grads)
        apply_pool_grads_op = cnn_optimizer.apply_gradient_with_pool_momentum(optimizer, start_lr, global_step)
        update_pool_velocity_ops = cnn_optimizer.update_pool_momentum_velocity(tf_pool_avg_gradvars)
        tf_mean_pool_activations = mean_tower_activations(tower_pool_activation_update_ops)
        mean_pool_loss = tf.reduce_mean(tower_pool_losses)

    with tf.device('/gpu:0'):

        increment_global_step_op = tf.assign(global_step, global_step + 1)

        # GLOBAL: Tensorflow operations for hard_pool
        with tf.name_scope('pool') as scope:
            tf.get_variable_scope().reuse_variables()
            pool_pred = predict_with_dataset(tf_pool_data_batch[0], tf_cnn_hyperparameters)

        # GLOBAL: Tensorflow operations for test data
        # Valid data (Next train batch) Unseen
        tf_valid_data_batch = tf.placeholder(tf.float32,
                                             shape=(batch_size, image_size, image_size, num_channels),
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

                tf_action_info = tf.placeholder(shape=[3], dtype=tf.int32,
                                                name='tf_action')  # [op_id,action_id,amount] (action_id 0 - add, 1 -remove)
                tf_running_activations = tf.placeholder(shape=(None,), dtype=tf.float32, name='running_activations')

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

                tf_weight_shape = tf.placeholder(shape=[4], dtype=tf.int32, name='weight_shape')
                tf_in_size = tf.placeholder(dtype=tf.int32, name='input_size')


                #tf_reset_cnn = cnn_intializer.reset_cnn(init_cnn_hyperparameters)

                for tmp_op in cnn_ops:
                    if 'conv' in tmp_op:
                        tf_update_hyp_ops[tmp_op] = ada_cnn_adapter.update_tf_hyperparameters(tmp_op, tf_weight_shape, tf_in_size)
                        tf_add_filters_ops[tmp_op] = ada_cnn_adapter.add_with_action(tmp_op, tf_action_info, tf_weights_this,
                                                                     tf_bias_this, tf_weights_next,
                                                                     tf_running_activations,
                                                                     tf_wvelocity_this, tf_bvelocity_this,
                                                                     tf_wvelocity_next)
                        tf_rm_filters_ops[tmp_op] = ada_cnn_adapter.remove_with_action(tmp_op, tf_action_info,
                                                                       tf_running_activations,
                                                                       tf_cnn_hyperparameters)
                        # tf_replace_ind_ops[tmp_op] = get_rm_indices_with_distance(tmp_op,tf_action_info,tf_cnn_hyperparameters)
                        tf_slice_optimize[tmp_op], tf_slice_vel_update[tmp_op] = cnn_optimizer.optimize_masked_momentum_gradient(
                            optimizer, tf_indices,
                            tmp_op, tf_avg_grad_and_vars, tf_cnn_hyperparameters,
                            tf.constant(start_lr, dtype=tf.float32), global_step
                        )

                    elif 'fulcon' in tmp_op:
                        tf_update_hyp_ops[tmp_op] = ada_cnn_adapter.update_tf_hyperparameters(tmp_op, tf_weight_shape, tf_in_size)


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


def augment_pool_data(hard_pool):
    global pool_dataset, pool_labels
    pool_dataset, pool_labels = hard_pool.get_pool_data(True)
    '''if research_parameters['pool_randomize'] and np.random.random() < \
            research_parameters['pool_randomize_rate']:
        if use_multiproc:
            try:
                pool = MPPool(processes=pool_workers)
                distorted_imgs = pool.map(distort_img, pool_dataset)
                pool_dataset = np.asarray(distorted_imgs)
                pool.close()
                pool.join()
            except Exception:
                raise AssertionError
        else:
            distorted_imgs = []
            for img in pool_dataset:
                distorted_imgs.append(distort_img(img))
            pool_dataset = np.vstack(distorted_imgs)'''

    return pool_dataset, pool_labels


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


def fintune_with_pool_ft(hard_pool_ft):
    global apply_pool_grads_op, update_pool_velocity_ops

    if hard_pool_ft.get_size() > batch_size:
        # Randomize data in the batch
        pool_dataset, pool_labels = augment_pool_data(hard_pool_ft)  # Train with latter half of the data

        for pool_id in range(0,
                             (hard_pool_ft.get_size() // batch_size) - 1, num_gpus):
            if np.random.random() < research_parameters['finetune_rate']:
                pool_feed_dict = {}
                for gpu_id in range(num_gpus):
                    pbatch_data = pool_dataset[
                                  (pool_id + gpu_id) * batch_size:(pool_id + gpu_id + 1) * batch_size, :, :,
                                  :]
                    pbatch_labels = pool_labels[
                                    (pool_id + gpu_id) * batch_size:(pool_id + gpu_id + 1) * batch_size, :]
                    pool_feed_dict.update({tf_pool_data_batch[gpu_id]: pbatch_data,
                                           tf_pool_label_batch[gpu_id]: pbatch_labels})

                _, _ = session.run([apply_pool_grads_op, update_pool_velocity_ops],
                                   feed_dict=pool_feed_dict)


def run_actual_add_operation(session, current_op, li, last_conv_id, hard_pool_ft):
    '''
    Run the add operation using the given Session
    :param session:
    :param current_op:
    :param li:
    :param last_conv_id:
    :param hard_pool_ft:
    :return:
    '''
    global current_adapted_op,current_adapted_indices
    amount_to_add = ai[1]

    if current_op != last_conv_id:
        next_conv_op = \
            [tmp_op for tmp_op in cnn_ops[cnn_ops.index(current_op) + 1:] if 'conv' in tmp_op][0]

    # Run the session and change the tensorflow variables (weights bias and velocities)
    _ = session.run(tf_add_filters_ops[current_op],
                    feed_dict={
                        tf_action_info: np.asarray([li, 1, ai[1]]),
                        tf_weights_this: np.random.normal(scale=0.01, size=(
                            cnn_hyperparameters[current_op]['weights'][0],
                            cnn_hyperparameters[current_op]['weights'][1],
                            cnn_hyperparameters[current_op]['weights'][2], amount_to_add)),
                        tf_bias_this: np.random.normal(scale=0.01, size=(amount_to_add)),

                        tf_weights_next: np.random.normal(scale=0.01, size=(
                            cnn_hyperparameters[next_conv_op]['weights'][0],
                            cnn_hyperparameters[next_conv_op]['weights'][1],
                            amount_to_add, cnn_hyperparameters[next_conv_op]['weights'][3])
                                                          ) if last_conv_id != current_op else
                        np.random.normal(scale=0.01, size=(
                            amount_to_add * final_2d_width * final_2d_width,
                            cnn_hyperparameters[first_fc]['out'], 1, 1)),
                        tf_running_activations: rolling_ativation_means[current_op],

                        tf_wvelocity_this: np.zeros(shape=(
                            cnn_hyperparameters[current_op]['weights'][0],
                            cnn_hyperparameters[current_op]['weights'][1],
                            cnn_hyperparameters[current_op]['weights'][2], amount_to_add),
                            dtype=np.float32),
                        tf_bvelocity_this: np.zeros(shape=(amount_to_add,), dtype=np.float32),
                        tf_wvelocity_next: np.zeros(shape=(
                            cnn_hyperparameters[next_conv_op]['weights'][0],
                            cnn_hyperparameters[next_conv_op]['weights'][1],
                            amount_to_add, cnn_hyperparameters[next_conv_op]['weights'][3]),
                            dtype=np.float32) if last_conv_id != current_op else
                        np.zeros(shape=(final_2d_width * final_2d_width * amount_to_add,
                                        cnn_hyperparameters[first_fc]['out'], 1, 1),
                                 dtype=np.float32),
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
            tf_in_size: cnn_hyperparameters[first_fc]['in']
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
    # Turned off issue #11
    '''pool_dataset, pool_labels = hard_pool_ft.get_pool_data(True)
    if research_parameters['pool_randomize'] and np.random.random() < research_parameters[
        'pool_randomize_rate']:
        try:
            pool = MPPool(processes=10)
            distorted_imgs = pool.map(distort_img, pool_dataset)
            pool_dataset = np.asarray(distorted_imgs)
            pool.close()
            pool.join()
        except Exception:
            raise AssertionError'''

    # this was done to increase performance and reduce overfitting
    # instead of optimizing with every single batch in the pool
    # we select few at random

    logger.info('\t(Before) Size of Rolling mean vector for %s: %s', current_op,
                rolling_ativation_means[current_op].shape)

    rolling_ativation_means[current_op] = np.append(rolling_ativation_means[current_op],
                                                    np.zeros(ai[1]))

    logger.info('\tSize of Rolling mean vector for %s: %s', current_op,
                rolling_ativation_means[current_op].shape)

    # This is a pretty important step
    # Unless you run this once, the sizes of weights do not change
    _ = session.run([tower_logits, tower_activation_update_ops], feed_dict=train_feed_dict)
    pbatch_train_count = 0

    current_adapted_op = current_op
    current_adapted_indices = np.arange(cnn_hyperparameters[current_op]['weights'][3] - ai[1],
                                      cnn_hyperparameters[current_op]['weights'][3])

    # Finetune the net with hard_pool_ft
    if hard_pool_ft.get_size() > batch_size:
        pool_dataset, pool_labels = hard_pool_ft.get_pool_data(True)
        if research_parameters['pool_randomize'] and np.random.random() < \
                research_parameters['pool_randomize_rate']:
            try:
                pool = MPPool(processes=10)
                distorted_imgs = pool.map(distort_img, pool_dataset)
                pool_dataset = np.asarray(distorted_imgs)
                pool.close()
                pool.join()
            except Exception:
                raise AssertionError
        # Train with latter half of the data
        for pool_id in range(0,
                             (hard_pool_ft.get_size() // batch_size) - 1, num_gpus):
            if np.random.random() < research_parameters['finetune_rate']:
                pool_feed_dict = {}
                for gpu_id in range(num_gpus):
                    pbatch_data = pool_dataset[(pool_id + gpu_id) * batch_size:(
                                                                                   pool_id + gpu_id + 1) * batch_size,
                                  :, :, :]
                    pbatch_labels = pool_labels[(pool_id + gpu_id) * batch_size:(
                                                                                    pool_id + gpu_id + 1) * batch_size,
                                    :]
                    pool_feed_dict.update({tf_pool_data_batch[gpu_id]: pbatch_data,
                                           tf_pool_label_batch[gpu_id]: pbatch_labels})

                _, _ = session.run([apply_pool_grads_op, update_pool_velocity_ops],
                                   feed_dict=pool_feed_dict)


def run_actual_remove_operation(session, current_op, li, last_conv_id, hard_pool_ft):
    _, rm_indices = session.run(tf_rm_filters_ops[current_op],
                                feed_dict={
                                    tf_action_info: np.asarray([li, 0, ai[1]]),
                                    tf_running_activations: rolling_ativation_means[current_op]
                                })
    rm_indices = rm_indices.flatten()
    amount_to_rmv = ai[1]

    with tf.variable_scope(TF_GLOBAL_SCOPE + TF_SCOPE_DIVIDER + current_op, reuse=True):
        current_op_weights = tf.get_variable(TF_WEIGHTS)

    if research_parameters['remove_filters_by'] == 'Activation':
        logger.debug('\tRemoving filters for op %s', current_op)
        logger.debug('\t\t\tIndices: %s', rm_indices[:10])

    elif research_parameters['remove_filters_by'] == 'Distance':
        logger.debug('\tRemoving filters for op %s', current_op)

        logger.debug('\t\tSimilarity summary')
        logger.debug('\t\t\tIndices: %s', rm_indices[:10])

        logger.debug('\t\tSize of indices to remove: %s/%d', rm_indices.size,
                     cnn_hyperparameters[current_op]['weights'][3])
        indices_of_filters_keep = list(
            set(np.arange(cnn_hyperparameters[current_op]['weights'][3])) - set(
                rm_indices.tolist()))
        logger.debug('\t\tSize of indices to keep: %s/%d', len(indices_of_filters_keep),
                     cnn_hyperparameters[current_op]['weights'][3])

    cnn_hyperparameters[current_op]['weights'][3] -= amount_to_rmv
    if research_parameters['debugging']:
        logger.debug('\tSize after feature map reduction: %s,%s', current_op,
                     tf.shape(current_op_weights).eval())
        assert tf.shape(current_op_weights).eval()[3] == \
               cnn_hyperparameters[current_op]['weights'][3]

    session.run(tf_update_hyp_ops[current_op], feed_dict={
        tf_weight_shape: cnn_hyperparameters[current_op]['weights']
    })

    if current_op == last_conv_id:
        with tf.variable_scope(TF_GLOBAL_SCOPE + TF_SCOPE_DIVIDER + first_fc, reuse=True):
            first_fc_weights = tf.get_variable(TF_WEIGHTS)

        cnn_hyperparameters[first_fc]['in'] -= final_2d_width * final_2d_width * amount_to_rmv
        if research_parameters['debugging']:
            logger.debug('\tSize after feature map reduction: %s,%s',
                         first_fc, str(tf.shape(first_fc_weights).eval()))

        session.run(tf_update_hyp_ops[first_fc], feed_dict={
            tf_in_size: cnn_hyperparameters[first_fc]['in']
        })

    else:
        next_conv_op = \
            [tmp_op for tmp_op in cnn_ops[cnn_ops.index(current_op) + 1:] if 'conv' in tmp_op][0]
        assert current_op != next_conv_op

        with tf.variable_scope(TF_GLOBAL_SCOPE + TF_SCOPE_DIVIDER + next_conv_op, reuse=True):
            next_conv_op_weights = tf.get_variable(TF_WEIGHTS)

        cnn_hyperparameters[next_conv_op]['weights'][2] -= amount_to_rmv

        if research_parameters['debugging']:
            logger.debug('\tSize after feature map reduction: %s,%s', next_conv_op,
                         str(tf.shape(next_conv_op_weights).eval()))
            assert tf.shape(next_conv_op_weights).eval()[2] == \
                   cnn_hyperparameters[next_conv_op]['weights'][2]

        session.run(tf_update_hyp_ops[next_conv_op], feed_dict={
            tf_weight_shape: cnn_hyperparameters[next_conv_op]['weights']
        })

    logger.info('\t(Before) Size of Rolling mean vector for %s: %s', current_op,
                rolling_ativation_means[current_op].shape)
    rolling_ativation_means[current_op] = np.delete(rolling_ativation_means[current_op],
                                                    rm_indices)
    logger.info('\tSize of Rolling mean vector for %s: %s', current_op,
                rolling_ativation_means[current_op].shape)

    # This is a pretty important step
    # Unless you run this onces, the sizes of weights do not change
    _ = session.run([tower_logits, tower_activation_update_ops], feed_dict=train_feed_dict)

    if hard_pool_ft.get_size() > batch_size:
        pool_dataset, pool_labels = hard_pool_ft.get_pool_data(True)
        '''if research_parameters['pool_randomize'] and np.random.random() < \
                research_parameters['pool_randomize_rate']:
            try:
                pool = MPPool(processes=10)
                distorted_imgs = pool.map(distort_img, pool_dataset)
                pool_dataset = np.asarray(distorted_imgs)
                pool.close()
                pool.join()
            except Exception:
                raise AssertionError'''

        # Train with latter hard_pool_ft
        for pool_id in range(0, (hard_pool_ft.get_size() // batch_size) - 1, num_gpus):
            if np.random.random() < research_parameters['finetune_rate']:
                pool_feed_dict = {}
                for gpu_id in range(num_gpus):
                    pbatch_data = pool_dataset[(pool_id + gpu_id) * batch_size:(
                                                                                   pool_id + gpu_id + 1) * batch_size,
                                  :, :, :]
                    pbatch_labels = pool_labels[(pool_id + gpu_id) * batch_size:(
                                                                                    pool_id + gpu_id + 1) * batch_size,
                                    :]
                    pool_feed_dict.update({tf_pool_data_batch[gpu_id]: pbatch_data,
                                           tf_pool_label_batch[gpu_id]: pbatch_labels})

                _, _ = session.run([apply_pool_grads_op, update_pool_velocity_ops],
                                   feed_dict=pool_feed_dict)

def run_actual_finetune_operation(hard_pool_ft):
    '''
    Run the finetune opeartion in the default session
    :param hard_pool_ft:
    :return:
    '''
    op = cnn_ops[li]
    pool_dataset, pool_labels = hard_pool_ft.get_pool_data(True)

    if research_parameters['pool_randomize'] and np.random.random() < research_parameters[
        'pool_randomize_rate']:
        '''try:
            pool = MPPool(processes=10)
            distorted_imgs = pool.map(distort_img, pool_dataset)
            pool_dataset = np.asarray(distorted_imgs)
            pool.close()
            pool.join()
        except Exception:
            raise AssertionError'''

    # without if can give problems in exploratory stage because of no data in the pool
    if hard_pool_ft.get_size() > batch_size:
        # Train with latter half of the data

        for pool_id in range(0, (hard_pool_ft.get_size() // batch_size) - 1, num_gpus):
            if np.random.random() < research_parameters['finetune_rate']:
                pool_feed_dict = {}
                for gpu_id in range(num_gpus):
                    pbatch_data = pool_dataset[(pool_id + gpu_id) * batch_size:(
                                                                                   pool_id + gpu_id + 1) * batch_size,
                                  :, :, :]
                    pbatch_labels = pool_labels[(pool_id + gpu_id) * batch_size:(
                                                                                    pool_id + gpu_id + 1) * batch_size,
                                    :]
                    pool_feed_dict.update({tf_pool_data_batch[gpu_id]: pbatch_data,
                                           tf_pool_label_batch[gpu_id]: pbatch_labels})

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


def get_explore_action_probs(epoch, trial_phase, n_conv):
    '''
    Explration action probabilities. We manually specify a probabilities of a stochastic policy
    used to explore the state space adequately
    :param epoch:
    :param trial_phase: use the global_trial_phase because at this time we only care about exploring actions
    :param n_conv:
    :return:
    '''
    if epoch == 0 and trial_phase<0.4:
        logger.info('Finetune phase')
        trial_action_probs = [0.0 / (1.0 * n_conv) for _ in range(n_conv)]  # remove
        trial_action_probs.extend([0.3 / (1.0 * n_conv) for _ in range(n_conv)])  # add
        trial_action_probs.extend([0.2, .5])

    elif epoch == 0 and trial_phase>=0.4 and trial_phase < 1.0:
        logger.info('Growth phase')
        # There is 0.1 amount probability to be divided between all the remove actions
        # We give 1/10 th as for other remove actions for the last remove action
        remove_action_prob = 0.1*(10.0/11.0)
        trial_action_probs_without_last = [remove_action_prob/(1.0*(n_conv-1)) for _ in range(n_conv-1)]
        trial_action_probs = list(trial_action_probs_without_last) + [0.1-remove_action_prob]

        trial_action_probs.extend([0.6 / (1.0 * n_conv) for _ in range(n_conv)])  # add
        trial_action_probs.extend([0.1, 0.2])

    elif epoch==1 and trial_phase>=1.0 and trial_phase<1.7:
        logger.info('Shrink phase')
        # There is 0.6 amount probability to be divided between all the remove actions
        # We give 1/10 th as for other remove actions for the last remove action
        remove_action_prob = 0.6 * (10.0 / 11.0)
        trial_action_probs_without_last = [remove_action_prob / (1.0 * (n_conv - 1)) for _ in range(n_conv - 1)]
        trial_action_probs = list(trial_action_probs_without_last) + [0.6 - remove_action_prob]

        trial_action_probs.extend([0.1 / (1.0 * n_conv) for _ in range(n_conv)])  # add
        trial_action_probs.extend([0.1, 0.2])

    elif epoch==1 and trial_phase>=1.7 and trial_phase<2.0:
        logger.info('Finetune phase')

        remove_action_prob = 0.3 * (10.0 / 11.0)
        trial_action_probs_without_last = [remove_action_prob / (1.0 * (n_conv - 1)) for _ in range(n_conv - 1)]
        trial_action_probs = list(trial_action_probs_without_last) + [0.3 - remove_action_prob]

        trial_action_probs.extend([0.0 / (1.0 * n_conv) for _ in range(n_conv)])  # add
        trial_action_probs.extend([0.2, 0.5])

    return trial_action_probs

# Continuous adaptation
def get_continuous_adaptation_action_in_different_epochs(q_learner, data, epoch, global_trial_phase, local_trial_phase, n_conv, eps, adaptation_period):
    '''
    Continuously adapting the structure
    :param q_learner:
    :param data:
    :param epoch:
    :param trial_phase (global and local):
    :param n_conv:
    :param eps:
    :param adaptation_period:
    :return:
    '''

    adapting_now = None
    if epoch == 0 or epoch == 1:
        # Grow the network mostly (Until half) then fix
        logger.info('Explore Stage')
        state, action, invalid_actions = q_learner.output_action_with_type(
            data, 'Explore', p_action=get_explore_action_probs(epoch, global_trial_phase, n_conv)
        )
        adapting_now = True

    else:
        logger.info('Epsilon: %.3f',eps)
        if adaptation_period=='first':
            if local_trial_phase<=0.5:
                logger.info('Greedy Adapting period of epoch')
                if np.random.random() >= eps:
                    state, action, invalid_actions = q_learner.output_action_with_type(
                        data, 'Greedy')

                else:
                    state, action, invalid_actions = q_learner.output_action_with_type(
                        data, 'Stochastic'
                    )
                adapting_now = True

            else:
                logger.info('Greedy Not adapting period of epoch')
                state, action, invalid_actions = q_learner.get_finetune_action(data)
                adapting_now = False

        elif adaptation_period == 'last':
            if local_trial_phase > 0.5:
                logger.info('Greedy Adapting period of epoch')
                if np.random.random() >= eps:
                    state, action, invalid_actions = q_learner.output_action_with_type(
                        data, 'Greedy')

                else:
                    state, action, invalid_actions = q_learner.output_action_with_type(
                        data, 'Stochastic'
                    )
                adapting_now=True
            else:
                logger.info('Not adapting period of epoch')
                state, action, invalid_actions = q_learner.get_finetune_action(data)
                adapting_now = False

        elif adaptation_period =='both':

            logger.info('Greedy Adapting period of epoch (both)')
            if np.random.random() >= eps:
                state, action, invalid_actions = q_learner.output_action_with_type(
                    data, 'Greedy')

            else:
                state, action, invalid_actions = q_learner.output_action_with_type(
                    data, 'Stochastic'
                )
            adapting_now = True

        else:
            raise NotImplementedError

    return state, action, invalid_actions, adapting_now


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


if __name__ == '__main__':

    # Various run-time arguments specified
    #
    allow_growth = False
    use_multiproc = False
    try:
        opts, args = getopt.getopt(
            sys.argv[1:], "", ["output_dir=", "num_gpus=", "memory=", 'pool_workers=', 'allow_growth=',
                               'dataset_type=', 'dataset_behavior=',
                               'adapt_structure=', 'rigid_pooling=',
                               'use_multiproc='])
    except getopt.GetoptError as err:
        print(err)
        print('<filename>.py --output_dir= --num_gpus= --memory= --pool_workers=')

    if len(opts) != 0:
        for opt, arg in opts:
            if opt == '--output_dir':
                output_dir = arg
            if opt == '--num_gpus':
                num_gpus = int(arg)
            if opt == '--memory':
                mem_frac = float(arg)
            if opt == '--pool_workers':
                pool_workers = int(arg)
            if opt == '--allow_growth':
                allow_growth = bool(arg)
            if opt == '--dataset_type':
                datatype = str(arg)
            if opt == '--dataset_behavior':
                behavior = str(arg)
            if opt == '--adapt_structure':
                adapt_structure = bool(int(arg))
            if opt == '--rigid_pooling':
                rigid_pooling = bool(int(arg))
            if opt == '--use_multiproc':
                use_multiproc = bool(arg)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Setting up loggers
    logger, perf_logger, cnn_structure_logger, \
    q_logger, class_dist_logger, pool_dist_logger, \
    hyp_logger, error_logger = setup_loggers(adapt_structure)

    logger.info('Created loggers')

    logger.info('Created Output directory: %s', output_dir)
    logger.info('Received all the required user arguments at Runtime')
    logger.debug('Output DIR: %s', output_dir)
    logger.debug('Number of GPUs: %d', num_gpus)
    logger.debug('Memory fraction per GPU: %.3f', mem_frac)
    logger.debug('Number of pool workers for MultiProcessing: %d', pool_workers)
    logger.debug('Dataset Name: %s', datatype)
    logger.debug('Data Behavior: %s', behavior)
    logger.debug('Use AdaCNN: %d', adapt_structure)
    logger.debug('Use rigid pooling: %d', rigid_pooling)
    # =====================================================================
    # VARIOS SETTING UPS
    # SET FROM MAIN FUNCTIONS OF OTHER CLASSES
    set_varialbes_with_input_arguments(datatype, behavior, adapt_structure,rigid_pooling)
    cnn_intializer.set_from_main(research_parameters, logging_level, logging_format)

    logger.info('Creating CNN hyperparameters and operations in the correct format')
    # Getting hyperparameters
    cnn_ops, cnn_hyperparameters, final_2d_width = utils.get_ops_hyps_from_string(dataset_info, cnn_string)
    init_cnn_ops, init_cnn_hyperparameters, final_2d_width = utils.get_ops_hyps_from_string(dataset_info, cnn_string)

    logger.info('Created CNN hyperparameters and operations in the correct format successfully\n')

    ada_cnn_adapter.set_from_main(research_parameters, final_2d_width,cnn_ops, cnn_hyperparameters, logging_level, logging_format)
    cnn_optimizer.set_from_main(research_parameters,model_hyperparameters,logging_level,logging_format,cnn_ops)

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

    elif datatype=='cifar-100':
        dataset_file = h5py.File("data" + os.sep + "cifar_100_dataset.hdf5", "r")

    train_dataset, train_labels = dataset_file['/train/images'], dataset_file['/train/labels']
    test_dataset, test_labels = dataset_file['/test/images'], dataset_file['/test/labels']

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

    # Defining pool
    logger.info('Defining pools of data (validation and finetuning)')
    hardness = 0.5
    hard_pool_valid = Pool(size=pool_size//2, batch_size=batch_size, image_size=image_size,
                           num_channels=num_channels, num_labels=num_labels, assert_test=False)
    hard_pool_ft = Pool(size=pool_size//2, batch_size=batch_size, image_size=image_size,
                           num_channels=num_channels, num_labels=num_labels, assert_test=False)
    logger.info('Defined pools of data successfully\n')

    first_fc = 'fulcon_out' if 'fulcon_0' not in cnn_ops else 'fulcon_0'
    # -1 is because we don't want to count pool_global

    layer_count = len([op for op in cnn_ops if 'conv' in op or 'pool' in op]) - 1

    # ids of the convolution ops
    convolution_op_ids = []
    for op_i, op in enumerate(cnn_ops):
        if 'conv' in op:
            convolution_op_ids.append(op_i)
    logger.info('Found all convolution opeartion ids')

    # Defining initial hyperparameters as TF variables (so we can change them later during adaptation)
    # Define global set (used for learning rate decay)
    # Define CNN variables
    # Define CNN variable momentums
    # Define all the required operations
    with tf.variable_scope(TF_GLOBAL_SCOPE,reuse=False) as scope:
        global_step = tf.get_variable(initializer=0, dtype=tf.int32, trainable=False, name='global_step')
        logger.info('Defining TF Hyperparameters')
        tf_cnn_hyperparameters = cnn_intializer.init_tf_hyperparameters(cnn_ops, cnn_hyperparameters)
        logger.info('Defining Weights and Bias for CNN operations')
        _ = cnn_intializer.initialize_cnn_with_ops(cnn_ops, cnn_hyperparameters)
        logger.info('Defining Velocities for Weights and Bias for CNN operations')
        _ = cnn_intializer.define_velocity_vectors(scope, cnn_ops, cnn_hyperparameters)
    logger.info(('=' * 80) + '\n')

    if research_parameters['adapt_structure']:
        # Adapting Policy Learner
        state_history_length = 4
        adapter = ada_cnn_qlearner.AdaCNNAdaptingQLearner(
            discount_rate=0.9, fit_interval=1,
            exploratory_tries_factor=5, exploratory_interval=10000, stop_exploring_after=10,
            filter_vector=filter_vector,
            conv_ids=convolution_op_ids, net_depth=layer_count,
            n_conv=len([op for op in cnn_ops if 'conv' in op]),
            epsilon=0.5, target_update_rate=20,
            batch_size=32, persist_dir=output_dir,
            session=session, random_mode=False,
            state_history_length=state_history_length,
            hidden_layers=[128, 64, 32], momentum=0.9, learning_rate=0.01,
            rand_state_length=32, add_amount=model_hyperparameters['add_amount'], remove_amount=model_hyperparameters['remove_amount'],
            num_classes=num_labels, filter_min_threshold=model_hyperparameters['filter_min_threshold'],
            trial_phase_threshold=1.0, qlearner_id=0
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

    data_gen = data_generator.DataGenerator(batch_size,num_labels,dataset_info['train_size'],dataset_info['n_slices'],
                                            image_size, dataset_info['n_channels'], dataset_info['resize_to'],
                                            dataset_info['dataset_name'], session)

    if datatype=='cifar-10':
        labels_per_task = 5
        labels_of_each_task = [[0,1,2,3,4],[2,3,4,5,6],[5,6,7,8,9],[8,9,0,1,2]]

    elif datatype=='cifar-100':
        labels_per_task = 25

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
    current_state, current_action,curr_adaptation_status = None, None,None
    prev_unseen_valid_accuracy = 0

    current_q_learn_op_id = 0
    logger.info('Convolutional Op IDs: %s', convolution_op_ids)

    logger.info('Starting Training Phase')

    # Check if loss is stabilized (for starting adaptations)
    previous_loss = 1e5  # used for the check to start adapting

    # Reward for Q-Learner
    prev_pool_accuracy = 0
    max_pool_accuracy = 0

    # Stop and start adaptations when necessary
    start_adapting = False
    stop_adapting = False
    adapt_period = 'both'
    # need to have a starting value because if the algorithm choose to add the data to validation set very first step
    train_accuracy = 0

    n_iter_per_task = n_iterations//n_tasks

    current_adapted_op, current_adapted_indices, current_action_type = None,None, adapter.get_donothing_action_type()

    for epoch in range(n_epochs):

        for task in range(n_tasks):

            if np.random.random()<0.8:
                research_parameters['momentum']=0.9
                research_parameters['pool_momentum']=0.0
            else:
                research_parameters['momentum']=0.0
                research_parameters['pool_momentum']=0.9

            cnn_optimizer.update_hyperparameters(research_parameters)

            # we stop 'num_gpus' items before the ideal number of training batches
            # because we always keep num_gpus items for valid data in memory
            for batch_id in range(0, n_iter_per_task - num_gpus, num_gpus):

                global_batch_id = (n_iterations * epoch) + (task*n_iter_per_task) + batch_id
                global_trial_phase = (global_batch_id * 1.0 / n_iterations)
                local_trial_phase = batch_id * 1.0/n_iter_per_task

                t0 = time.clock()  # starting time for a batch

                logger.debug('=' * 80)
                #logger.debug('tf op count: %d', len(graph.get_operations()))
                logger.debug('=' * 80)

                #logger.info('\tTraining with batch %d', batch_id)

                # We load 1 extra batch (chunk_size+1) because we always make the valid batch the batch_id+1

                if batch_id==0:
                    logger.info('\tDataset shape: %s', train_dataset.shape)
                    logger.info('\tLabels shape: %s', train_labels.shape)

                # Feed dicitonary with placeholders for each tower
                batch_data, batch_labels, batch_weights = [], [], []
                train_feed_dict = {}

                # ========================================================
                # Creating data batchs for the towers
                for gpu_id in range(num_gpus):
                    label_seq = label_sequence_generator.sample_label_sequence_for_batch(n_iterations, data_prior,
                                                                                         batch_size, num_labels)
                    logger.debug('Got label sequence (for batch %d)', global_batch_id)
                    logger.debug(Counter(label_seq))

                    b_d, b_l = data_gen.generate_data_with_label_sequence(train_dataset, train_labels, label_seq, dataset_info)
                    batch_data.append(b_d)
                    batch_labels.append(b_l)

                    if (batch_id + gpu_id) % research_parameters['log_distribution_every'] == 0:
                        cnt = Counter(label_seq)
                        dist_str = ''
                        for li in range(num_labels):
                            dist_str += str(cnt[li] / len(label_seq)) + ',' if li in cnt else str(0) + ','
                        class_dist_logger.info('%d,%s', batch_id, dist_str)

                    cnt = Counter(np.argmax(batch_labels[-1], axis=1))
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

                l, super_loss_vec, current_activations_list, train_predictions = session.run(
                    [mean_loss_op, concat_loss_vec_op,
                     tf_mean_activation, tower_predictions], feed_dict=train_feed_dict
                )
                # =========================================================

                # ==========================================================
                # Updating Pools of data

                if np.random.random()<0.1*(valid_acc_decay**(epoch)):
                    if adapt_structure or rigid_pooling:

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

                        train_accuracy = np.mean(
                            [accuracy(train_predictions[gid], batch_labels[gid]) for gid in range(num_gpus)]) / 100.0
                        hard_pool_valid.add_hard_examples(single_iteration_batch_data, single_iteration_batch_labels,
                                                          super_loss_vec,
                                                          min(research_parameters['hard_pool_max_threshold'],
                                                              max(0.1, (1.0 - train_accuracy))))

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
                    if np.random.random()<0.2:
                        hard_pool_ft.add_hard_examples(single_iteration_batch_data, single_iteration_batch_labels,
                                                    super_loss_vec, min(research_parameters['hard_pool_max_threshold'],
                                                                  max(0.1, (1.0 - train_accuracy))))
                        logger.debug('\tPool size (FT): %d', hard_pool_ft.get_size())

                    # =========================================================
                    # Training Phase (Optimization)
                    # If not adapt structure, naively optimize CNN
                    if not adapt_structure:
                        for _ in range(iterations_per_batch):
                            _, _ = session.run(
                                [apply_grads_op, update_train_velocity_op], feed_dict=train_feed_dict
                            )
                    # if adapt_structure, only optimize the newly added parameters with the
                    # new training data
                    else:
                        for _ in range(iterations_per_batch):
                            if current_action_type==adapter.get_add_action_type():

                                train_feed_dict.update({tf_indices: current_adapted_indices})
                                _, _ = session.run(
                                    [tf_slice_optimize[current_adapted_op], tf_slice_vel_update[current_op]], feed_dict=train_feed_dict
                                )

                            elif current_action_type == adapter.get_donothing_action_type() or not curr_adaptation_status:
                                _, _ = session.run(
                                    [apply_grads_op, update_train_velocity_op], feed_dict=train_feed_dict
                                )

                t1_train = time.clock()

                current_activations = get_activation_dictionary(current_activations_list, cnn_ops, convolution_op_ids)

                if np.isnan(l):
                    logger.critical('Diverged (NaN detected) (batchID) %d (last Cost) %.3f', batch_id,
                                    train_losses[-1])
                assert not np.isnan(l)

                # rolling activation mean update
                if research_parameters['adapt_structure']:
                    for op, op_activations in current_activations.items():
                        logger.debug('checking %s', op)
                        logger.debug('\tRolling size (%s): %s', op, rolling_ativation_means[op].shape)
                        logger.debug('\tCurrent size (%s): %s', op, op_activations.shape)
                    for op, op_activations in current_activations.items():
                        assert current_activations[op].size == cnn_hyperparameters[op]['weights'][3], \
                            'did not match (op %s). activation %d cnn_hyp %d' \
                            % (op, current_activations[op].size, cnn_hyperparameters[op]['weights'][3])

                        rolling_ativation_means[op] = act_decay * rolling_ativation_means[op] + current_activations[op]

                train_losses.append(l)

                # =============================================================
                # Validation Phase (Use single tower) (Before adaptations)
                v_label_seq = label_sequence_generator.sample_label_sequence_for_batch(
                    n_iterations,data_prior,batch_size,num_labels,freeze_index_increment=True
                )
                batch_valid_data,batch_valid_labels = data_gen.generate_data_with_label_sequence(
                    train_dataset,train_labels,v_label_seq,dataset_info
                )

                feed_valid_dict = {tf_valid_data_batch: batch_valid_data, tf_valid_label_batch: batch_valid_labels}
                unseen_valid_predictions = session.run(valid_predictions_op, feed_dict=feed_valid_dict)
                unseen_valid_accuracy = accuracy(unseen_valid_predictions, batch_valid_labels)
                # =============================================================

                # ================================================================
                # Things done if one of below scenarios
                # For AdaCNN if adaptations stopped
                # For rigid pool CNNs from the beginning
                if ((research_parameters['pooling_for_nonadapt'] and
                         not research_parameters['adapt_structure']) or stop_adapting) and \
                        (batch_id > 0 and batch_id % interval_parameters['finetune_interval'] == 0):

                    logger.info('Pooling for non-adaptive CNN')

                    if research_parameters['adapt_structure']:
                        logger.info('Adaptations stopped. Finetune is at its maximum utility (Batch: %d)' % (
                        global_batch_id))

                        logger.info('Using dropout rate of: %.3f', dropout_rate)

                    # ===============================================================
                    # Finetune with data in hard_pool_ft
                    fintune_with_pool_ft(hard_pool_ft)
                    # =================================================================
                    # Calculate pool accuracy (hard_pool_valid)
                    mean_pool_accuracy = get_pool_valid_accuracy(hard_pool_valid)
                    logger.info('\tPool accuracy (hard_pool_valid): %.5f', mean_pool_accuracy)
                    # ==================================================================

                # ==============================================================
                # Testing Phase
                if batch_id % interval_parameters['test_interval'] == 0:
                    mean_train_loss = np.mean(train_losses)
                    logger.info('=' * 60)
                    logger.info('\tBatch ID: %d' % batch_id)
                    if decay_learning_rate:
                        logger.info('\tLearning rate: %.5f' % session.run(tf_learning_rate))
                    else:
                        logger.info('\tLearning rate: %.5f' % start_lr)

                    logger.info('\tMinibatch Mean Loss: %.3f' % mean_train_loss)
                    logger.info('\tValidation Accuracy (Unseen): %.3f' % unseen_valid_accuracy)
                    logger.info('\tValidation accumulation rate %.3f', 0.5*(valid_acc_decay**epoch))
                    logger.info('\tTrial phase: %.3f (Local) %.3f (Global)', local_trial_phase, global_trial_phase)

                    test_accuracies = []
                    for test_batch_id in range(test_size // batch_size):
                        batch_test_data = test_dataset[test_batch_id * batch_size:(test_batch_id + 1) * batch_size, :, :, :]
                        batch_test_labels = test_labels[test_batch_id * batch_size:(test_batch_id + 1) * batch_size, :]

                        batch_ohe_test_labels = np.zeros((batch_size,num_labels),dtype=np.float32)
                        batch_ohe_test_labels[np.arange(batch_size),batch_test_labels[:,0]] = 1.0
                        feed_test_dict = {tf_test_dataset: batch_test_data, tf_test_labels: batch_ohe_test_labels}
                        test_predictions = session.run(test_predicitons_op, feed_dict=feed_test_dict)
                        test_accuracies.append(accuracy(test_predictions, batch_ohe_test_labels))

                        if test_batch_id < 10:

                            logger.debug('=' * 80)
                            logger.debug('Actual Test Labels %d', test_batch_id)
                            logger.debug(np.argmax(batch_test_labels, axis=1).flatten()[:5])
                            logger.debug('Predicted Test Labels %d', test_batch_id)
                            logger.debug(np.argmax(test_predictions, axis=1).flatten()[:5])
                            logger.debug('Test: %d, %.3f', test_batch_id, accuracy(test_predictions, batch_test_labels))
                            logger.debug('=' * 80)

                    current_test_accuracy = np.mean(test_accuracies)
                    logger.info('\tTest Accuracy: %.3f' % current_test_accuracy)
                    logger.info('=' * 60)
                    logger.info('')

                    # Logging error
                    prev_test_accuracy = current_test_accuracy
                    error_logger.info('%d,%.3f,%.3f,%.3f',
                                      global_batch_id, mean_train_loss,
                                      unseen_valid_accuracy, np.mean(test_accuracies)
                                      )
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

                        current_action_type = adapter.get_donothing_action_type()
                        pool_dataset, pool_labels = hard_pool_ft.get_pool_data(True)

                        # without if can give problems in exploratory stage because of no data in the pool
                        if hard_pool_ft.get_size() > batch_size:
                            # Train with latter half of the data
                            for pool_id in range(0, (hard_pool_ft.get_size() // batch_size) - 1, num_gpus):
                                if np.random.random() < research_parameters['finetune_rate']:
                                    pool_feed_dict = {}
                                    for gpu_id in range(num_gpus):
                                        pbatch_data = pool_dataset[
                                                      (pool_id + gpu_id) * batch_size:
                                                      (pool_id + gpu_id + 1) * batch_size, :, :, :]
                                        pbatch_labels = pool_labels[
                                                        (pool_id + gpu_id) * batch_size:
                                                        (pool_id + gpu_id + 1) * batch_size,:]

                                        pool_feed_dict.update({tf_pool_data_batch[gpu_id]: pbatch_data,
                                                               tf_pool_label_batch[gpu_id]: pbatch_labels})

                                    _, _ = session.run([apply_pool_grads_op, update_pool_velocity_ops],
                                                       feed_dict=pool_feed_dict)
                    # ==================================================================

                    # ==================================================================
                    # Actual Adaptations
                    if (start_adapting and not stop_adapting) and batch_id > 0 and \
                                            batch_id % interval_parameters['policy_interval'] == 0:

                        # ==================================================================
                        # Policy Update (Update policy only when we take actions actually using the qlearner)
                        # (Not just outputting finetune action)
                        # ==================================================================
                        if current_state is not None and curr_adaptation_status:

                            # ==================================================================
                            # Calculating pool accuracy
                            pool_accuracy = []
                            pool_dataset, pool_labels = hard_pool_valid.get_pool_data(False)

                            for pool_id in range(hard_pool_valid.get_size() // batch_size):
                                pbatch_data = pool_dataset[pool_id * batch_size:(pool_id + 1) * batch_size, :, :, :]
                                pbatch_labels = pool_labels[pool_id * batch_size:(pool_id + 1) * batch_size, :]
                                pool_feed_dict = {tf_pool_data_batch[0]: pbatch_data,
                                                  tf_pool_label_batch[0]: pbatch_labels}
                                p_predictions = session.run(pool_pred, feed_dict=pool_feed_dict)
                                if num_labels <= 100:
                                    pool_accuracy.append(accuracy(p_predictions, pbatch_labels))
                                else:
                                    pool_accuracy.append(top_n_accuracy(p_predictions, pbatch_labels, 5))
                            # ===============================================================================

                            # don't use current state as the next state, current state is for a different layer
                            # Calculate the new state after executing current action on the previous state
                            next_state = []
                            affected_layer_index = 0
                            for li, la in enumerate(current_action):
                                if la is None:
                                    assert li not in convolution_op_ids
                                    next_state.append(0)
                                    continue
                                elif la[0] == 'add':
                                    next_state.append(current_state[li] + la[1])
                                    affected_layer_index = li
                                elif la[0] == 'remove':
                                    next_state.append(current_state[li] - la[1])
                                    affected_layer_index = li
                                else:
                                    next_state.append(current_state[li])

                            next_state = tuple(next_state)

                            logger.info(('=' * 25)+' Update Summary ' + ('=' * 25))
                            logger.info('\tState (prev): %s', str(current_state))
                            logger.info('\tAction (prev): %s', str(current_action))
                            logger.info('\tState (next): %s', str(next_state))
                            p_accuracy = np.mean(pool_accuracy) if len(pool_accuracy) > 2 else 0
                            logger.info('\tPool Accuracy: %.3f', p_accuracy)
                            logger.info('\tValid accuracy (Before Adapt): %.3f', prev_unseen_valid_accuracy)
                            logger.info('\tValid accuracy (After Adapt): %.3f', unseen_valid_accuracy)
                            logger.info('\tPrev pool Accuracy: %.3f\n', prev_pool_accuracy)
                            logger.info(('=' * 80) + '\n')
                            assert not np.isnan(p_accuracy)

                            # ================================================================================
                            # Actual updating of the policy
                            adapter.update_policy({'prev_state': current_state, 'prev_action': current_action,
                                                   'curr_state': next_state,
                                                   'next_accuracy': None,
                                                   'prev_accuracy': None,
                                                   'pool_accuracy': p_accuracy,
                                                   'prev_pool_accuracy': prev_pool_accuracy,
                                                   'max_pool_accuracy': max_pool_accuracy,
                                                   'unseen_valid_accuracy': unseen_valid_accuracy,
                                                   'prev_unseen_valid_accuracy': prev_unseen_valid_accuracy,
                                                   'invalid_actions': curr_invalid_actions,
                                                   'batch_id': global_batch_id,
                                                   'layer_index': affected_layer_index}, True)
                            # ===================================================================================

                            cnn_structure_logger.info(
                                '%d:%s:%s:%.5f:%s', global_batch_id, current_state,
                                current_action, np.mean(pool_accuracy),
                                utils.get_cnn_string_from_ops(cnn_ops, cnn_hyperparameters)
                            )

                            q_logger.info('%d,%.5f', global_batch_id, adapter.get_average_Q())

                            logger.debug('Resetting both data distribution means')

                            max_pool_accuracy = max(max_pool_accuracy, p_accuracy)
                            prev_pool_accuracy = p_accuracy
                            prev_unseen_valid_accuracy = unseen_valid_accuracy

                        # ============================================================
                        # Outputting the action and executing the action
                        # =============================================================
                        filter_dict, filter_list = {}, []
                        for op_i, op in enumerate(cnn_ops):
                            if 'conv' in op:
                                filter_dict[op_i] = cnn_hyperparameters[op]['weights'][3]
                                filter_list.append(cnn_hyperparameters[op]['weights'][3])
                            elif 'pool' in op and op != 'pool_global':
                                filter_dict[op_i] = 0
                                filter_list.append(0)

                        # current_state, current_action, curr_invalid_actions = adapter.output_action(
                        #    {'filter_counts': filter_dict, 'filter_counts_list': filter_list})
                        current_state, current_action, curr_invalid_actions,curr_adaptation_status = get_continuous_adaptation_action_in_different_epochs(
                            adapter, data = {'filter_counts': filter_dict, 'filter_counts_list': filter_list}, epoch=epoch,
                            global_trial_phase=global_trial_phase, local_trial_phase=local_trial_phase, n_conv=len(convolution_op_ids),
                            eps=start_eps, adaptation_period=adapt_period)

                        current_action_type = adapter.get_action_type_with_action_list(current_action)
                        logger.info('Current action type: %s',current_action_type)

                        adapter.update_trial_phase(global_trial_phase)

                        for li, la in enumerate(current_action):
                            # pooling and fulcon layers
                            if la is None or la[0] == 'do_nothing':
                                continue

                            logger.info('Got state: %s, action: %s', str(current_state), str(la))

                            # where all magic happens (adding and removing filters)
                            si, ai = current_state, la
                            current_op = cnn_ops[li]

                            for tmp_op in reversed(cnn_ops):
                                if 'conv' in tmp_op:
                                    last_conv_id = tmp_op
                                    break

                            if 'conv' in current_op and ai[0] == 'add':

                                run_actual_add_operation(session,current_op,li,last_conv_id,hard_pool_ft)

                            elif 'conv' in current_op and ai[0] == 'remove':

                                run_actual_remove_operation(session,current_op,li,last_conv_id,hard_pool_ft)

                            elif 'conv' in current_op and ai[0] == 'finetune':
                                # pooling takes place here
                                run_actual_finetune_operation(hard_pool_ft)
                                break

                    if batch_id > 0 and batch_id % interval_parameters['history_dump_interval'] == 0:

                        # with open(output_dir + os.sep + 'Q_' + str(epoch) + "_" + str(batch_id)+'.pickle', 'wb') as f:
                        #    pickle.dump(adapter.get_Q(), f, pickle.HIGHEST_PROTOCOL)

                        pool_dist_string = ''
                        for val in hard_pool_valid.get_class_distribution():
                            pool_dist_string += str(val) + ','

                        pool_dist_logger.info('%s%d', pool_dist_string, hard_pool_valid.get_size())

                    t1 = time.clock()
                    op_count = len(tf.get_default_graph().get_operations())
                    var_count = len(tf.global_variables()) + len(tf.local_variables()) + len(tf.model_variables())
                    perf_logger.info('%d,%.5f,%.5f,%d,%d', global_batch_id, t1 - t0,
                                     (t1_train - t0_train) / num_gpus, op_count, var_count)

        # =======================================================
        # Decay learning rate (if set)
        if (research_parameters['adapt_structure'] or research_parameters['pooling_for_nonadapt']) and decay_learning_rate and epoch > 1:
            session.run(increment_global_step_op)
        # ======================================================

        # AdaCNN Algorithm
        if research_parameters['adapt_structure']:
            if epoch > 1:
                start_eps = max([start_eps*eps_decay,0.1])
                adapt_period = np.random.choice(['first','last','both'])
                # Turned off issue #11
                # adapt_period = np.random.choice(['first','last','both'])
                # At the moment not stopping adaptations for any reason
                # stop_adapting = adapter.check_if_should_stop_adapting()
        else:
            # Noninc pool algorithm
            if research_parameters['pooling_for_nonadapt']:
                session.run(increment_global_step_op)
            # Noninc algorithm
            else:
                session.run(increment_global_step_op)
