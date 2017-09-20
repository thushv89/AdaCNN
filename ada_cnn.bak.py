__author__ = 'Thushan Ganegedara'

import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import numpy as np
import os
from math import ceil, floor
import load_data
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

##################################################################
# AdaCNN Adapter
# ===============================================================
# AdaCNN Adapter will adapt the final model proposed by AdaCNN Constructor
# Then runs e-greedy Q-Learn for all the layer to decide of more filters required
# or should remove some. But keep in mind number of layers will be fixed
##################################################################

logger = None
logging_level = logging.INFO
logging_format = '[%(funcName)s] %(message)s'

dataset_name = 'imagenet-250'
dataset_behavior = 'non-stationary'
dataset_dir = ''
adapt_structure = True
use_pooling = False

if dataset_name == 'imagenet-250':
    num_labels = 250
elif dataset_name == 'cifar-10':
    num_labels = 10
elif dataset_name == 'svhn-10':
    num_labels = 10
else:
    raise NotImplementedError

# interval parameters
interval_parameters = cnn_hyperparameters_getter.get_interval_related_hyperparameters(dataset_name)

# Research parameters
research_parameters = cnn_hyperparameters_getter.get_research_hyperparameters(dataset_name, adapt_structure, use_pooling)

# Data specific parameters
data_hyperparameters = cnn_hyperparameters_getter.get_data_specific_hyperparameters(dataset_name, dataset_behavior, dataset_dir)
image_size = data_hyperparameters['image_size']
num_channels = data_hyperparameters['n_channels']

# Constant Strings
TF_WEIGHTS = constants.TF_WEIGHTS
TF_BIAS = constants.TF_BIAS
TF_ACTIVAIONS_STR = constants.TF_ACTIVAIONS_STR
TF_FC_WEIGHT_IN_STR = constants.TF_FC_WEIGHT_IN_STR
TF_FC_WEIGHT_OUT_STR = constants.TF_FC_WEIGHT_OUT_STR
TF_LOSS_VEC_STR = constants.TF_LOSS_VEC_STR

# Model Hyperparameters
model_hyperparameters = cnn_hyperparameters_getter.get_model_specific_hyperparameters(dataset_name, dataset_behavior, adapt_structure, use_pooling, num_labels)

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


def inference(dataset, tf_cnn_hyperparameters, training):
    global cnn_ops

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
                    op, cnn_hyperparameters_getter[op]['weights'], cnn_hyperparameters_getter[op]['stride']))
                logger.debug('\t\tWeights: %s', tf.shape(tf.get_variable(TF_WEIGHTS)).eval())
                w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)

                x = tf.nn.conv2d(x, w, cnn_hyperparameters_getter[op]['stride'],
                                 padding=cnn_hyperparameters_getter[op]['padding'])
                x = utils.lrelu(x + b, name=scope.name + '/top')

                activation_ops.append(
                    tf.assign(tf.get_variable(TF_ACTIVAIONS_STR), tf.reduce_max(x, [0, 1, 2]), validate_shape=False))

                if use_loc_res_norm and op == last_conv_id:
                    x = tf.nn.local_response_normalization(x, depth_radius=lrn_radius, alpha=lrn_alpha,
                                                           beta=lrn_beta)  # hyperparameters from tensorflow cifar10 tutorial

        if 'pool' in op:
            logger.debug('\tPooling (%s) with Kernel:%s Stride:%s' % (
                op, cnn_hyperparameters_getter[op]['kernel'], cnn_hyperparameters_getter[op]['stride']))
            if cnn_hyperparameters_getter[op]['type'] is 'max':
                x = tf.nn.max_pool(x, ksize=cnn_hyperparameters_getter[op]['kernel'],
                                   strides=cnn_hyperparameters_getter[op]['stride'],
                                   padding=cnn_hyperparameters_getter[op]['padding'])
            elif cnn_hyperparameters_getter[op]['type'] is 'avg':
                x = tf.nn.avg_pool(x, ksize=cnn_hyperparameters_getter[op]['kernel'],
                                   strides=cnn_hyperparameters_getter[op]['stride'],
                                   padding=cnn_hyperparameters_getter[op]['padding'])
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

                    logger.debug('Input size of fulcon_out : %d', cnn_hyperparameters_getter[op]['in'])
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


def inc_global_step(global_step):
    return global_step + 1


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


def distort_img_with_tf(distort_img_batch):
    dist_img_list = []
    img_list = tf.unpack(distort_img_batch, axis=0)
    for img in img_list:

        if np.random.random() < 0.2:
            img = tf.image.flip_left_right(img)
        if np.random.random() < 0.6:
            img = tf.image.random_brightness(img, 0.6)
        if np.random.random() < 0.8:
            img = tf.image.random_contrast(img, 0.2, 1.2)

        dist_img_list.append(img)
    return tf.pack(dist_img_list, axis=0)


def setup_loggers(research_parameters):
    logger = logging.getLogger('main_logger')
    logger.setLevel(logging_level)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(logging_format))
    console.setLevel(logging_level)
    logger.addHandler(console)

    error_logger = logging.getLogger('error_logger')
    error_logger.setLevel(logging.INFO)
    errHandler = logging.FileHandler(output_dir + os.sep + 'Error.log', mode='w')
    errHandler.setFormatter(logging.Formatter('%(message)s'))
    error_logger.addHandler(errHandler)
    error_logger.info('#Batch_ID,Loss(Train),Valid(Unseen),Test Accuracy')

    perf_logger = logging.getLogger('time_logger')
    perf_logger.setLevel(logging.INFO)
    perfHandler = logging.FileHandler(output_dir + os.sep + 'time.log', mode='w')
    perfHandler.setFormatter(logging.Formatter('%(message)s'))
    perf_logger.addHandler(perfHandler)
    perf_logger.info('#Batch_ID,Time(Full),Time(Train),Op count, Var count')

    if research_parameters['adapt_structure']:
        accuracy_drop_logger = logging.getLogger('accuracy_drop_logger')
        accuracy_drop_logger.setLevel(logging.INFO)
        dropHandler = logging.FileHandler(output_dir + os.sep + 'test_accuracy_drop.log', mode='w')
        dropHandler.setFormatter(logging.Formatter('%(message)s'))
        accuracy_drop_logger.addHandler(dropHandler)
        accuracy_drop_logger.info(
            '#Accuracy Drop Logger\n#last 10 (states,actions,current_pool_accuracy,prev_pool_accuracy) and accuracy drop\n')

        cnn_structure_logger = logging.getLogger('cnn_structure_logger')
        cnn_structure_logger.setLevel(logging.INFO)
        structHandler = logging.FileHandler(output_dir + os.sep + 'cnn_structure.log', mode='w')
        structHandler.setFormatter(logging.Formatter('%(message)s'))
        cnn_structure_logger.addHandler(structHandler)
        cnn_structure_logger.info('#batch_id:state:action:reward:#layer_1_hyperparameters#layer_2_hyperparameters#...')

        q_logger = logging.getLogger('q_logger')
        q_logger.setLevel(logging.INFO)
        qHandler = logging.FileHandler(output_dir + os.sep + 'QMetric.log', mode='w')
        qHandler.setFormatter(logging.Formatter('%(message)s'))
        q_logger.addHandler(qHandler)
        q_logger.info('#batch_id,pred_q')

    if research_parameters['log_class_distribution']:
        class_dist_logger = logging.getLogger('class_dist_logger')
        class_dist_logger.setLevel(logging.INFO)
        class_distHandler = logging.FileHandler(output_dir + os.sep + 'class_distribution.log', mode='w')
        class_distHandler.setFormatter(logging.Formatter('%(message)s'))
        class_dist_logger.addHandler(class_distHandler)

    pool_dist_logger = logging.getLogger('pool_distribution_logger')
    pool_dist_logger.setLevel(logging.INFO)
    poolHandler = logging.FileHandler(output_dir + os.sep + 'pool_distribution.log', mode='w')
    poolHandler.setFormatter(logging.Formatter('%(message)s'))
    pool_dist_logger.addHandler(poolHandler)
    pool_dist_logger.info('#Class distribution')

    return logger, perf_logger, accuracy_drop_logger, cnn_structure_logger, q_logger, class_dist_logger, pool_dist_logger


def get_activation_dictionary(activation_list, cnn_ops, conv_op_ids):
    current_activations = {}
    for act_i, layer_act in enumerate(current_activations_list):
        current_activations[cnn_ops[conv_op_ids[act_i]]] = layer_act
    return current_activations

state_action_history = []

cnn_ops, cnn_hyperparameters = None, None
num_gpus = -1


def define_tf_ops():
    tf_train_data_batch, tf_train_label_batch, tf_data_weights = [], [], []
    tf_pool_data_batch, tf_pool_label_batch = [], []
    tf_valid_data_batch, tf_valid_label_batch = None, None

    # tower_grads will contain
    # [[(grad0gpu0,var0gpu0),...,(gradNgpu0,varNgpu0)],...,[(grad0gpuD,var0gpuD),...,(gradNgpuD,varNgpuD)]]
    tower_grads, tower_loss_vectors, tower_losses, tower_activation_update_ops, tower_predictions = [], [], [], [], []
    tower_pool_grads, tower_pool_losses, tower_pool_activation_update_ops = [], [], []
    tower_logits = []
    with tf.variable_scope(TF_GLOBAL_SCOPE):
        for gpu_id in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_id):
                with tf.name_scope('%s_%d' % (TOWER_NAME, gpu_id)) as scope:
                    tf.get_variable_scope().reuse_variables()
                    # Input train data
                    tf_train_data_batch.append(tf.placeholder(tf.float32,
                                                              shape=(
                                                                  batch_size, image_size, image_size, num_channels),
                                                              name='TrainDataset'))
                    tf_train_label_batch.append(
                        tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='TrainLabels'))
                    tf_data_weights.append(tf.placeholder(tf.float32, shape=(batch_size), name='TrainWeights'))

                    # Training data opearations
                    tower_logit_op, tower_tf_activation_ops = inference(tf_train_data_batch[-1],
                                                                        tf_cnn_hyperparameters, True)
                    tower_logits.append(tower_logit_op)
                    tower_activation_update_ops.append(tower_tf_activation_ops)

                    tf_tower_loss = tower_loss(tf_train_data_batch[-1], tf_train_label_batch[-1], True,
                                               tf_data_weights[-1], tf_cnn_hyperparameters)
                    print(session.run(tf_cnn_hyperparameters))
                    tower_losses.append(tf_tower_loss)
                    tf_tower_loss_vec = calc_loss_vector(scope, tf_train_data_batch[-1], tf_train_label_batch[-1],
                                                         tf_cnn_hyperparameters)
                    tower_loss_vectors.append(tf_tower_loss_vec)

                    tower_grad = gradients(optimizer, tf_tower_loss, global_step,
                                           tf.constant(start_lr, dtype=tf.float32))
                    tower_grads.append(tower_grad)

                    tower_pred = predict_with_dataset(tf_train_data_batch[-1], tf_cnn_hyperparameters)
                    tower_predictions.append(tower_pred)

                    # Pooling data operations
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
                        single_pool_grad = gradients(optimizer, single_pool_loss, global_step, start_lr)
                        tower_pool_grads.append(single_pool_grad)

    logger.info('GLOBAL_VARIABLES (all)')
    logger.info('\t%s\n', [v.name for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
    logger.info('GLOBAL_VARIABLES')
    logger.info('\t%s\n', [v.name for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=TF_GLOBAL_SCOPE)])
    logger.info('TRAINABLE_VARIABLES')
    logger.info('\t%s\n',
                [v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=TF_GLOBAL_SCOPE)])

    with tf.device('/gpu:0'):
        # Train data operations
        # avg_grad_and_vars = [(avggrad0,var0),(avggrad1,var1),...]
        tf_avg_grad_and_vars = average_gradients(tower_grads)
        apply_grads_op = apply_gradient_with_momentum(optimizer, start_lr, global_step)
        concat_loss_vec_op = concat_loss_vector_towers(tower_loss_vectors)
        update_train_velocity_op = update_train_momentum_velocity(tf_avg_grad_and_vars)
        tf_mean_activation = mean_tower_activations(tower_activation_update_ops)
        mean_loss_op = tf.reduce_mean(tower_losses)

        # Pool data operations
        tf_pool_avg_gradvars = average_gradients(tower_pool_grads)
        apply_pool_grads_op = apply_gradient_with_pool_momentum(optimizer, start_lr, global_step)
        update_pool_velocity_ops = update_pool_momentum_velocity(tf_pool_avg_gradvars)
        tf_mean_pool_activations = mean_tower_activations(tower_pool_activation_update_ops)
        mean_pool_loss = tf.reduce_mean(tower_pool_losses)

    with tf.variable_scope(TF_GLOBAL_SCOPE) as main_scope, tf.device('/gpu:0'):

        increment_global_step_op = tf.assign(global_step, global_step + 1)

        # GLOBAL: Tensorflow operations for hard_pool
        with tf.name_scope('pool') as scope:
            tf.get_variable_scope().reuse_variables()
            pool_pred = predict_with_dataset(tf_pool_data_batch[0], tf_cnn_hyperparameters)

        # TODO: Scope?
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
                tf_slice_optimize = {}
                tf_slice_vel_update = {}
                tf_add_filters_ops, tf_rm_filters_ops, tf_replace_ind_ops = {}, {}, {}

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
                tf_update_hyp_ops = {}

                tf_reset_cnn = reset_cnn(init_cnn_hyperparameters)

                for tmp_op in cnn_ops:
                    if 'conv' in tmp_op:
                        tf_update_hyp_ops[tmp_op] = update_tf_hyperparameters(tmp_op, tf_weight_shape, tf_in_size)
                        tf_add_filters_ops[tmp_op] = add_with_action(tmp_op, tf_action_info, tf_weights_this,
                                                                     tf_bias_this, tf_weights_next,
                                                                     tf_running_activations,
                                                                     tf_wvelocity_this, tf_bvelocity_this,
                                                                     tf_wvelocity_next)
                        tf_rm_filters_ops[tmp_op] = remove_with_action(tmp_op, tf_action_info,
                                                                       tf_running_activations,
                                                                       tf_cnn_hyperparameters)
                        # tf_replace_ind_ops[tmp_op] = get_rm_indices_with_distance(tmp_op,tf_action_info)
                        tf_slice_optimize[tmp_op], tf_slice_vel_update[tmp_op] = optimize_masked_momentum_gradient(
                            optimizer, tf_indices,
                            tmp_op, tf_pool_avg_gradvars, tf_cnn_hyperparameters,
                            tf.constant(start_lr, dtype=tf.float32), global_step
                        )

                    elif 'fulcon' in tmp_op:
                        tf_update_hyp_ops[tmp_op] = update_tf_hyperparameters(tmp_op, tf_weight_shape, tf_in_size)


if __name__ == '__main__':
    global cnn_ops, cnn_hyperparameters

    try:
        opts, args = getopt.getopt(
            sys.argv[1:], "", ["output_dir=", "num_gpus=", "memory="])
    except getopt.GetoptError as err:
        print('<filename>.py --output_dir= --num_gpus= --memory=')

    if len(opts) != 0:
        for opt, arg in opts:
            if opt == '--output_dir':
                output_dir = arg
            if opt == '--num_gpus':
                num_gpus = int(arg)
            if opt == '--memory':
                mem_frac = float(arg)

    assert interval_parameters['test_interval'] % num_gpus == 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_info['image_w'] = image_size
    dataset_info['num_labels'] = num_labels
    dataset_info['num_channels'] = num_channels
    dataset_info['dataset_size'] = dataset_size
    dataset_info['chunk_size'] = chunk_size
    dataset_info['num_labels'] = num_labels

    logger, perf_logger, accuracy_drop_logger, cnn_structure_logger, \
    q_logger, class_dist_logger, pool_dist_logger = setup_loggers(research_parameters)

    # Loading test data
    train_dataset, train_labels = None, None

    test_dataset, test_labels = load_data_from_memmap(dataset_info, test_dataset_filename, test_label_filename, 0,
                                                      test_size, False)

    assert chunk_size % batch_size == 0
    batches_in_chunk = chunk_size // batch_size

    logger.info('=' * 80)
    logger.info('\tTrain Size: %d' % dataset_size)
    logger.info('=' * 80)

    cnn_ops, cnn_hyperparameters, final_2d_width = utils.get_ops_hyps_from_string(dataset_info, cnn_string)
    init_cnn_ops, init_cnn_hyperparameters, final_2d_width = utils.get_ops_hyps_from_string(dataset_info, cnn_string)

    print(cnn_ops)
    print(cnn_hyperparameters_getter)

    hyp_logger = logging.getLogger('hyperparameter_logger')
    hyp_logger.setLevel(logging.INFO)
    hypHandler = logging.FileHandler(output_dir + os.sep + 'Hyperparameter.log', mode='w')
    hypHandler.setFormatter(logging.Formatter('%(message)s'))
    hyp_logger.addHandler(hypHandler)
    hyp_logger.info('#Starting hyperparameters')
    hyp_logger.info(cnn_hyperparameters_getter)
    hyp_logger.info('#Dataset info')
    hyp_logger.info(dataset_info)
    hyp_logger.info('Learning rate: %.5f', start_lr)
    hyp_logger.info('Batch size: %d', batch_size)
    hyp_logger.info('#Research parameters')
    hyp_logger.info(research_parameters)
    hyp_logger.info('#Interval parameters')
    hyp_logger.info(interval_parameters)

    # Resetting this every policy interval
    rolling_data_distribution = {}
    mean_data_distribution = {}
    prev_mean_distribution = {}

    for li in range(num_labels):
        rolling_data_distribution[li] = 0.0
        mean_data_distribution[li] = 1.0 / float(num_labels)
        prev_mean_distribution[li] = 1.0 / float(num_labels)

    graph = tf.Graph()
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = False

    if mem_frac is not None:
        config.gpu_options.per_process_gpu_memory_fraction = mem_frac
    else:
        config.gpu_options.per_process_gpu_memory_fraction = 0.9

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        hardness = 0.5
        hard_pool = Pool(size=pool_size, batch_size=batch_size, image_size=image_size, num_channels=num_channels,
                         num_labels=num_labels, assert_test=False)

        first_fc = 'fulcon_out' if 'fulcon_0' not in cnn_ops else 'fulcon_0'
        # -1 is because we don't want to count pool_global
        print(cnn_ops)
        layer_count = len([op for op in cnn_ops if 'conv' in op or 'pool' in op]) - 1
        print(layer_count)
        # ids of the convolution ops
        convolution_op_ids = []
        for op_i, op in enumerate(cnn_ops):
            if 'conv' in op:
                convolution_op_ids.append(op_i)

        # Test data (Global)
        tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels),
                                         name='TestDataset')
        tf_test_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='TestLabels')

        # GLOBAL:
        with tf.variable_scope(TF_GLOBAL_SCOPE) as scope:
            global_step = tf.get_variable(initializer=0, dtype=tf.int32, trainable=False, name='global_step')
            tf_cnn_hyperparameters = init_tf_hyperparameters()
            _ = initialize_cnn_with_ops(cnn_ops, cnn_hyperparameters_getter)
            _ = define_velocity_vectors(scope)

        logger.debug('CNN_HYPERPARAMETERS 2')
        logger.debug('\t%s\n', tf_cnn_hyperparameters)

        session = tf.InteractiveSession(config=config)

        if research_parameters['adapt_structure']:
            # Adapting Policy Learner
            state_history_length = 4
            adapter = qlearner.AdaCNNAdaptingQLearner(
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
                rand_state_length=32, add_amount=add_amount, remove_amount=remove_amount,
                num_classes=num_labels, filter_min_threshold=filter_min_threshold,
                trial_phase_threshold=trial_phase_threshold
            )
            reward_queue = queue.Queue(maxsize=state_history_length - 1)

        # custom momentum optimizing we calculate momentum manually
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

        init_op = tf.global_variables_initializer()
        _ = session.run(init_op)



        init_op = tf.global_variables_initializer()
        _ = session.run(init_op)

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
                rolling_ativation_means[op] = np.zeros([cnn_hyperparameters_getter[op]['weights'][3]])
        dist_decay = 0.5
        act_decay = 0.9
        current_state, current_action = None, None
        next_valid_accuracy = 0, 0

        current_q_learn_op_id = 0
        logger.info('Convolutional Op IDs: %s', convolution_op_ids)

        logger.info('Starting Training Phase')
        # TODO: Think about decaying learning rate (should or shouldn't)

        logger.info('Dataset Size: %d', dataset_size)
        logger.info('Batch Size: %d', batch_size)
        logger.info('Chunk Size: %d', chunk_size)
        logger.info('Batches in Chunk : %d', batches_in_chunk)

        previous_loss = 1e5  # used for the check to start adapting
        prev_pool_accuracy = 0
        max_pool_accuracy = 0
        start_adapting = False
        stop_adapting = False
        batch_id_multiplier = (dataset_size // batch_size) - interval_parameters['test_interval']

        assert batches_in_chunk % num_gpus == 0
        assert num_gpus > 0

        for epoch in range(epochs):
            memmap_idx = 0

            for batch_id in range(0, ceil(dataset_size // batch_size) - num_gpus, num_gpus):

                if batch_id + 1 >= research_parameters['stop_training_at']:
                    break

                t0 = time.clock()  # starting time for a batch

                logger.debug('=' * 80)
                logger.debug('tf op count: %d', len(graph.get_operations()))
                logger.debug('=' * 80)

                logger.debug('\tTraining with batch %d', batch_id)

                chunk_batch_id = batch_id % batches_in_chunk

                if chunk_batch_id == 0:
                    # We load 1 extra batch (chunk_size+1) because we always make the valid batch the batch_id+1
                    logger.info('\tCurrent memmap start index: %d', memmap_idx)
                    if memmap_idx + chunk_size + batch_size < dataset_size:
                        train_dataset, train_labels = load_data_from_memmap(dataset_info, dataset_filename,
                                                                            label_filename, memmap_idx,
                                                                            chunk_size + batch_size, True)
                    else:
                        train_dataset, train_labels = load_data_from_memmap(dataset_info, dataset_filename,
                                                                            label_filename,
                                                                            memmap_idx, chunk_size, True)
                    memmap_idx += chunk_size
                    logger.info('Loading dataset chunk of size (chunk size + batch size): %d', train_dataset.shape[0])
                    logger.info('\tDataset shape: %s', train_dataset.shape)
                    logger.info('\tLabels shape: %s', train_labels.shape)
                    logger.info('\tNext memmap start index: %d', memmap_idx)

                # Feed dicitonary with placeholders for each tower
                batch_data, batch_labels, batch_weights = [], [], []
                train_feed_dict = {}
                for gpu_id in range(num_gpus):

                    batch_data.append(
                        train_dataset[(chunk_batch_id + gpu_id) * batch_size:(chunk_batch_id + gpu_id + 1) * batch_size,
                        :, :, :])
                    batch_labels.append(
                        train_labels[(chunk_batch_id + gpu_id) * batch_size:(chunk_batch_id + gpu_id + 1) * batch_size,
                        :])

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

                t0_train = time.clock()
                for _ in range(iterations_per_batch):
                    _, _, l, super_loss_vec, current_activations_list, train_predictions = session.run(
                        [apply_grads_op, update_train_velocity_op, mean_loss_op, concat_loss_vec_op, tf_mean_activation,
                         tower_predictions], feed_dict=train_feed_dict
                    )

                t1_train = time.clock()

                current_activations = get_activation_dictionary(current_activations_list, cnn_ops, convolution_op_ids)

                # this snippet logs the normalized class distribution every specified interval
                for bl in batch_labels:
                    if chunk_batch_id % research_parameters['log_distribution_every'] == 0:
                        dist_cnt = Counter(np.argmax(bl, axis=1))
                        norm_dist = ''
                        for li in range(num_labels):
                            norm_dist += '%.5f,' % (dist_cnt[li] * 1.0 / batch_size)
                        class_dist_logger.info(norm_dist)

                if np.isnan(l):
                    logger.critical('Diverged (NaN detected) (batchID) %d (last Cost) %.3f', chunk_batch_id,
                                    train_losses[-1])
                assert not np.isnan(l)

                # rolling activation mean update
                for op, op_activations in current_activations.items():
                    logger.debug('checking %s', op)
                    logger.debug('\tRolling size (%s): %s', op, rolling_ativation_means[op].shape)
                    logger.debug('\tCurrent size (%s): %s', op, op_activations.shape)
                for op, op_activations in current_activations.items():
                    assert current_activations[op].size == cnn_hyperparameters_getter[op]['weights'][3]
                    rolling_ativation_means[op] = act_decay * rolling_ativation_means[op] + current_activations[op]

                train_losses.append(l)

                # validation batch (Unseen)
                batch_valid_data = train_dataset[(chunk_batch_id + num_gpus) * batch_size:(
                                                                                          chunk_batch_id + num_gpus + 1) * batch_size,
                                   :, :, :]
                batch_valid_labels = train_labels[(chunk_batch_id + num_gpus) * batch_size:(
                                                                                           chunk_batch_id + num_gpus + 1) * batch_size,
                                     :]

                feed_valid_dict = {tf_valid_data_batch: batch_valid_data, tf_valid_label_batch: batch_valid_labels}
                next_valid_predictions = session.run(valid_predictions_op, feed_dict=feed_valid_dict)
                next_valid_accuracy = accuracy(next_valid_predictions, batch_valid_labels)

                # concat all the tower data to a single array
                single_iteration_batch_data, single_iteration_batch_labels = None, None
                for gpu_id in range(num_gpus):
                    if single_iteration_batch_data is None and single_iteration_batch_labels is None:
                        single_iteration_batch_data, single_iteration_batch_labels = batch_data[gpu_id], batch_labels[
                            gpu_id]
                    else:
                        single_iteration_batch_data = np.append(single_iteration_batch_data, batch_data[gpu_id], axis=0)
                        single_iteration_batch_labels = np.append(single_iteration_batch_labels, batch_labels[gpu_id],
                                                                  axis=0)

                if ((research_parameters['pooling_for_nonadapt'] and (
                not research_parameters['adapt_structure'])) or stop_adapting) and \
                        (batch_id > 0 and batch_id % interval_parameters['finetune_interval'] == 0):
                    logger.info('Pooling for non-adaptive CNN')
                    if research_parameters['adapt_structure']:
                        interval_parameters['finetune_interval'] = orig_finetune_interval
                        logger.info('Adaptations stopped. Finetune is at its maximum utility (Batch: %d)' % (
                        batch_id_multiplier * epoch + batch_id))

                    if hard_pool.get_size() > batch_size:
                        pool_dataset, pool_labels = hard_pool.get_pool_data(True)
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

                        for pool_id in range((hard_pool.get_size() // batch_size) // 2,
                                             (hard_pool.get_size() // batch_size) - 1, num_gpus):
                            if np.random.random() < research_parameters['finetune_rate']:
                                pool_feed_dict = {}
                                for gpu_id in range(num_gpus):
                                    pbatch_data = pool_dataset[
                                                  (pool_id + gpu_id) * batch_size:(pool_id + gpu_id + 1) * batch_size,
                                                  :, :, :]
                                    pbatch_labels = pool_labels[
                                                    (pool_id + gpu_id) * batch_size:(pool_id + gpu_id + 1) * batch_size,
                                                    :]
                                    pool_feed_dict.update({tf_pool_data_batch[gpu_id]: pbatch_data,
                                                           tf_pool_label_batch[gpu_id]: pbatch_labels})

                                _, _ = session.run([apply_pool_grads_op, update_pool_velocity_ops],
                                                   feed_dict=pool_feed_dict)

                        tmp_pool_accuracy = []
                        pool_dataset, pool_labels = hard_pool.get_pool_data(False)
                        for pool_id in range((hard_pool.get_size() // batch_size) // 2):
                            pbatch_data = pool_dataset[pool_id * batch_size:(pool_id + 1) * batch_size, :, :, :]
                            pbatch_labels = pool_labels[pool_id * batch_size:(pool_id + 1) * batch_size, :]
                            pool_feed_dict = {tf_pool_data_batch[0]: pbatch_data, tf_pool_label_batch[0]: pbatch_labels}

                            p_predictions = session.run(pool_pred, feed_dict=pool_feed_dict)
                            tmp_pool_accuracy.append(accuracy(p_predictions, pbatch_labels))

                        logger.info('\tPool accuracy non-adapt: %.5f', np.mean(tmp_pool_accuracy))

                if research_parameters['adapt_structure'] and np.random.random() < research_parameters[
                    'hard_pool_acceptance_rate']:
                    train_accuracy = np.mean(
                        [accuracy(train_predictions[gid], batch_labels[gid]) for gid in range(num_gpus)]) / 100.0
                    hard_pool.add_hard_examples(single_iteration_batch_data, single_iteration_batch_labels,
                                                super_loss_vec, min(research_parameters['hard_pool_max_threshold'],
                                                                    max(0.1, (1.0 - train_accuracy))))
                    # logger.debug('Adding %.3f examples',min(research_parameters['hard_pool_max_threshold'],max(0.1,(1.0-train_accuracy)))*2*batch_size)
                    # logger.debug('Accuracy: %.3f',train_accuracy)
                    logger.debug('Pooling data summary')
                    logger.debug('\tData batch size %d', single_iteration_batch_data.shape[0])
                    logger.debug('\tAccuracy %.3f', train_accuracy)
                    logger.debug('\tPool size (after): %d', hard_pool.get_size())
                    assert hard_pool.get_size() > 1
                if not research_parameters['adapt_structure'] and research_parameters['pooling_for_nonadapt']:
                    hard_pool.add_hard_examples(single_iteration_batch_data, single_iteration_batch_labels,
                                                super_loss_vec, 1.0)
                    logger.debug('\tPool size (after): %d', hard_pool.get_size())

                if batch_id % interval_parameters['test_interval'] == 0:
                    mean_train_loss = np.mean(train_losses)
                    logger.info('=' * 60)
                    logger.info('\tBatch ID: %d' % batch_id)
                    if decay_learning_rate:
                        logger.info('\tLearning rate: %.5f' % session.run(
                            tf.train.exponential_decay(start_lr, global_step, decay_steps=1, decay_rate=decay_rate,
                                                       staircase=True)))
                    else:
                        logger.info('\tLearning rate: %.5f' % start_lr)

                    logger.info('\tMinibatch Mean Loss: %.3f' % mean_train_loss)
                    logger.info('\tValidation Accuracy (Unseen): %.3f' % next_valid_accuracy)
                    logger.info('\tHard pool acceptance rate: %.2f', research_parameters['hard_pool_acceptance_rate'])
                    test_accuracies = []
                    for test_batch_id in range(test_size // batch_size):
                        batch_test_data = test_dataset[test_batch_id * batch_size:(test_batch_id + 1) * batch_size, :,
                                          :, :]
                        batch_test_labels = test_labels[test_batch_id * batch_size:(test_batch_id + 1) * batch_size, :]

                        feed_test_dict = {tf_test_dataset: batch_test_data, tf_test_labels: batch_test_labels}
                        test_predictions = session.run(test_predicitons_op, feed_dict=feed_test_dict)
                        test_accuracies.append(accuracy(test_predictions, batch_test_labels))
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

                    if research_parameters['adapt_structure'] and abs(current_test_accuracy - prev_test_accuracy) > 20:
                        accuracy_drop_logger.info('%s,%d', state_action_history,
                                                  current_test_accuracy - prev_test_accuracy)

                    prev_test_accuracy = current_test_accuracy
                    error_logger.info('%d,%.3f,%.3f,%.3f',
                                      batch_id_multiplier * epoch + batch_id, mean_train_loss,
                                      next_valid_accuracy, np.mean(test_accuracies)
                                      )
                    if research_parameters['adapt_structure'] and \
                            not start_adapting and \
                            (previous_loss - mean_train_loss < research_parameters['loss_diff_threshold'] and batch_id >
                                research_parameters['start_adapting_after']):
                        start_adapting = True
                        research_parameters['hard_pool_acceptance_rate'] /= 2.0
                        logger.info('=' * 80)
                        logger.info('Loss Stabilized: Starting structural adaptations...')
                        logger.info('Hardpool acceptance rate: %.2f', research_parameters['hard_pool_acceptance_rate'])
                        logger.info('=' * 80)

                    previous_loss = mean_train_loss

                    if research_parameters['save_train_test_images']:
                        local_dir = output_dir + os.sep + 'saved_images_' + str(batch_id)
                        if not os.path.exists(local_dir):
                            os.mkdir(local_dir)
                        train_dir = local_dir + os.sep + 'train'
                        if not os.path.exists(train_dir):
                            os.mkdir(train_dir)
                        for img_id in range(50):
                            img_lbl = np.asscalar(np.argmax(batch_labels[img_id, :]))
                            img_data = batch_data[img_id, :, :, :]
                            if not os.path.exists(train_dir + os.sep + str(img_lbl)):
                                os.mkdir(train_dir + os.sep + str(img_lbl))
                            imsave(train_dir + os.sep + str(img_lbl) + os.sep + 'image_' + str(img_id) + '.png',
                                   img_data)

                        test_dir = local_dir + os.sep + 'test'
                        if not os.path.exists(test_dir):
                            os.mkdir(test_dir)
                        for img_id in range(50):
                            img_lbl = np.asscalar(np.argmax(test_labels[img_id, :]))
                            img_data = test_dataset[img_id, :, :, :]
                            if not os.path.exists(test_dir + os.sep + str(img_lbl)):
                                os.mkdir(test_dir + os.sep + str(img_lbl))
                            imsave(test_dir + os.sep + str(img_lbl) + os.sep + 'image_' + str(img_id) + '.png',
                                   img_data)

                    # reset variables
                    mean_train_loss = 0.0
                    train_losses = []

                if research_parameters['adapt_structure']:

                    # calculate rolling mean of data distribution (amounts of different classes)
                    for li in range(num_labels):
                        rolling_data_distribution[li] = (1 - dist_decay) * rolling_data_distribution[
                            li] + dist_decay * float(cnt[li]) / float(batch_size)
                        mean_data_distribution[li] += float(cnt[li]) / (
                        float(batch_size) * float(interval_parameters['policy_interval']))

                    # ====================== Policy update and output ======================
                    if not start_adapting and batch_id > 0 and batch_id % (
                    interval_parameters['finetune_interval']) == 0:
                        logger.info(
                            'Finetuning before starting adaptations. (To gain a reasonable accuracy to start with)')

                        pool_dataset, pool_labels = hard_pool.get_pool_data(True)

                        # without if can give problems in exploratory stage because of no data in the pool

                        if hard_pool.get_size() > batch_size:
                            # Train with latter half of the data

                            for pool_id in range((hard_pool.get_size() // batch_size) // 2,
                                                 (hard_pool.get_size() // batch_size) - 1, num_gpus):
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

                    if (start_adapting and not stop_adapting) and batch_id > 0 and batch_id % interval_parameters[
                        'policy_interval'] == 0:

                        # update distance measure for class distirbution
                        # distMSE = 0.0
                        # for li in range(num_labels):
                        #    distMSE += (prev_mean_distribution[li]-rolling_data_distribution[li])**2
                        # distMSE = np.sqrt(distMSE)

                        filter_dict, filter_list = {}, []
                        for op_i, op in enumerate(cnn_ops):
                            if 'conv' in op:
                                filter_dict[op_i] = cnn_hyperparameters_getter[op]['weights'][3]
                                filter_list.append(cnn_hyperparameters_getter[op]['weights'][3])
                            elif 'pool' in op and op != 'pool_global':
                                filter_dict[op_i] = 0
                                filter_list.append(0)

                        current_state, current_action, curr_invalid_actions = adapter.output_action(
                            {'filter_counts': filter_dict, 'filter_counts_list': filter_list})
                        stop_adapting = adapter.get_stop_adapting_boolean()

                        adapter.update_trial_phase(
                            (batch_id_multiplier * epoch + batch_id) * 1.0 / (dataset_size // batch_size))

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

                                amount_to_add = ai[1]

                                if current_op != last_conv_id:
                                    next_conv_op = \
                                    [tmp_op for tmp_op in cnn_ops[cnn_ops.index(current_op) + 1:] if 'conv' in tmp_op][
                                        0]

                                _ = session.run(tf_add_filters_ops[current_op],
                                                feed_dict={
                                                    tf_action_info: np.asarray([li, 1, ai[1]]),
                                                    tf_weights_this: np.random.normal(scale=0.01, size=(
                                                        cnn_hyperparameters_getter[current_op]['weights'][0],
                                                        cnn_hyperparameters_getter[current_op]['weights'][1],
                                                        cnn_hyperparameters_getter[current_op]['weights'][2], amount_to_add)),
                                                    tf_bias_this: np.random.normal(scale=0.01, size=(amount_to_add)),

                                                    tf_weights_next: np.random.normal(scale=0.01, size=(
                                                        cnn_hyperparameters_getter[next_conv_op]['weights'][0],
                                                        cnn_hyperparameters_getter[next_conv_op]['weights'][1],
                                                        amount_to_add, cnn_hyperparameters_getter[next_conv_op]['weights'][3])
                                                                                      ) if last_conv_id != current_op else
                                                    np.random.normal(scale=0.01, size=(
                                                        amount_to_add * final_2d_width * final_2d_width,
                                                        cnn_hyperparameters_getter[first_fc]['out'], 1, 1)),
                                                    tf_running_activations: rolling_ativation_means[current_op],

                                                    tf_wvelocity_this: np.zeros(shape=(
                                                        cnn_hyperparameters_getter[current_op]['weights'][0],
                                                        cnn_hyperparameters_getter[current_op]['weights'][1],
                                                        cnn_hyperparameters_getter[current_op]['weights'][2], amount_to_add),
                                                        dtype=np.float32),
                                                    tf_bvelocity_this: np.zeros(shape=(amount_to_add,),
                                                                                dtype=np.float32),
                                                    tf_wvelocity_next: np.zeros(shape=(
                                                        cnn_hyperparameters_getter[next_conv_op]['weights'][0],
                                                        cnn_hyperparameters_getter[next_conv_op]['weights'][1],
                                                        amount_to_add, cnn_hyperparameters_getter[next_conv_op]['weights'][3]),
                                                        dtype=np.float32) if last_conv_id != current_op else
                                                    np.zeros(shape=(final_2d_width * final_2d_width * amount_to_add,
                                                                    cnn_hyperparameters_getter[first_fc]['out'], 1, 1),
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
                                cnn_hyperparameters_getter[current_op]['weights'][3] += amount_to_add
                                if research_parameters['debugging']:
                                    assert cnn_hyperparameters_getter[current_op]['weights'][2] == \
                                           tf.shape(current_op_weights).eval()[2]

                                session.run(tf_update_hyp_ops[current_op], feed_dict={
                                    tf_weight_shape: cnn_hyperparameters_getter[current_op]['weights']
                                })

                                if current_op == last_conv_id:

                                    with tf.variable_scope(TF_GLOBAL_SCOPE + TF_SCOPE_DIVIDER + first_fc, reuse=True):
                                        first_fc_weights = tf.get_variable(TF_WEIGHTS)
                                    cnn_hyperparameters_getter[first_fc][
                                        'in'] += final_2d_width * final_2d_width * amount_to_add

                                    if research_parameters['debugging']:
                                        logger.debug('\tNew %s in: %d', first_fc, cnn_hyperparameters_getter[first_fc]['in'])
                                        logger.debug('\tSummary of changes to weights of %s', first_fc)
                                        logger.debug('\t\tNew Weights: %s', str(tf.shape(first_fc_weights).eval()))

                                    session.run(tf_update_hyp_ops[first_fc], feed_dict={
                                        tf_in_size: cnn_hyperparameters_getter[first_fc]['in']
                                    })

                                else:

                                    next_conv_op = \
                                        [tmp_op for tmp_op in cnn_ops[cnn_ops.index(current_op) + 1:] if
                                         'conv' in tmp_op][0]
                                    assert current_op != next_conv_op

                                    with tf.variable_scope(TF_GLOBAL_SCOPE + TF_SCOPE_DIVIDER + next_conv_op,
                                                           reuse=True):
                                        next_conv_op_weights = tf.get_variable(TF_WEIGHTS)

                                    if research_parameters['debugging']:
                                        logger.debug('\tSummary of changes to weights of %s', next_conv_op)
                                        logger.debug('\t\tCurrent Weights: %s',
                                                     str(tf.shape(next_conv_op_weights).eval()))

                                    cnn_hyperparameters_getter[next_conv_op]['weights'][2] += amount_to_add

                                    if research_parameters['debugging']:
                                        assert cnn_hyperparameters_getter[next_conv_op]['weights'][2] == \
                                               tf.shape(next_conv_op_weights).eval()[2]

                                    session.run(tf_update_hyp_ops[next_conv_op], feed_dict={
                                        tf_weight_shape: cnn_hyperparameters_getter[next_conv_op]['weights']
                                    })

                                # optimize the newly added fiterls only
                                pool_dataset, pool_labels = hard_pool.get_pool_data(True)
                                if research_parameters['pool_randomize'] and np.random.random() < research_parameters[
                                    'pool_randomize_rate']:
                                    try:
                                        pool = MPPool(processes=10)
                                        distorted_imgs = pool.map(distort_img, pool_dataset)
                                        pool_dataset = np.asarray(distorted_imgs)
                                        pool.close()
                                        pool.join()
                                    except Exception:
                                        raise AssertionError

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
                                # Unless you run this onces, the sizes of weights do not change
                                _ = session.run([tower_logits, tower_activation_update_ops], feed_dict=train_feed_dict)
                                pbatch_train_count = 0

                                # Train only with half of the batch
                                for pool_id in range((hard_pool.get_size() // batch_size) // 2,
                                                     (hard_pool.get_size() // batch_size) - 1, num_gpus):
                                    if np.random.random() < research_parameters['replace_op_train_rate']:
                                        pbatch_data, pbatch_labels = [], []
                                        pool_feed_dict = {
                                            tf_indices: np.arange(cnn_hyperparameters_getter[current_op]['weights'][3] - ai[1],
                                                                  cnn_hyperparameters_getter[current_op]['weights'][3])}

                                        for gpu_id in range(num_gpus):
                                            pbatch_data.append(pool_dataset[(pool_id + gpu_id) * batch_size:(
                                                                                                            pool_id + gpu_id + 1) * batch_size,
                                                               :, :, :])
                                            pbatch_labels.append(pool_labels[(pool_id + gpu_id) * batch_size:(
                                                                                                             pool_id + gpu_id + 1) * batch_size,
                                                                 :])
                                            pool_feed_dict.update({tf_pool_data_batch[gpu_id]: pbatch_data[-1],
                                                                   tf_pool_label_batch[gpu_id]: pbatch_labels[-1]})

                                        pbatch_train_count += 1

                                        current_activations_list, _, _ = session.run(
                                            [tf_mean_pool_activations, tf_slice_optimize[current_op],
                                             tf_slice_vel_update[current_op]],
                                            feed_dict=pool_feed_dict
                                        )

                                        current_activations = get_activation_dictionary(current_activations_list,
                                                                                        cnn_ops, convolution_op_ids)
                                        # update rolling activation means
                                        for op, op_activations in current_activations.items():
                                            assert current_activations[op].size == cnn_hyperparameters_getter[op]['weights'][3]
                                            rolling_ativation_means[op] = act_decay * rolling_ativation_means[op] + \
                                                                          current_activations[op]

                                if hard_pool.get_size() > batch_size:
                                    pool_dataset, pool_labels = hard_pool.get_pool_data(True)
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
                                    for pool_id in range((hard_pool.get_size() // batch_size) // 2,
                                                         (hard_pool.get_size() // batch_size) - 1, num_gpus):
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

                            elif 'conv' in current_op and ai[0] == 'remove':

                                _, rm_indices = session.run(tf_rm_filters_ops[current_op],
                                                            feed_dict={
                                                                tf_action_info: np.asarray([li, 0, ai[1]]),
                                                                tf_running_activations: rolling_ativation_means[
                                                                    current_op]
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
                                                 cnn_hyperparameters_getter[current_op]['weights'][3])
                                    indices_of_filters_keep = list(
                                        set(np.arange(cnn_hyperparameters_getter[current_op]['weights'][3])) - set(
                                            rm_indices.tolist()))
                                    logger.debug('\t\tSize of indices to keep: %s/%d', len(indices_of_filters_keep),
                                                 cnn_hyperparameters_getter[current_op]['weights'][3])

                                cnn_hyperparameters_getter[current_op]['weights'][3] -= amount_to_rmv
                                if research_parameters['debugging']:
                                    logger.debug('\tSize after feature map reduction: %s,%s', current_op,
                                                 tf.shape(current_op_weights).eval())
                                    assert tf.shape(current_op_weights).eval()[3] == \
                                           cnn_hyperparameters_getter[current_op]['weights'][3]

                                session.run(tf_update_hyp_ops[current_op], feed_dict={
                                    tf_weight_shape: cnn_hyperparameters_getter[current_op]['weights']
                                })

                                if current_op == last_conv_id:
                                    with tf.variable_scope(TF_GLOBAL_SCOPE + TF_SCOPE_DIVIDER + first_fc, reuse=True):
                                        first_fc_weights = tf.get_variable(TF_WEIGHTS)

                                    cnn_hyperparameters_getter[first_fc][
                                        'in'] -= final_2d_width * final_2d_width * amount_to_rmv
                                    if research_parameters['debugging']:
                                        logger.debug('\tSize after feature map reduction: %s,%s',
                                                     first_fc, str(tf.shape(first_fc_weights).eval()))

                                    session.run(tf_update_hyp_ops[first_fc], feed_dict={
                                        tf_in_size: cnn_hyperparameters_getter[first_fc]['in']
                                    })

                                else:
                                    next_conv_op = \
                                    [tmp_op for tmp_op in cnn_ops[cnn_ops.index(current_op) + 1:] if 'conv' in tmp_op][
                                        0]
                                    assert current_op != next_conv_op

                                    with tf.variable_scope(TF_GLOBAL_SCOPE + TF_SCOPE_DIVIDER + next_conv_op,
                                                           reuse=True):
                                        next_conv_op_weights = tf.get_variable(TF_WEIGHTS)

                                    cnn_hyperparameters_getter[next_conv_op]['weights'][2] -= amount_to_rmv

                                    if research_parameters['debugging']:
                                        logger.debug('\tSize after feature map reduction: %s,%s', next_conv_op,
                                                     str(tf.shape(next_conv_op_weights).eval()))
                                        assert tf.shape(next_conv_op_weights).eval()[2] == \
                                               cnn_hyperparameters_getter[next_conv_op]['weights'][2]

                                    session.run(tf_update_hyp_ops[next_conv_op], feed_dict={
                                        tf_weight_shape: cnn_hyperparameters_getter[next_conv_op]['weights']
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

                                if hard_pool.get_size() > batch_size:
                                    pool_dataset, pool_labels = hard_pool.get_pool_data(True)
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
                                    for pool_id in range((hard_pool.get_size() // batch_size) // 2,
                                                         (hard_pool.get_size() // batch_size) - 1, num_gpus):
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

                            elif 'conv' in current_op and ai[0] == 'finetune':
                                # pooling takes place here

                                op = cnn_ops[li]
                                pool_dataset, pool_labels = hard_pool.get_pool_data(True)

                                if research_parameters['pool_randomize'] and np.random.random() < research_parameters[
                                    'pool_randomize_rate']:
                                    try:
                                        pool = MPPool(processes=10)
                                        distorted_imgs = pool.map(distort_img, pool_dataset)
                                        pool_dataset = np.asarray(distorted_imgs)
                                        pool.close()
                                        pool.join()
                                    except Exception:
                                        raise AssertionError

                                # without if can give problems in exploratory stage because of no data in the pool
                                if hard_pool.get_size() > batch_size:
                                    # Train with latter half of the data

                                    for pool_id in range((hard_pool.get_size() // batch_size) // 2,
                                                         (hard_pool.get_size() // batch_size) - 1, num_gpus):
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

                                break  # action include finetune actions for number of conv layers there are

                        assert hard_pool.get_size() > 0
                        # ======================================
                        # Policy Update
                        # ======================================
                        pool_accuracy = []
                        pool_dataset, pool_labels = hard_pool.get_pool_data(False)

                        for pool_id in range((hard_pool.get_size() // batch_size) // 2):
                            pbatch_data = pool_dataset[pool_id * batch_size:(pool_id + 1) * batch_size, :, :, :]
                            pbatch_labels = pool_labels[pool_id * batch_size:(pool_id + 1) * batch_size, :]
                            pool_feed_dict = {tf_pool_data_batch[0]: pbatch_data, tf_pool_label_batch[0]: pbatch_labels}
                            p_predictions = session.run(pool_pred, feed_dict=pool_feed_dict)
                            pool_accuracy.append(accuracy(p_predictions, pbatch_labels))

                        # don't use current state as the next state, current state is for a different layer
                        next_state = []
                        for li, la in enumerate(current_action):
                            if la is None:
                                assert li not in convolution_op_ids
                                next_state.append(0)
                                continue
                            elif la[0] == 'add':
                                next_state.append(current_state[li] + la[1])
                            elif la[0] == 'remove':
                                next_state.append(current_state[li] - la[1])
                            else:
                                next_state.append(current_state[li])

                        next_state = tuple(next_state)

                        logger.info('\tState (prev): %s', str(current_state))
                        logger.info('\tAction (prev): %s', str(current_action))
                        logger.info('\tState (next): %s\n', str(next_state))
                        p_accuracy = np.mean(pool_accuracy) if len(pool_accuracy) > 2 else 0
                        logger.info('\tPool Accuracy: %.3f\n', p_accuracy)
                        logger.info('\tPrev pool Accuracy: %.3f\n', prev_pool_accuracy)
                        assert not np.isnan(p_accuracy)
                        adapter.update_policy({'prev_state': current_state, 'prev_action': current_action,
                                               'curr_state': next_state,
                                               'next_accuracy': None,
                                               'prev_accuracy': None,
                                               'pool_accuracy': p_accuracy,
                                               'prev_pool_accuracy': prev_pool_accuracy,
                                               'max_pool_accuracy': max_pool_accuracy,
                                               'invalid_actions': curr_invalid_actions,
                                               'batch_id': (batch_id_multiplier * epoch) + batch_id}, True)

                        if len(state_action_history) > 10:
                            del state_action_history[0]

                        state_action_history.append([current_state, current_action, p_accuracy, prev_pool_accuracy])

                        cnn_structure_logger.info(
                            '%d:%s:%s:%.5f:%s', (batch_id_multiplier * epoch) + batch_id, current_state, current_action,
                            np.mean(pool_accuracy),
                            utils.get_cnn_string_from_ops(cnn_ops, cnn_hyperparameters_getter)
                        )

                        q_logger.info('%d,%.5f', epoch * batch_id_multiplier + batch_id, adapter.get_average_Q())

                        logger.debug('Resetting both data distribution means')

                        prev_mean_distribution = mean_data_distribution
                        for li in range(num_labels):
                            rolling_data_distribution[li] = 0.0
                            mean_data_distribution[li] = 0.0

                        max_pool_accuracy = max(max_pool_accuracy, p_accuracy)
                        prev_pool_accuracy = p_accuracy

                    if batch_id > 0 and batch_id % interval_parameters['history_dump_interval'] == 0:

                        # with open(output_dir + os.sep + 'Q_' + str(epoch) + "_" + str(batch_id)+'.pickle', 'wb') as f:
                        #    pickle.dump(adapter.get_Q(), f, pickle.HIGHEST_PROTOCOL)

                        pool_dist_string = ''
                        for val in hard_pool.get_class_distribution():
                            pool_dist_string += str(val) + ','

                        pool_dist_logger.info('%s%d', pool_dist_string, hard_pool.get_size())

                t1 = time.clock()
                op_count = len(graph.get_operations())
                var_count = len(tf.global_variables()) + len(tf.local_variables()) + len(tf.model_variables())
                perf_logger.info('%d,%.5f,%.5f,%d,%d', (batch_id_multiplier * epoch) + batch_id, t1 - t0,
                                 (t1_train - t0_train) / num_gpus, op_count, var_count)

            if decay_learning_rate and epoch >= 1:
                session.run(inc_global_step(global_step))

            # reset the network
            if research_parameters['adapt_structure'] and epoch == 0:
                adapter.update_trial_phase(1.0)
                hard_pool.reset_pool()
                session.run(tf_reset_cnn)

                cnn_hyperparameters = copy.deepcopy(init_cnn_hyperparameters)

                for op in cnn_ops:
                    if 'conv' in op:
                        session.run(tf_update_hyp_ops[op],
                                    feed_dict={tf_weight_shape: init_cnn_hyperparameters[op]['weights']})
                        rolling_ativation_means[op] = np.zeros([init_cnn_hyperparameters[op]['weights'][3]])
                    elif 'fulcon' in op:
                        session.run(tf_update_hyp_ops[op], feed_dict={tf_in_size: init_cnn_hyperparameters[op]['in']})
                print(session.run(tf_cnn_hyperparameters))
                _ = session.run(tower_logits, feed_dict=train_feed_dict)
                prev_pool_accuracy = 0
                max_pool_accuracy = 0

                start_adapting = False
                research_parameters['hard_pool_acceptance_rate'] = 0.2

            if research_parameters['adapt_structure'] and epoch == 1:
                stop_adapting = True

            # Inc Algorithm
            if research_parameters['adapt_structure']:
                if epoch > 2:
                    session.run(increment_global_step_op)
            else:
                # Noninc pool algorithm
                if research_parameters['pooling_for_nonadapt']:
                    session.run(increment_global_step_op)
                # Noninc algorithm
                else:
                    session.run(increment_global_step_op)
