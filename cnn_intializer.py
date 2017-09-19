import tensorflow as tf
import constants
import logging
import numpy as np

TF_WEIGHTS = constants.TF_WEIGHTS
TF_BIAS = constants.TF_BIAS
TF_TRAIN_MOMENTUM = constants.TF_TRAIN_MOMENTUM
TF_POOL_MOMENTUM = constants.TF_POOL_MOMENTUM
TF_GLOBAL_SCOPE = constants.TF_GLOBAL_SCOPE
TF_CONV_WEIGHT_SHAPE_STR = constants.TF_CONV_WEIGHT_SHAPE_STR
TF_FC_WEIGHT_IN_STR = constants.TF_FC_WEIGHT_IN_STR
TF_FC_WEIGHT_OUT_STR = constants.TF_FC_WEIGHT_OUT_STR
TF_ACTIVAIONS_STR = constants.TF_ACTIVAIONS_STR
TF_SCOPE_DIVIDER = constants.TF_SCOPE_DIVIDER

logger = logging.getLogger('cnn_initializer_logger')
logger.setLevel(logging_level)
console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter(logging_format))
console.setLevel(logging_level)
logger.addHandler(console)

research_parameters = None
model_parameters = None


def set_from_main(research_params, model_params):
    global research_parameters, model_parameters
    research_parameters = research_params
    model_parameters = model_params


def initialize_cnn_with_ops(cnn_ops, cnn_hyps):
    '''
    Initialize the variables of the CNN (convolution filters fully-connected filters
    :param cnn_ops:
    :param cnn_hyps:
    :return:
    '''

    logger.info('CNN Hyperparameters')
    logger.info('%s\n', cnn_hyps)

    logger.info('Initializing the iConvNet (conv_global,pool_global,classifier)...\n')
    for op in cnn_ops:

        if 'conv' in op:
            with tf.variable_scope(op) as scope:
                tf.get_variable(
                    name=TF_WEIGHTS, shape=cnn_hyps[op]['weights'],
                    initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    validate_shape=False, dtype=tf.float32)
                tf.get_variable(
                    name=TF_BIAS,
                    initializer=tf.constant(np.random.random() * 0.001, shape=[cnn_hyps[op]['weights'][3]]),
                    validate_shape=False, dtype=tf.float32)

                tf.get_variable(
                    name=TF_ACTIVAIONS_STR,
                    initializer=tf.zeros(shape=[cnn_hyps[op]['weights'][3]],
                                         dtype=tf.float32, name=scope.name + '_activations'),
                    validate_shape=False, trainable=False)

                logger.debug('Weights for %s initialized with size %s', op, str(cnn_hyps[op]['weights']))
                logger.debug('Biases for %s initialized with size %d', op, cnn_hyps[op]['weights'][3])

        if 'fulcon' in op:
            with tf.variable_scope(op) as scope:
                tf.get_variable(
                    name=TF_WEIGHTS, shape=[cnn_hyps[op]['in'], cnn_hyps[op]['out']],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    validate_shape=False, dtype=tf.float32)
                tf.get_variable(
                    name=TF_BIAS,
                    initializer=tf.constant(np.random.random() * 0.001, shape=[cnn_hyps[op]['out']]),
                    validate_shape=False, dtype=tf.float32)

                logger.debug('Weights for %s initialized with size %d,%d',
                             op, cnn_hyps[op]['in'], cnn_hyps[op]['out'])
                logger.debug('Biases for %s initialized with size %d', op, cnn_hyps[op]['out'])



def define_velocity_vectors(main_scope):
    # if using momentum
    vel_var_list = []
    if research_parameters['optimizer'] == 'Momentum':
        for tmp_op in cnn_ops:
            op_scope = tmp_op
            if 'conv' in tmp_op:
                for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope=main_scope.name + TF_SCOPE_DIVIDER + op_scope):
                    if TF_WEIGHTS in v.name:
                        with tf.variable_scope(op_scope + TF_SCOPE_DIVIDER + TF_WEIGHTS) as scope:
                            vel_var_list.append(
                                tf.get_variable(name=TF_TRAIN_MOMENTUM,
                                                initializer=tf.zeros(shape=cnn_hyperparameters[tmp_op]['weights'],
                                                                     dtype=tf.float32),
                                                dtype=tf.float32, trainable=False))
                            vel_var_list.append(
                                tf.get_variable(name=TF_POOL_MOMENTUM,
                                                initializer=tf.zeros(shape=cnn_hyperparameters[tmp_op]['weights'],
                                                                     dtype=tf.float32),
                                                dtype=tf.float32, trainable=False))
                    elif TF_BIAS in v.name:
                        with tf.variable_scope(op_scope + TF_SCOPE_DIVIDER + TF_BIAS) as scope:
                            vel_var_list.append(tf.get_variable(name=TF_TRAIN_MOMENTUM,
                                                                initializer=tf.zeros(
                                                                    shape=cnn_hyperparameters[tmp_op]['weights'][3],
                                                                    dtype=tf.float32),
                                                                dtype=tf.float32, trainable=False))

                            vel_var_list.append(tf.get_variable(name=TF_POOL_MOMENTUM,
                                                                initializer=tf.zeros(
                                                                    shape=[cnn_hyperparameters[tmp_op]['weights'][3]],
                                                                    dtype=tf.float32),
                                                                dtype=tf.float32, trainable=False))

            elif 'fulcon' in tmp_op:
                for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope=main_scope.name + TF_SCOPE_DIVIDER + op_scope):
                    if TF_WEIGHTS in v.name:
                        with tf.variable_scope(op_scope + TF_SCOPE_DIVIDER + TF_WEIGHTS) as scope:
                            vel_var_list.append(tf.get_variable(name=TF_TRAIN_MOMENTUM,
                                                                initializer=tf.zeros(
                                                                    shape=[cnn_hyperparameters[tmp_op]['in'],
                                                                           cnn_hyperparameters[tmp_op]['out']],
                                                                    dtype=tf.float32),
                                                                dtype=tf.float32, trainable=False))
                            vel_var_list.append(tf.get_variable(name=TF_POOL_MOMENTUM,
                                                                initializer=tf.zeros(
                                                                    shape=[cnn_hyperparameters[tmp_op]['in'],
                                                                           cnn_hyperparameters[tmp_op]['out']],
                                                                    dtype=tf.float32),
                                                                dtype=tf.float32, trainable=False))
                    elif TF_BIAS in v.name:
                        with tf.variable_scope(op_scope + TF_SCOPE_DIVIDER + TF_BIAS) as scope:
                            vel_var_list.append(tf.get_variable(name=TF_TRAIN_MOMENTUM,
                                                                initializer=tf.zeros(
                                                                    shape=[cnn_hyperparameters[tmp_op]['out']],
                                                                    dtype=tf.float32),
                                                                dtype=tf.float32, trainable=False))

                            vel_var_list.append(tf.get_variable(name=TF_POOL_MOMENTUM,
                                                                initializer=tf.zeros(
                                                                    shape=[cnn_hyperparameters[tmp_op]['out']],
                                                                    dtype=tf.float32),
                                                                dtype=tf.float32, trainable=False))

    return vel_var_list



def reset_cnn(cnn_hyps):
    reset_ops = []
    logger.info('CNN Hyperparameters')
    logger.info('%s\n', cnn_hyps)

    logger.info('Initializing the iConvNet (conv_global,pool_global,classifier)...\n')
    for op in cnn_ops:

        if 'conv' in op:
            with tf.variable_scope(op) as scope:
                weights = tf.get_variable(name=TF_WEIGHTS)
                new_weights = tf.random_uniform(cnn_hyps[op]['weights'],
                                                minval=-np.sqrt(
                                                    6. / (cnn_hyps[op]['weights'][0] * cnn_hyps[op]['weights'][1] *
                                                          (cnn_hyps[op]['weights'][-2] + cnn_hyps[op]['weights'][-1]))
                                                    ),
                                                maxval=np.sqrt(
                                                    6. / (cnn_hyps[op]['weights'][0] * cnn_hyps[op]['weights'][1] *
                                                          (cnn_hyps[op]['weights'][-2] + cnn_hyps[op]['weights'][-1]))
                                                    )
                                                )

                reset_ops.append(tf.assign(weights, new_weights, validate_shape=False))

                with tf.variable_scope(TF_WEIGHTS) as child_scope:
                    w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
                    pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)

                    new_w_vel = tf.zeros(shape=cnn_hyps[op]['weights'], dtype=tf.float32)
                    reset_ops.append(tf.assign(w_vel, new_w_vel, validate_shape=False))
                    reset_ops.append(tf.assign(pool_w_vel, new_w_vel, validate_shape=False))

                bias = tf.get_variable(name=TF_BIAS)
                new_bias = tf.constant(np.random.random() * 0.001, shape=[cnn_hyps[op]['weights'][3]])

                reset_ops.append(tf.assign(bias, new_bias, validate_shape=False))

                with tf.variable_scope(TF_BIAS) as child_scope:
                    b_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
                    pool_b_vel = tf.get_variable(TF_POOL_MOMENTUM)

                    new_b_vel = tf.zeros(shape=[cnn_hyps[op]['weights'][3]], dtype=tf.float32)
                    reset_ops.append(tf.assign(b_vel, new_b_vel, validate_shape=False))
                    reset_ops.append(tf.assign(pool_b_vel, new_b_vel, validate_shape=False))

                act_var = tf.get_variable(name=TF_ACTIVAIONS_STR)
                new_act_var = tf.zeros(shape=[cnn_hyps[op]['weights'][3]],
                                       dtype=tf.float32)
                reset_ops.append(tf.assign(act_var, new_act_var, validate_shape=False))

        if 'fulcon' in op:
            with tf.variable_scope(op) as scope:
                weights = tf.get_variable(name=TF_WEIGHTS)
                new_weights = tf.random_uniform([cnn_hyps[op]['in'], cnn_hyps[op]['out']],
                                                minval=-np.sqrt(6. / (cnn_hyps[op]['in'] + cnn_hyps[op]['out'])),
                                                maxval=np.sqrt(6. / (cnn_hyps[op]['in'] + cnn_hyps[op]['out']))
                                                )
                reset_ops.append(tf.assign(weights, new_weights, validate_shape=False))

                with tf.variable_scope(TF_WEIGHTS) as child_scope:
                    w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
                    pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)

                    new_w_vel = tf.zeros(shape=[cnn_hyps[op]['in'], cnn_hyps[op]['out']], dtype=tf.float32)
                    reset_ops.append(tf.assign(w_vel, new_w_vel, validate_shape=False))
                    reset_ops.append(tf.assign(pool_w_vel, new_w_vel, validate_shape=False))

                bias = tf.get_variable(name=TF_BIAS)
                new_bias = tf.constant(np.random.random() * 0.001, shape=[cnn_hyps[op]['out']])
                reset_ops.append(tf.assign(bias, new_bias, validate_shape=False))

                with tf.variable_scope(TF_BIAS) as child_scope:
                    b_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
                    pool_b_vel = tf.get_variable(TF_POOL_MOMENTUM)

                    new_b_vel = tf.zeros(shape=[cnn_hyps[op]['out']], dtype=tf.float32)
                    reset_ops.append(tf.assign(b_vel, new_b_vel, validate_shape=False))
                    reset_ops.append(tf.assign(pool_b_vel, new_b_vel, validate_shape=False))

    return reset_ops



def init_tf_hyperparameters():
    global cnn_ops, cnn_hyperparameters
    tf_hyp_list = {}
    for op in cnn_ops:
        if 'conv' in op:
            with tf.variable_scope(op) as scope:
                tf_hyp_list[op] = {TF_CONV_WEIGHT_SHAPE_STR:
                                       tf.get_variable(name=TF_CONV_WEIGHT_SHAPE_STR,
                                                       initializer=cnn_hyperparameters[op]['weights'],
                                                       dtype=tf.int32, trainable=False)
                                   }

        if 'fulcon' in op:
            with tf.variable_scope(op) as scope:
                tf_hyp_list[op] = {TF_FC_WEIGHT_IN_STR: tf.get_variable(name=TF_FC_WEIGHT_IN_STR,
                                                                        initializer=cnn_hyperparameters[op]['in'],
                                                                        dtype=tf.int32, trainable=False)
                    , TF_FC_WEIGHT_OUT_STR: tf.get_variable(name=TF_FC_WEIGHT_OUT_STR,
                                                            initializer=cnn_hyperparameters[op]['out'],
                                                            dtype=tf.int32, trainable=False)
                                   }
    return tf_hyp_list
