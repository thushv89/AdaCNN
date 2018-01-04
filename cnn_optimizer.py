import tensorflow as tf
import constants
import logging
import sys

TF_WEIGHTS = constants.TF_WEIGHTS
TF_BIAS = constants.TF_BIAS
TF_TRAIN_MOMENTUM = constants.TF_TRAIN_MOMENTUM
TF_POOL_MOMENTUM = constants.TF_POOL_MOMENTUM
TF_GLOBAL_SCOPE = constants.TF_GLOBAL_SCOPE
TF_CONV_WEIGHT_SHAPE_STR = constants.TF_CONV_WEIGHT_SHAPE_STR
TF_FC_WEIGHT_IN_STR = constants.TF_FC_WEIGHT_IN_STR
TF_FC_WEIGHT_OUT_STR = constants.TF_FC_WEIGHT_OUT_STR
TF_SCOPE_DIVIDER = constants.TF_SCOPE_DIVIDER

research_parameters = None
model_parameters = None
final_2d_width = None
add_amout, add_fulcon_amount = None,None
logging_level, logging_format = None, None
logger = None
rms_epsilon = 1e-5

cnn_ops = None
def set_from_main(research_params, model_params, logging_level, logging_format, ops, final_2d_w):
    global research_parameters, model_parameters,logger, cnn_ops, add_amout, add_fulcon_amount, final_2d_width
    research_parameters = research_params
    model_parameters = model_params
    if model_parameters['adapt_structure']:
        add_amout = model_parameters['add_amount']
        add_fulcon_amount = model_parameters['add_fulcon_amount']
        final_2d_width = final_2d_w

    cnn_ops = ops

    logger = logging.getLogger('cnn_optimizer_logger')
    logger.setLevel(logging_level)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(logging_format))
    console.setLevel(logging_level)
    logger.addHandler(console)

def update_hyperparameters(research_hyp):
    global research_parameters
    research_parameters = research_hyp


def gradients(optimizer, loss, global_step, learning_rate):
    # grad_and_vars [(grads_w,w),(grads_b,b)]
    grad_and_vars = optimizer.compute_gradients(loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                        scope=TF_GLOBAL_SCOPE))
    return grad_and_vars

def apply_gradient_with_rmsprop(optimizer, learning_rate, global_step, grads_and_vars_vanilla):
    global cnn_ops

    grads_and_vars,vel_update_ops = [],[]
    # for each trainable variable
    if model_parameters['decay_learning_rate']:
        learning_rate = tf.maximum(
            model_parameters['min_learning_rate'],
            tf.train.exponential_decay(
                learning_rate, global_step, decay_steps=1,
                decay_rate=model_parameters['decay_rate'],
                staircase=True)
        )

    # update velocity vector
    for lyr_i, op in enumerate(cnn_ops):
        if 'pool' in op:
            continue

        for (g,v) in grads_and_vars_vanilla:

            with tf.variable_scope(op, reuse=True):
                w,b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)

                with tf.variable_scope('age', reuse=True):
                    age_w, age_b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)

                if v.name == w.name:

                    with tf.variable_scope(TF_WEIGHTS, reuse=True):

                        vel = tf.get_variable(TF_TRAIN_MOMENTUM)
                        vel_update_ops.append(
                            tf.assign(vel,
                                      research_parameters['momentum'] * vel +
                                      (1.0 - research_parameters['momentum']) * g ** 2)
                        )

                        adaptive_w_lr = learning_rate / (tf.sqrt(vel + rms_epsilon)*age_w)

                        grads_and_vars.append((g * adaptive_w_lr, w))

                if v.name == b.name:

                    with tf.variable_scope(TF_BIAS, reuse=True):

                        vel = tf.get_variable(TF_TRAIN_MOMENTUM)
                        vel_update_ops.append(
                            tf.assign(vel,
                                      research_parameters['momentum'] * vel +
                                      (1.0 - research_parameters['momentum']) * g ** 2)
                        )

                        adaptive_b_lr = learning_rate / (tf.sqrt(vel + rms_epsilon)*age_b)
                        grads_and_vars.append((g * adaptive_b_lr, b))

    return optimizer.apply_gradients(grads_and_vars), vel_update_ops

def apply_pool_gradient_with_rmsprop(optimizer, learning_rate, global_step, grads_and_vars_vanilla):
    global cnn_ops

    grads_and_vars,vel_update_ops = [],[]
    # for each trainable variable
    if model_parameters['decay_learning_rate']:
        learning_rate = tf.maximum(model_parameters['min_learning_rate'],
                                   tf.train.exponential_decay(learning_rate, global_step, decay_steps=1,
                                                              decay_rate=model_parameters['decay_rate'],
                                                              staircase=True))

    # update velocity vector
    for lyr_i, op in enumerate(cnn_ops):
        if 'pool' in op:
            continue
        for (g,v) in grads_and_vars_vanilla:
            with tf.variable_scope(op, reuse=True):
                w,b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)

                if v.name == w.name:

                    with tf.variable_scope(TF_WEIGHTS, reuse=True):

                        vel = tf.get_variable(TF_POOL_MOMENTUM)
                        vel_update_ops.append(
                            tf.assign(vel,
                                      research_parameters['momentum'] * vel +
                                      (1.0 - research_parameters['momentum']) * g ** 2)
                        )

                        adaptive_w_lr = learning_rate / (tf.sqrt(vel + rms_epsilon))
                        grads_and_vars.append((g * adaptive_w_lr, w))

                if v.name == b.name:

                    with tf.variable_scope(TF_BIAS, reuse=True):

                        vel = tf.get_variable(TF_POOL_MOMENTUM)
                        vel_update_ops.append(
                            tf.assign(vel,
                                      research_parameters['momentum'] * vel +
                                      (1.0 - research_parameters['momentum']) * g ** 2)
                        )

                        adaptive_b_lr = learning_rate / (tf.sqrt(vel + rms_epsilon))
                        grads_and_vars.append((g * adaptive_b_lr, b))

    return optimizer.apply_gradients(grads_and_vars), vel_update_ops


def optimize_masked_momentum_gradient_end_to_end(optimizer, filter_indices_to_replace, adapted_op, avg_grad_and_vars,
                                      tf_cnn_hyperparameters, learning_rate, global_step, use_pool_momentum,tf_scale_parameter, select_from_top):
    global cnn_ops, cnn_hyperparameters, add_amout, add_fulcon_amount

    assert add_amout>0 and add_fulcon_amount > 0

    decay_lr = model_parameters['decay_learning_rate']
    # decay_lr = False

    # Define the learning rate decay
    if decay_lr:
        learning_rate = tf.maximum(model_parameters['min_learning_rate'],
                                   tf.train.exponential_decay(learning_rate, global_step, decay_steps=1,
                                                              decay_rate=model_parameters['adapt_decay_rate'],
                                                              staircase=True))
    else:
        learning_rate = tf.constant(model_parameters['start_lr'], dtype=tf.float32, name='learning_rate')

    vel_update_ops = [] # ops that contain velocity updates
    grad_ops = [] # ops that contain actual gradients

    mask_grads_w, mask_grads_b = {}, {}

    filter_indices_to_replace = tf.reshape(filter_indices_to_replace, [-1, 1])
    replace_amnt = tf.shape(filter_indices_to_replace)[0]

    prev_indices = None
    prev_op = None
    print(adapted_op)
    for lyr_i, tmp_op in enumerate(cnn_ops):
        print('\t',tmp_op)
        if 'conv' in tmp_op:
            with tf.variable_scope(tmp_op, reuse=True) as scope:
                w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)
                with tf.variable_scope('age', reuse=True):
                    age_w, age_b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)

                for (g, v) in avg_grad_and_vars:
                    if v.name == w.name:
                        grads_w = g * tf_scale_parameter[lyr_i]
                    if v.name == b.name:
                        grads_b = g * tf_scale_parameter[lyr_i]

                lyr_conv_shape = tf_cnn_hyperparameters[tmp_op][TF_CONV_WEIGHT_SHAPE_STR]
                transposed_shape = [tf_cnn_hyperparameters[tmp_op][TF_CONV_WEIGHT_SHAPE_STR][3],
                                    tf_cnn_hyperparameters[tmp_op][TF_CONV_WEIGHT_SHAPE_STR][0],
                                    tf_cnn_hyperparameters[tmp_op][TF_CONV_WEIGHT_SHAPE_STR][1],
                                    tf_cnn_hyperparameters[tmp_op][TF_CONV_WEIGHT_SHAPE_STR][2]]

                logger.debug('Applying gradients for %s', tmp_op)
                logger.debug('\tAnd filter IDs: %s', filter_indices_to_replace)

                layer_ind_to_replace = None
                '''if tmp_op==adapted_op:
                    layer_ind_to_replace = filter_indices_to_replace
                else:'''
                # this selects indices either from top or bottom filters
                if select_from_top:
                    layer_ind_to_replace = tf.range(tf_cnn_hyperparameters[tmp_op][TF_CONV_WEIGHT_SHAPE_STR][3]//2,
                                                    tf_cnn_hyperparameters[tmp_op][TF_CONV_WEIGHT_SHAPE_STR][3])
                    adapt_amount = tf_cnn_hyperparameters[tmp_op][TF_CONV_WEIGHT_SHAPE_STR][3] - tf_cnn_hyperparameters[tmp_op][TF_CONV_WEIGHT_SHAPE_STR][3]//2

                else:
                    layer_ind_to_replace = tf.range(0,
                                                    tf_cnn_hyperparameters[tmp_op][TF_CONV_WEIGHT_SHAPE_STR][3]//2)
                    adapt_amount = tf_cnn_hyperparameters[tmp_op][TF_CONV_WEIGHT_SHAPE_STR][3]//2

                # Out channel masking
                mask_grads_w[tmp_op] = tf.scatter_nd(
                    layer_ind_to_replace,
                    tf.ones(shape=[adapt_amount,
                                   transposed_shape[1], transposed_shape[2], transposed_shape[3]],
                            dtype=tf.float32),
                    shape=transposed_shape
                )
                mask_grads_w[tmp_op] = tf.transpose(mask_grads_w[tmp_op], [1, 2, 3, 0])
                grads_w *= mask_grads_w[tmp_op]

                # In channel masking
                '''if prev_op is not None:
                    mask_grads_w[tmp_op] = tf.scatter_nd(
                        prev_indices,
                        tf.ones(shape=[tf_cnn_hyperparameters[prev_op][TF_CONV_WEIGHT_SHAPE_STR][3]//2, lyr_conv_shape[0], lyr_conv_shape[1], lyr_conv_shape[3]],
                                dtype=tf.float32),
                        shape=[lyr_conv_shape[2], lyr_conv_shape[0], lyr_conv_shape[1], lyr_conv_shape[3]]
                    )
                    mask_grads_w[tmp_op] = tf.transpose(mask_grads_w[tmp_op], [1, 2, 0, 3])
                    grads_w *= mask_grads_w[tmp_op]'''

                mask_grads_b[tmp_op] = tf.scatter_nd(
                    layer_ind_to_replace,
                    tf.ones([adapt_amount], dtype=tf.float32),
                    shape=[tf_cnn_hyperparameters[tmp_op][TF_CONV_WEIGHT_SHAPE_STR][3]]
                )

                grads_b *= mask_grads_b[tmp_op]

                with tf.variable_scope(TF_WEIGHTS) as child_scope:
                    pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)
                    w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)

                with tf.variable_scope(TF_BIAS) as child_scope:
                    pool_b_vel = tf.get_variable(TF_POOL_MOMENTUM)
                    b_vel = tf.get_variable(TF_TRAIN_MOMENTUM)

                if use_pool_momentum:
                    vel_update_ops.append(
                        tf.assign(pool_w_vel, research_parameters['pool_momentum'] * pool_w_vel+ (1.0 - research_parameters['pool_momentum'])*grads_w**2))
                    vel_update_ops.append(
                        tf.assign(pool_b_vel, research_parameters['pool_momentum'] * pool_b_vel + (1.0 - research_parameters['pool_momentum']) * grads_b**2))

                    adaptive_w_lr = learning_rate / tf.sqrt(pool_w_vel + rms_epsilon)
                    adaptive_b_lr = learning_rate / tf.sqrt(pool_b_vel + rms_epsilon)

                else:
                    vel_update_ops.append(
                        tf.assign(w_vel, research_parameters['momentum'] * w_vel + (1.0 - research_parameters['momentum']) * grads_w**2))
                    vel_update_ops.append(
                        tf.assign(b_vel, research_parameters['momentum'] * b_vel + (1.0 - research_parameters['momentum']) * grads_b**2))

                    adaptive_w_lr = learning_rate / (tf.sqrt(w_vel + rms_epsilon)*age_w)
                    adaptive_b_lr = learning_rate / (tf.sqrt(b_vel + rms_epsilon)*age_b)

                grad_ops.append(optimizer.apply_gradients([(grads_w * adaptive_w_lr, w), (grads_b * adaptive_b_lr, b)]))

                #grad_ops.append(
                #    optimizer.apply_gradients([(w_vel * learning_rate  * mask_grads_w[tmp_op], w),
                #                               (b_vel * learning_rate * mask_grads_b[tmp_op], b)]))
                prev_indices = layer_ind_to_replace
                prev_op = tmp_op

        elif 'fulcon' in tmp_op and tmp_op!='fulcon_out':

            with tf.variable_scope(tmp_op, reuse=True) as scope:
                w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)

                with tf.variable_scope('age', reuse=True):
                    age_w, age_b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)

                for (g, v) in avg_grad_and_vars:
                    if v.name == w.name:
                        grads_w = g * tf_scale_parameter[lyr_i]
                    if v.name == b.name:
                        grads_b = g * tf_scale_parameter[lyr_i]

                lyr_fulcon_shape = [tf_cnn_hyperparameters[tmp_op][TF_FC_WEIGHT_IN_STR],
                                    tf_cnn_hyperparameters[tmp_op][TF_FC_WEIGHT_OUT_STR]]

                transposed_shape = [tf_cnn_hyperparameters[tmp_op][TF_FC_WEIGHT_OUT_STR],
                                    tf_cnn_hyperparameters[tmp_op][TF_FC_WEIGHT_IN_STR],
                                    ]

                logger.debug('Applying gradients for %s', tmp_op)
                logger.debug('\tAnd filter IDs: %s', filter_indices_to_replace)

                layer_ind_to_replace = None
                '''if tmp_op==adapted_op:
                    layer_ind_to_replace = filter_indices_to_replace

                else:'''
                if select_from_top:
                    layer_ind_to_replace = tf.range(tf_cnn_hyperparameters[tmp_op][TF_FC_WEIGHT_OUT_STR] // 2,
                                                    tf_cnn_hyperparameters[tmp_op][TF_FC_WEIGHT_OUT_STR])
                    adapt_amount = tf_cnn_hyperparameters[tmp_op][TF_FC_WEIGHT_OUT_STR] - tf_cnn_hyperparameters[tmp_op][TF_FC_WEIGHT_OUT_STR]//2
                else:
                    layer_ind_to_replace = tf.range(0, tf_cnn_hyperparameters[tmp_op][TF_FC_WEIGHT_OUT_STR]//2)
                    adapt_amount = tf_cnn_hyperparameters[tmp_op][TF_FC_WEIGHT_OUT_STR]//2

                # Out channel masking
                mask_grads_w[tmp_op] = tf.scatter_nd(
                    layer_ind_to_replace,
                    tf.ones(shape=[adapt_amount, transposed_shape[1]],
                            dtype=tf.float32),
                    shape=transposed_shape
                )
                mask_grads_w[tmp_op] = tf.transpose(mask_grads_w[tmp_op], [1, 0])
                grads_w = grads_w * mask_grads_w[tmp_op]

                # In channel masking
                '''if 'conv' in prev_op:
                    offset = tf.reshape(tf.range(0,final_2d_width*final_2d_width),[1,-1])
                    prev_fulcon_ind = tf.tile(tf.reshape(prev_indices,[-1,1]),[1,final_2d_width*final_2d_width]) + offset
                    prev_fulcon_ind = tf.reshape(prev_fulcon_ind,[-1])

                    mask_grads_w[tmp_op] = tf.scatter_nd(
                        prev_fulcon_ind,
                        tf.ones(shape=[(tf_cnn_hyperparameters[prev_op][TF_CONV_WEIGHT_SHAPE_STR][3]//2)*final_2d_width*final_2d_width, lyr_fulcon_shape[1]],
                                dtype=tf.float32),
                        shape=lyr_fulcon_shape
                    )
                    grads_w = grads_w * mask_grads_w[tmp_op]
                else:
                    mask_grads_w[tmp_op] = tf.scatter_nd(
                        prev_indices,
                        tf.ones(shape=[tf_cnn_hyperparameters[prev_op][TF_FC_WEIGHT_OUT_STR]//2, lyr_fulcon_shape[1]],
                                dtype=tf.float32),
                        shape=lyr_fulcon_shape
                    )
                    grads_w = grads_w * mask_grads_w[tmp_op]'''

                mask_grads_b[tmp_op] = tf.scatter_nd(
                    layer_ind_to_replace,
                    tf.ones([adapt_amount], dtype=tf.float32),
                    shape=[tf_cnn_hyperparameters[tmp_op][TF_FC_WEIGHT_OUT_STR]]
                )

                grads_b = grads_b * mask_grads_b[tmp_op]

                with tf.variable_scope(TF_WEIGHTS) as child_scope:
                    pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)
                    w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
                with tf.variable_scope(TF_BIAS) as child_scope:
                    pool_b_vel = tf.get_variable(TF_POOL_MOMENTUM)
                    b_vel = tf.get_variable(TF_TRAIN_MOMENTUM)

                if use_pool_momentum:
                    vel_update_ops.append(
                        tf.assign(pool_w_vel, research_parameters['pool_momentum'] * pool_w_vel + (1.0 - research_parameters['pool_momentum'])*grads_w**2))
                    vel_update_ops.append(
                        tf.assign(pool_b_vel, research_parameters['pool_momentum'] * pool_b_vel + (1.0 -research_parameters['pool_momentum'])*grads_b**2))

                    adaptive_w_lr = learning_rate / tf.sqrt(pool_w_vel + rms_epsilon)
                    adaptive_b_lr = learning_rate / tf.sqrt(pool_b_vel + rms_epsilon)
                else:

                    vel_update_ops.append(
                        tf.assign(w_vel, research_parameters['momentum'] * w_vel + (1.0 - research_parameters['momentum']) *grads_w**2))
                    vel_update_ops.append(
                        tf.assign(b_vel, research_parameters['momentum'] * b_vel + (1.0 - research_parameters['momentum']) * grads_b**2))

                    adaptive_w_lr = learning_rate / (tf.sqrt(w_vel + rms_epsilon)*age_w)
                    adaptive_b_lr = learning_rate / (tf.sqrt(b_vel + rms_epsilon)*age_b)

                grad_ops.append(optimizer.apply_gradients(
                    [(grads_w*adaptive_w_lr, w), (grads_b * adaptive_b_lr, b)]))

            prev_indices = layer_ind_to_replace
            prev_op = tmp_op

        elif tmp_op=='fulcon_out':

            with tf.variable_scope(tmp_op, reuse=True) as scope:
                w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)
                with tf.variable_scope('age', reuse=True):
                    age_w, age_b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)

                for (g, v) in avg_grad_and_vars:
                    if v.name == w.name:
                        grads_w = g
                    if v.name == b.name:
                        grads_b = g

                with tf.variable_scope(TF_WEIGHTS) as child_scope:
                    pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)
                    w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
                with tf.variable_scope(TF_BIAS) as child_scope:
                    pool_b_vel = tf.get_variable(TF_POOL_MOMENTUM)
                    b_vel = tf.get_variable(TF_TRAIN_MOMENTUM)

                if use_pool_momentum:
                    vel_update_ops.append(
                        tf.assign(pool_w_vel, research_parameters['pool_momentum'] * pool_w_vel + (1.0 - research_parameters['pool_momentum']) * grads_w**2))
                    vel_update_ops.append(
                        tf.assign(pool_b_vel, research_parameters['pool_momentum'] * pool_b_vel + (1.0 - research_parameters['pool_momentum']) * grads_b**2))

                    adaptive_w_lr = learning_rate / (tf.sqrt(pool_w_vel) + rms_epsilon)
                    adaptive_b_lr = learning_rate / (tf.sqrt(pool_b_vel) + rms_epsilon)
                else:
                    vel_update_ops.append(
                        tf.assign(w_vel, research_parameters['momentum'] * w_vel + (1-research_parameters['momentum'])*grads_w**2))
                    vel_update_ops.append(
                        tf.assign(b_vel, research_parameters['momentum'] * b_vel + (1-research_parameters['momentum'])*grads_b**2))

                    adaptive_w_lr = learning_rate / (tf.sqrt(w_vel + rms_epsilon)*age_w)
                    adaptive_b_lr = learning_rate / (tf.sqrt(b_vel + rms_epsilon)*age_b)

                grad_ops.append(optimizer.apply_gradients(
                    [(grads_w * adaptive_w_lr, w), (grads_b * adaptive_b_lr, b)]))

    return grad_ops, vel_update_ops


def optimize_masked_momentum_gradient(optimizer, filter_indices_to_replace, op, avg_grad_and_vars,
                                      tf_cnn_hyperparameters, learning_rate, global_step):
    '''
    Any adaptation of a convolutional layer would result in a change in the following layer.
    This optimization optimize the filters/weights responsible in both those layer
    :param loss:
    :param filter_indices_to_replace:
    :param op:
    :param w:
    :param b:
    :param cnn_hyps:
    :param cnn_ops:
    :return:
    '''
    global cnn_ops, cnn_hyperparameters

    decay_lr = model_parameters['decay_learning_rate']
    #decay_lr = False
    if decay_lr:
        learning_rate = tf.maximum(model_parameters['min_learning_rate'],
                                   tf.train.exponential_decay(learning_rate, global_step, decay_steps=1,
                                                              decay_rate=model_parameters['adapt_decay_rate'], staircase=True))
    else:
        learning_rate = tf.constant(model_parameters['start_lr'], dtype=tf.float32, name='learning_rate')

    vel_update_ops = []
    grad_ops = []

    mask_grads_w, mask_grads_b = {}, {}

    filter_indices_to_replace = tf.reshape(filter_indices_to_replace, [-1, 1])
    replace_amnt = tf.shape(filter_indices_to_replace)[0]

    if 'conv' in op:
        with tf.variable_scope(op, reuse=True) as scope:
            w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)
            for (g, v) in avg_grad_and_vars:
                if v.name == w.name:
                    grads_w = g
                if v.name == b.name:
                    grads_b = g

            transposed_shape = [tf_cnn_hyperparameters[op][TF_CONV_WEIGHT_SHAPE_STR][3],
                                tf_cnn_hyperparameters[op][TF_CONV_WEIGHT_SHAPE_STR][0],
                                tf_cnn_hyperparameters[op][TF_CONV_WEIGHT_SHAPE_STR][1],
                                tf_cnn_hyperparameters[op][TF_CONV_WEIGHT_SHAPE_STR][2]]

            logger.debug('Applying gradients for %s', op)
            logger.debug('\tAnd filter IDs: %s', filter_indices_to_replace)

            mask_grads_w[op] = tf.scatter_nd(
                filter_indices_to_replace,
                tf.ones(shape=[replace_amnt, transposed_shape[1], transposed_shape[2], transposed_shape[3]],
                        dtype=tf.float32),
                shape=transposed_shape
            )

            mask_grads_w[op] = tf.transpose(mask_grads_w[op], [1, 2, 3, 0])

            mask_grads_b[op] = tf.scatter_nd(
                filter_indices_to_replace,
                tf.ones([replace_amnt], dtype=tf.float32),
                shape=[tf_cnn_hyperparameters[op][TF_CONV_WEIGHT_SHAPE_STR][3]]
            )

            grads_w = grads_w * mask_grads_w[op]
            grads_b = grads_b * mask_grads_b[op]

            with tf.variable_scope(TF_WEIGHTS) as child_scope:
                pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)
                w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
            with tf.variable_scope(TF_BIAS) as child_scope:
                pool_b_vel = tf.get_variable(TF_POOL_MOMENTUM)
                b_vel = tf.get_variable(TF_TRAIN_MOMENTUM)

            vel_update_ops.append(
                tf.assign(pool_w_vel, research_parameters['pool_momentum'] * pool_w_vel + (1.0 - research_parameters['pool_momentum'])*grads_w**2))
            vel_update_ops.append(
                tf.assign(pool_b_vel, research_parameters['pool_momentum'] * pool_b_vel + (1.0-research_parameters['pool_momentum'])*grads_b**2))

            adaptive_w_lr = learning_rate / tf.sqrt(pool_w_vel + rms_epsilon)
            adaptive_b_lr = learning_rate / tf.sqrt(pool_b_vel + rms_epsilon)
            grad_ops.append(optimizer.apply_gradients([(grads_w * adaptive_w_lr, w), (grads_b * adaptive_b_lr, b)]))
            #grad_ops.append(optimizer.apply_gradients([(w_vel * learning_rate * mask_grads_w[op], w), (b_vel * learning_rate * mask_grads_b[op], b)]))

    next_op = None
    for tmp_op in cnn_ops[cnn_ops.index(op) + 1:]:
        if 'conv' in tmp_op or 'fulcon' in tmp_op:
            next_op = tmp_op
            break
    logger.debug('Next conv op: %s', next_op)

    if 'conv' in next_op:
        with tf.variable_scope(next_op, reuse=True) as scope:
            w = tf.get_variable(TF_WEIGHTS)
            for (g, v) in avg_grad_and_vars:
                if v.name == w.name:
                    grads_w = g
                    break

            transposed_shape = [tf_cnn_hyperparameters[next_op][TF_CONV_WEIGHT_SHAPE_STR][2],
                                tf_cnn_hyperparameters[next_op][TF_CONV_WEIGHT_SHAPE_STR][0],
                                tf_cnn_hyperparameters[next_op][TF_CONV_WEIGHT_SHAPE_STR][1],
                                tf_cnn_hyperparameters[next_op][TF_CONV_WEIGHT_SHAPE_STR][3]]

            logger.debug('Applying gradients for %s', next_op)
            logger.debug('\tAnd filter IDs: %s', filter_indices_to_replace)

            mask_grads_w[next_op] = tf.scatter_nd(
                tf.reshape(filter_indices_to_replace, [-1, 1]),
                tf.ones(shape=[replace_amnt, transposed_shape[1], transposed_shape[2], transposed_shape[3]],
                        dtype=tf.float32),
                shape=transposed_shape
            )

            mask_grads_w[next_op] = tf.transpose(mask_grads_w[next_op], [1, 2, 0, 3])
            grads_w = grads_w * mask_grads_w[next_op]

            with tf.variable_scope(TF_WEIGHTS) as child_scope:
                pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)
                w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)

            vel_update_ops.append(
                tf.assign(pool_w_vel, research_parameters['pool_momentum'] * pool_w_vel + (1.0-research_parameters['pool_momentum'])*grads_w**2))

            adaptive_w_lr = learning_rate / tf.sqrt(pool_w_vel + rms_epsilon)
            grad_ops.append(optimizer.apply_gradients([(grads_w * adaptive_w_lr, w)]))

    elif 'fulcon' in next_op:
        with tf.variable_scope(next_op, reuse=True) as scope:
            w = tf.get_variable(TF_WEIGHTS)
            for (g, v) in avg_grad_and_vars:
                if v.name == w.name:
                    grads_w = g
                    break

            logger.debug('Applying gradients for %s', next_op)
            logger.debug('\tAnd filter IDs: %s', filter_indices_to_replace)

            mask_grads_w[next_op] = tf.scatter_nd(
                tf.reshape(filter_indices_to_replace, [-1, 1]),
                tf.ones(shape=[replace_amnt, tf_cnn_hyperparameters[next_op][TF_FC_WEIGHT_OUT_STR]],
                        dtype=tf.float32),
                shape=[tf_cnn_hyperparameters[next_op][TF_FC_WEIGHT_IN_STR],
                       tf_cnn_hyperparameters[next_op][TF_FC_WEIGHT_OUT_STR]]
            )

            grads_w = grads_w * mask_grads_w[next_op]
            with tf.variable_scope(TF_WEIGHTS) as child_scope:
                pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)
                w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)

            vel_update_ops.append(
                tf.assign(pool_w_vel,
                          research_parameters['pool_momentum'] * pool_w_vel + (1.0 - research_parameters['pool_momentum'])*grads_w**2))

            adaptive_w_lr = learning_rate / tf.sqrt(pool_w_vel + rms_epsilon)
            grad_ops.append(optimizer.apply_gradients([(grads_w * adaptive_w_lr, w)]))

    return grad_ops, vel_update_ops


def optimize_masked_momentum_gradient_for_fulcon(optimizer, filter_indices_to_replace, op, avg_grad_and_vars,
                                      tf_cnn_hyperparameters, learning_rate, global_step):
    '''
    Any adaptation of a convolutional layer would result in a change in the following layer.
    This optimization optimize the filters/weights responsible in both those layer
    :param loss:
    :param filter_indices_to_replace:
    :param op:
    :param w:
    :param b:
    :param cnn_hyps:
    :param cnn_ops:
    :return:
    '''
    global cnn_ops, cnn_hyperparameters

    decay_lr = model_parameters['decay_learning_rate']
    #decay_lr = False
    if decay_lr:
        learning_rate = tf.maximum(model_parameters['min_learning_rate'],
                                   tf.train.exponential_decay(learning_rate, global_step, decay_steps=1,
                                                              decay_rate=model_parameters['adapt_decay_rate'], staircase=True))
    else:
        learning_rate = tf.constant(model_parameters['start_lr'], dtype=tf.float32, name='learning_rate')

    vel_update_ops = []
    grad_ops = []

    mask_grads_w, mask_grads_b = {}, {}

    filter_indices_to_replace = tf.reshape(filter_indices_to_replace, [-1, 1])
    replace_amnt = tf.shape(filter_indices_to_replace)[0]

    if 'fulcon' in op:
        with tf.variable_scope(op, reuse=True) as scope:
            w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)
            for (g, v) in avg_grad_and_vars:
                if v.name == w.name:
                    grads_w = g
                if v.name == b.name:
                    grads_b = g

            transposed_shape = [tf_cnn_hyperparameters[op][TF_FC_WEIGHT_OUT_STR],
                                tf_cnn_hyperparameters[op][TF_FC_WEIGHT_IN_STR],
                                ]

            logger.debug('Applying gradients for %s', op)
            logger.debug('\tAnd filter IDs: %s', filter_indices_to_replace)

            mask_grads_w[op] = tf.scatter_nd(
                filter_indices_to_replace,
                tf.ones(shape=[replace_amnt, transposed_shape[1]],
                        dtype=tf.float32),
                shape=transposed_shape
            )

            mask_grads_w[op] = tf.transpose(mask_grads_w[op], [1, 0])

            mask_grads_b[op] = tf.scatter_nd(
                filter_indices_to_replace,
                tf.ones([replace_amnt], dtype=tf.float32),
                shape=[tf_cnn_hyperparameters[op][TF_FC_WEIGHT_OUT_STR]]
            )

            grads_w = grads_w * mask_grads_w[op]
            grads_b = grads_b * mask_grads_b[op]

            with tf.variable_scope(TF_WEIGHTS) as child_scope:
                pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)
                w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
            with tf.variable_scope(TF_BIAS) as child_scope:
                pool_b_vel = tf.get_variable(TF_POOL_MOMENTUM)
                b_vel = tf.get_variable(TF_TRAIN_MOMENTUM)

            vel_update_ops.append(
                tf.assign(pool_w_vel, research_parameters['pool_momentum'] * pool_w_vel + (1.0 - research_parameters['pool_momentum'])*grads_w**2))
            vel_update_ops.append(
                tf.assign(pool_b_vel, research_parameters['pool_momentum'] * pool_b_vel + (1.0 - research_parameters['pool_momentum'])*grads_b**2))

            adaptive_w_lr = learning_rate / tf.sqrt(pool_w_vel + rms_epsilon)
            adaptive_b_lr = learning_rate / tf.sqrt(pool_b_vel + rms_epsilon)
            grad_ops.append(optimizer.apply_gradients([(grads_w * adaptive_w_lr, w), (grads_b * adaptive_b_lr, b)]))

    next_op = cnn_ops[cnn_ops.index(op) + 1]

    logger.debug('Next fulcon op: %s', next_op)

    with tf.variable_scope(next_op, reuse=True) as scope:
        w = tf.get_variable(TF_WEIGHTS)
        for (g, v) in avg_grad_and_vars:
            if v.name == w.name:
                grads_w = g
                break

        logger.debug('Applying gradients for %s', next_op)
        logger.debug('\tAnd filter IDs: %s', filter_indices_to_replace)

        mask_grads_w[next_op] = tf.scatter_nd(
            tf.reshape(filter_indices_to_replace, [-1, 1]),
            tf.ones(shape=[replace_amnt, tf_cnn_hyperparameters[next_op][TF_FC_WEIGHT_OUT_STR]],
                    dtype=tf.float32),
            shape=[tf_cnn_hyperparameters[next_op][TF_FC_WEIGHT_IN_STR],
                   tf_cnn_hyperparameters[next_op][TF_FC_WEIGHT_OUT_STR]]
        )

        grads_w = grads_w * mask_grads_w[next_op]
        with tf.variable_scope(TF_WEIGHTS) as child_scope:
            pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)
            w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)

        vel_update_ops.append(
            tf.assign(pool_w_vel,
                      research_parameters['pool_momentum'] * pool_w_vel + (1.0 -research_parameters['pool_momentum'])*grads_w**2))

        adaptive_lr = learning_rate/tf.sqrt(pool_w_vel+rms_epsilon)
        grad_ops.append(optimizer.apply_gradients([(grads_w * adaptive_lr, w)]))

    return grad_ops, vel_update_ops
