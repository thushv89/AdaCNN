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


def update_train_momentum_velocity(grads_and_vars):
    vel_update_ops = []
    # update velocity vector

    for (g, v) in grads_and_vars:
        var_name_tokens = v.name.split(':')[0].split(TF_SCOPE_DIVIDER)
        new_var_name = ''
        for tok in var_name_tokens:
            if tok.startswith('conv') or tok.startswith('fulcon'):
                new_var_name += tok + TF_SCOPE_DIVIDER
            elif tok.startswith(TF_WEIGHTS) or tok.startswith(TF_BIAS):
                new_var_name += tok

        with tf.variable_scope(new_var_name, reuse=True) as scope:
            vel = tf.get_variable(TF_TRAIN_MOMENTUM)

            vel_update_ops.append(
                tf.assign(vel,
                          research_parameters['momentum'] * vel + g)
            )

    return vel_update_ops


def update_pool_momentum_velocity(grads_and_vars):
    vel_update_ops = []
    # update velocity vector

    for (g, v) in grads_and_vars:
        var_name_tokens = v.name.split(':')[0].split(TF_SCOPE_DIVIDER)
        new_var_name = ''
        for tok in var_name_tokens:
            if tok.startswith('conv') or tok.startswith('fulcon'):
                new_var_name += tok + TF_SCOPE_DIVIDER
            elif tok.startswith(TF_WEIGHTS) or tok.startswith(TF_BIAS):
                new_var_name += tok

        with tf.variable_scope(new_var_name, reuse=True) as scope:
            vel = tf.get_variable(TF_POOL_MOMENTUM)

            vel_update_ops.append(
                tf.assign(vel,
                          research_parameters['pool_momentum'] * vel + g)
            )
    return vel_update_ops


def apply_gradient_with_momentum(optimizer, learning_rate, global_step):
    grads_and_vars = []
    # for each trainable variable
    if model_parameters['decay_learning_rate']:
        learning_rate = tf.maximum(model_parameters['min_learning_rate'],
                                   tf.train.exponential_decay(learning_rate, global_step, decay_steps=1,
                                                              decay_rate=model_parameters['decay_rate'], staircase=True))
    for scope in cnn_ops:
        if 'pool' in scope:
            continue

        with tf.variable_scope(scope, reuse=True) as scope:
            w = tf.get_variable(TF_WEIGHTS)
            with tf.variable_scope(TF_WEIGHTS, reuse=True):
                logger.debug('Grads and Vars for variable %s (using scope %s)', w.name, scope.name)
                vel = tf.get_variable(TF_TRAIN_MOMENTUM)
                grads_and_vars.append((vel * learning_rate, w))

            b = tf.get_variable(TF_BIAS)
            with tf.variable_scope(TF_BIAS, reuse=True):
                logger.debug('Grads and Vars for variable %s (using scope %s)', b.name, scope.name)
                vel = tf.get_variable(TF_TRAIN_MOMENTUM)
                grads_and_vars.append((vel * learning_rate, b))

    return optimizer.apply_gradients(grads_and_vars)


def apply_gradient_with_pool_momentum(optimizer, learning_rate, global_step):
    grads_and_vars = []
    # for each trainable variable
    if model_parameters['decay_learning_rate']:
        learning_rate = tf.maximum(model_parameters['min_learning_rate'],
                                   tf.train.exponential_decay(learning_rate, global_step, decay_steps=1,
                                                              decay_rate=model_parameters['adapt_decay_rate'], staircase=True))
    for scope in cnn_ops:
        if 'pool' in scope:
            continue
        with tf.variable_scope(scope, reuse=True):
            w = tf.get_variable(TF_WEIGHTS)
            logger.debug('Grads and Vars for variable %s', w.name)
            with tf.variable_scope(TF_WEIGHTS, reuse=True):
                vel = tf.get_variable(TF_POOL_MOMENTUM)
                grads_and_vars.append((vel * learning_rate, w))

            b = tf.get_variable(TF_BIAS)
            logger.debug('Grads and Vars for variable %s', b.name)
            with tf.variable_scope(TF_BIAS, reuse=True):
                vel = tf.get_variable(TF_POOL_MOMENTUM)
                grads_and_vars.append((vel * learning_rate, b))

    return optimizer.apply_gradients(grads_and_vars)


def optimize_with_momentum(optimizer, loss, global_step, learning_rate):
    vel_update_ops, grad_update_ops = [], []

    if model_parameters['decay_learning_rate']:
        learning_rate = tf.maximum(model_parameters['min_learning_rate'],
                                   tf.train.exponential_decay(learning_rate, global_step, decay_steps=1,
                                                              decay_rate=model_parameters['decay_rate'], staircase=True))

    for op in cnn_ops:
        if 'conv' in op and 'fulcon' in op:
            with tf.variable_scope(op) as scope:
                w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)
                [(grads_w, w), (grads_b, b)] = optimizer.compute_gradients(loss, [w, b])

                # update velocity vector
                with tf.variable_scope(TF_WEIGHTS) as child_scope:
                    w_vel = tf.get_variable(TF_POOL_MOMENTUM)
                    vel_update_ops.append(
                        tf.assign(w_vel,
                                  research_parameters['momentum'] * w_vel + grads_w)
                    )
                with tf.variable_scope(TF_BIAS) as child_scope:
                    b_vel = tf.get_variable(TF_POOL_MOMENTUM)
                    vel_update_ops.append(
                        tf.assign(b_vel,
                                  research_parameters['momentum'] * b_vel + grads_b)
                    )
                grad_update_ops.append([(w_vel * learning_rate, w), (b_vel * learning_rate, b)])

    return grad_update_ops, vel_update_ops


def optimize_masked_momentum_gradient_end_to_end(optimizer, filter_indices_to_replace, adapted_op, avg_grad_and_vars,
                                      tf_cnn_hyperparameters, learning_rate, global_step, use_pool_momentum,tf_scale_parameter):
    global cnn_ops, cnn_hyperparameters

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
                if tmp_op==adapted_op:
                    layer_ind_to_replace = filter_indices_to_replace
                else:
                    layer_ind_to_replace = tf.random_shuffle(tf.range(0,tf_cnn_hyperparameters[tmp_op][TF_CONV_WEIGHT_SHAPE_STR][3]))[:add_amout]

                # Out channel masking
                mask_grads_w[tmp_op] = tf.scatter_nd(
                    layer_ind_to_replace,
                    tf.ones(shape=[add_amout, transposed_shape[1], transposed_shape[2], transposed_shape[3]],
                            dtype=tf.float32),
                    shape=transposed_shape
                )
                mask_grads_w[tmp_op] = tf.transpose(mask_grads_w[tmp_op], [1, 2, 3, 0])
                grads_w *= mask_grads_w[tmp_op]

                # In channel masking
                if prev_op is not None:
                    mask_grads_w[tmp_op] = tf.scatter_nd(
                        prev_indices,
                        tf.ones(shape=[add_amout, lyr_conv_shape[0], lyr_conv_shape[1], lyr_conv_shape[3]],
                                dtype=tf.float32),
                        shape=[lyr_conv_shape[2], lyr_conv_shape[0], lyr_conv_shape[1], lyr_conv_shape[3]]
                    )
                    mask_grads_w[tmp_op] = tf.transpose(mask_grads_w[tmp_op], [1, 2, 0, 3])
                    grads_w *= mask_grads_w[tmp_op]

                mask_grads_b[tmp_op] = tf.scatter_nd(
                    layer_ind_to_replace,
                    tf.ones([add_amout], dtype=tf.float32),
                    shape=[tf_cnn_hyperparameters[tmp_op][TF_CONV_WEIGHT_SHAPE_STR][3]]
                )

                grads_b *= mask_grads_b[tmp_op]

                if use_pool_momentum:
                    with tf.variable_scope(TF_WEIGHTS) as child_scope:
                        w_vel = tf.get_variable(TF_POOL_MOMENTUM)
                    with tf.variable_scope(TF_BIAS) as child_scope:
                        b_vel = tf.get_variable(TF_POOL_MOMENTUM)


                    vel_update_ops.append(
                        tf.assign(w_vel, research_parameters['pool_momentum'] * w_vel + grads_w))
                    vel_update_ops.append(
                        tf.assign(b_vel, research_parameters['pool_momentum'] * b_vel + grads_b))
                else:
                    with tf.variable_scope(TF_WEIGHTS) as child_scope:
                        w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
                    with tf.variable_scope(TF_BIAS) as child_scope:
                        b_vel = tf.get_variable(TF_TRAIN_MOMENTUM)

                    vel_update_ops.append(
                        tf.assign(w_vel, research_parameters['momentum'] * w_vel + grads_w))
                    vel_update_ops.append(
                        tf.assign(b_vel, research_parameters['momentum'] * b_vel + grads_b))

                grad_ops.append(
                    optimizer.apply_gradients([(w_vel * learning_rate  * mask_grads_w[tmp_op], w),
                                               (b_vel * learning_rate * mask_grads_b[tmp_op], b)]))
                prev_indices = layer_ind_to_replace
                prev_op = tmp_op

        elif 'fulcon' in tmp_op and tmp_op!='fulcon_out':

            with tf.variable_scope(tmp_op, reuse=True) as scope:
                w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)
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
                if tmp_op==adapted_op:
                    layer_ind_to_replace = filter_indices_to_replace

                else:
                    layer_ind_to_replace = tf.random_shuffle(tf.range(0, tf_cnn_hyperparameters[tmp_op][TF_FC_WEIGHT_OUT_STR]))[:add_fulcon_amount]

                # Out channel masking
                mask_grads_w[tmp_op] = tf.scatter_nd(
                    layer_ind_to_replace,
                    tf.ones(shape=[add_fulcon_amount, transposed_shape[1]],
                            dtype=tf.float32),
                    shape=transposed_shape
                )
                mask_grads_w[tmp_op] = tf.transpose(mask_grads_w[tmp_op], [1, 0])
                grads_w = grads_w * mask_grads_w[tmp_op]

                # In channel masking
                if 'conv' in prev_op:
                    offset = tf.reshape(tf.range(0,final_2d_width*final_2d_width),[1,-1])
                    prev_fulcon_ind = tf.tile(tf.reshape(prev_indices,[-1,1]),[1,final_2d_width*final_2d_width]) + offset
                    prev_fulcon_ind = tf.reshape(prev_fulcon_ind,[-1])

                    mask_grads_w[tmp_op] = tf.scatter_nd(
                        prev_fulcon_ind,
                        tf.ones(shape=[add_amout*final_2d_width*final_2d_width, lyr_fulcon_shape[1]],
                                dtype=tf.float32),
                        shape=lyr_fulcon_shape
                    )
                    grads_w = grads_w * mask_grads_w[tmp_op]
                else:
                    mask_grads_w[tmp_op] = tf.scatter_nd(
                        prev_indices,
                        tf.ones(shape=[add_fulcon_amount, lyr_fulcon_shape[1]],
                                dtype=tf.float32),
                        shape=lyr_fulcon_shape
                    )
                    grads_w = grads_w * mask_grads_w[tmp_op]

                mask_grads_b[tmp_op] = tf.scatter_nd(
                    layer_ind_to_replace,
                    tf.ones([add_fulcon_amount], dtype=tf.float32),
                    shape=[tf_cnn_hyperparameters[tmp_op][TF_FC_WEIGHT_OUT_STR]]
                )

                grads_b = grads_b * mask_grads_b[tmp_op]

                if use_pool_momentum:
                    with tf.variable_scope(TF_WEIGHTS) as child_scope:
                        w_vel = tf.get_variable(TF_POOL_MOMENTUM)
                    with tf.variable_scope(TF_BIAS) as child_scope:
                        b_vel = tf.get_variable(TF_POOL_MOMENTUM)

                    vel_update_ops.append(
                        tf.assign(w_vel, research_parameters['pool_momentum'] * w_vel + grads_w))
                    vel_update_ops.append(
                        tf.assign(b_vel, research_parameters['pool_momentum'] * b_vel + grads_b))
                else:

                    with tf.variable_scope(TF_WEIGHTS) as child_scope:
                        w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
                    with tf.variable_scope(TF_BIAS) as child_scope:
                        b_vel = tf.get_variable(TF_TRAIN_MOMENTUM)

                    vel_update_ops.append(
                        tf.assign(w_vel, research_parameters['momentum'] * w_vel + grads_w))
                    vel_update_ops.append(
                        tf.assign(b_vel, research_parameters['momentum'] * b_vel + grads_b))

                grad_ops.append(optimizer.apply_gradients(
                    [(w_vel * learning_rate * mask_grads_w[tmp_op], w), (b_vel * learning_rate * mask_grads_b[tmp_op], b)]))

            prev_indices = layer_ind_to_replace
            prev_op = tmp_op

        elif tmp_op=='fulcon_out':

            with tf.variable_scope(tmp_op, reuse=True) as scope:
                w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)
                for (g, v) in avg_grad_and_vars:
                    if v.name == w.name:
                        grads_w = g
                    if v.name == b.name:
                        grads_b = g

                if use_pool_momentum:
                    with tf.variable_scope(TF_WEIGHTS) as child_scope:
                        w_vel = tf.get_variable(TF_POOL_MOMENTUM)
                    with tf.variable_scope(TF_BIAS) as child_scope:
                        b_vel = tf.get_variable(TF_POOL_MOMENTUM)

                    vel_update_ops.append(
                        tf.assign(w_vel, research_parameters['pool_momentum'] * w_vel + grads_w))
                    vel_update_ops.append(
                        tf.assign(b_vel, research_parameters['pool_momentum'] * b_vel + grads_b))
                else:

                    with tf.variable_scope(TF_WEIGHTS) as child_scope:
                        w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
                    with tf.variable_scope(TF_BIAS) as child_scope:
                        b_vel = tf.get_variable(TF_TRAIN_MOMENTUM)

                    vel_update_ops.append(
                        tf.assign(w_vel, research_parameters['momentum'] * w_vel + grads_w))
                    vel_update_ops.append(
                        tf.assign(b_vel, research_parameters['momentum'] * b_vel + grads_b))

                grad_ops.append(optimizer.apply_gradients(
                    [(w_vel * learning_rate, w), (b_vel * learning_rate, b)]))

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
                w_vel = tf.get_variable(TF_POOL_MOMENTUM)
            with tf.variable_scope(TF_BIAS) as child_scope:
                b_vel = tf.get_variable(TF_POOL_MOMENTUM)

            vel_update_ops.append(
                tf.assign(w_vel, research_parameters['pool_momentum'] * w_vel + grads_w))
            vel_update_ops.append(
                tf.assign(b_vel, research_parameters['pool_momentum'] * b_vel + grads_b))

            grad_ops.append(optimizer.apply_gradients([(w_vel * learning_rate * mask_grads_w[op], w), (b_vel * learning_rate * mask_grads_b[op], b)]))

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
            vel_update_ops.append(
                tf.assign(pool_w_vel, research_parameters['pool_momentum'] * pool_w_vel + grads_w))

            grad_ops.append(optimizer.apply_gradients([(pool_w_vel * learning_rate * mask_grads_w[next_op], w)]))

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

            vel_update_ops.append(
                tf.assign(pool_w_vel,
                          research_parameters['pool_momentum'] * pool_w_vel + grads_w))

            grad_ops.append(optimizer.apply_gradients([(pool_w_vel * learning_rate * mask_grads_w[next_op], w)]))

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
                w_vel = tf.get_variable(TF_POOL_MOMENTUM)
            with tf.variable_scope(TF_BIAS) as child_scope:
                b_vel = tf.get_variable(TF_POOL_MOMENTUM)

            vel_update_ops.append(
                tf.assign(w_vel, research_parameters['pool_momentum'] * w_vel + grads_w))
            vel_update_ops.append(
                tf.assign(b_vel, research_parameters['pool_momentum'] * b_vel + grads_b))

            grad_ops.append(optimizer.apply_gradients([(w_vel * learning_rate *mask_grads_w[op], w), (b_vel * learning_rate * mask_grads_b[op], b)]))

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

        vel_update_ops.append(
            tf.assign(pool_w_vel,
                      research_parameters['pool_momentum'] * pool_w_vel + grads_w))

        grad_ops.append(optimizer.apply_gradients([(pool_w_vel * learning_rate * mask_grads_w[next_op], w)]))

    return grad_ops, vel_update_ops


def momentum_gradient_with_indices(optimizer, loss, filter_indices_to_replace, op, tf_cnn_hyperparameters):
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

    vel_update_ops = []
    grad_ops = []
    grads_w, grads_b = {}, {}
    mask_grads_w, mask_grads_b = {}, {}
    learning_rate = tf.constant(model_parameters['start_lr'], dtype=tf.float32, name='learning_rate')

    filter_indices_to_replace = tf.reshape(filter_indices_to_replace, [-1, 1])
    replace_amnt = tf.shape(filter_indices_to_replace)[0]

    if 'conv' in op:
        with tf.variable_scope(op, reuse=True) as scope:
            w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)
            [(grads_w[op], w), (grads_b[op], b)] = optimizer.compute_gradients(loss, [w, b])

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

            grads_w[op] = grads_w[op] * mask_grads_w[op]
            grads_b[op] = grads_b[op] * mask_grads_b[op]

            with tf.variable_scope(TF_WEIGHTS) as child_scope:
                w_vel = tf.get_variable(TF_POOL_MOMENTUM)
            with tf.variable_scope(TF_BIAS) as child_scope:
                b_vel = tf.get_variable(TF_POOL_MOMENTUM)

            vel_update_ops.append(
                tf.assign(w_vel, research_parameters['pool_momentum'] * w_vel + grads_w[op]))
            vel_update_ops.append(
                tf.assign(b_vel, research_parameters['pool_momentum'] * b_vel + grads_b[op]))

            grad_ops.append(optimizer.apply_gradients([(w_vel * learning_rate, w), (b_vel * learning_rate, b)]))

    next_op = None
    for tmp_op in cnn_ops[cnn_ops.index(op) + 1:]:
        if 'conv' in tmp_op or 'fulcon' in tmp_op:
            next_op = tmp_op
            break
    logger.debug('Next conv op: %s', next_op)

    if 'conv' in next_op:
        with tf.variable_scope(next_op, reuse=True) as scope:
            w = tf.get_variable(TF_WEIGHTS)
            [(grads_w[next_op], w)] = optimizer.compute_gradients(loss, [w])
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
            grads_w[next_op] = grads_w[next_op] * mask_grads_w[next_op]

            with tf.variable_scope(TF_WEIGHTS) as child_scope:
                pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)
            vel_update_ops.append(
                tf.assign(pool_w_vel, research_parameters['pool_momentum'] * pool_w_vel + grads_w[next_op]))

            grad_ops.append(optimizer.apply_gradients([(pool_w_vel * learning_rate, w)]))

    elif 'fulcon' in next_op:
        with tf.variable_scope(next_op, reuse=True) as scope:
            w = tf.get_variable(TF_WEIGHTS)

            [(grads_w[next_op], w)] = optimizer.compute_gradients(loss, [w])
            logger.debug('Applying gradients for %s', next_op)
            logger.debug('\tAnd filter IDs: %s', filter_indices_to_replace)

            mask_grads_w[next_op] = tf.scatter_nd(
                tf.reshape(filter_indices_to_replace, [-1, 1]),
                tf.ones(shape=[replace_amnt, tf_cnn_hyperparameters[next_op][TF_FC_WEIGHT_OUT_STR]],
                        dtype=tf.float32),
                shape=[tf_cnn_hyperparameters[next_op][TF_FC_WEIGHT_IN_STR],
                       tf_cnn_hyperparameters[next_op][TF_FC_WEIGHT_OUT_STR]]
            )

            grads_w[next_op] = grads_w[next_op] * mask_grads_w[next_op]
            with tf.variable_scope(TF_WEIGHTS) as child_scope:
                pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)

            vel_update_ops.append(
                tf.assign(pool_w_vel,
                          research_parameters['pool_momentum'] * pool_w_vel + grads_w[next_op]))

            grad_ops.append(optimizer.apply_gradients([(pool_w_vel * learning_rate, w)]))

    return grad_ops, vel_update_ops