import tensorflow as tf

def gradients(optimizer, loss, global_step, learning_rate):
    # grad_and_vars [(grads_w,w),(grads_b,b)]
    grad_and_vars = optimizer.compute_gradients(loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                        scope=TF_GLOBAL_SCOPE))
    return grad_and_vars


def update_train_momentum_velocity(grads_and_vars):
    vel_update_ops = []
    # update velocity vector

    for (g, v) in grads_and_vars:
        var_name = v.name.split(':')[0]

        with tf.variable_scope(var_name, reuse=True) as scope:
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
        var_name = v.name.split(':')[0]

        with tf.variable_scope(var_name, reuse=True) as scope:
            vel = tf.get_variable(TF_POOL_MOMENTUM)

            vel_update_ops.append(
                tf.assign(vel,
                          research_parameters['pool_momentum'] * vel + g)
            )
    return vel_update_ops


def apply_gradient_with_momentum(optimizer, learning_rate, global_step):
    grads_and_vars = []
    # for each trainable variable
    if decay_learning_rate:
        learning_rate = tf.maximum(min_learning_rate,
                                   tf.train.exponential_decay(learning_rate, global_step, decay_steps=1,
                                                              decay_rate=decay_rate, staircase=True))
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=TF_GLOBAL_SCOPE):
        with tf.variable_scope(v.name.split(':')[0], reuse=True):
            vel = tf.get_variable(TF_TRAIN_MOMENTUM)
            grads_and_vars.append((vel * learning_rate, v))

    return optimizer.apply_gradients(grads_and_vars)


def apply_gradient_with_pool_momentum(optimizer, learning_rate, global_step):
    grads_and_vars = []
    # for each trainable variable
    if decay_learning_rate:
        learning_rate = tf.maximum(min_learning_rate,
                                   tf.train.exponential_decay(learning_rate, global_step, decay_steps=1,
                                                              decay_rate=decay_rate, staircase=True))
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=TF_GLOBAL_SCOPE):
        with tf.variable_scope(v.name.split(':')[0], reuse=True):
            vel = tf.get_variable(TF_POOL_MOMENTUM)
            grads_and_vars.append((vel * learning_rate, v))

    return optimizer.apply_gradients(grads_and_vars)


def optimize_with_momentum(optimizer, loss, global_step, learning_rate):
    vel_update_ops, grad_update_ops = [], []

    if decay_learning_rate:
        learning_rate = tf.maximum(min_learning_rate,
                                   tf.train.exponential_decay(learning_rate, global_step, decay_steps=1,
                                                              decay_rate=decay_rate, staircase=True))

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

    if decay_learning_rate:
        learning_rate = tf.maximum(min_learning_rate,
                                   tf.train.exponential_decay(learning_rate, global_step, decay_steps=1,
                                                              decay_rate=decay_rate, staircase=True))
    else:
        learning_rate = tf.constant(start_lr, dtype=tf.float32, name='learning_rate')

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

            grad_ops.append(optimizer.apply_gradients([(pool_w_vel * learning_rate, w)]))

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

            grad_ops.append(optimizer.apply_gradients([(pool_w_vel * learning_rate, w)]))

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
    learning_rate = tf.constant(start_lr, dtype=tf.float32, name='learning_rate')

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