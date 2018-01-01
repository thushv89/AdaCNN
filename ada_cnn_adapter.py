import tensorflow as tf
import constants
import numpy as np
import logging
import sys


TF_WEIGHTS = constants.TF_WEIGHTS
TF_BIAS = constants.TF_BIAS
TF_TRAIN_MOMENTUM = constants.TF_TRAIN_MOMENTUM
TF_POOL_MOMENTUM = constants.TF_POOL_MOMENTUM

research_parameters = None

TF_FC_WEIGHT_IN_STR = constants.TF_FC_WEIGHT_IN_STR
TF_FC_WEIGHT_OUT_STR = constants.TF_FC_WEIGHT_OUT_STR
TF_CONV_WEIGHT_SHAPE_STR = constants.TF_CONV_WEIGHT_SHAPE_STR
final_2d_width = None
cnn_hyperparameters, cnn_ops = None, None
logger = None

def set_from_main(research_params,final_2d_w,ops,hyps, logging_level, logging_format):
    global research_parameters,final_2d_width, logger
    global cnn_hyperparameters, cnn_ops
    research_parameters = research_params
    final_2d_width = final_2d_w

    cnn_hyperparameters = hyps
    cnn_ops = ops

    logger = logging.getLogger('cnn_adapter_functions')
    logger.setLevel(logging_level)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(logging_format))
    console.setLevel(logging_level)
    logger.addHandler(console)


def add_with_action(
        op, tf_action_info, tf_weights_this, tf_bias_this,
        tf_weights_next, tf_wvelocity_this,
        tf_bvelocity_this, tf_wvelocity_next, tf_replicative_factor_vec,
        tf_act_this, tf_age_weights_this, tf_age_bias_this, tf_age_weights_next
):
    global cnn_hyperparameters, cnn_ops
    global logger

    age_epsilon = 0.01
    max_age = 2.0
    first_fc = 'fulcon_out' if 'fulcon_0' not in cnn_ops else 'fulcon_0'
    update_ops = []

    # find the id of the last conv operation of the net
    for tmp_op in reversed(cnn_ops):
        if 'conv' in tmp_op:
            last_conv_id = tmp_op
            break

    logger.debug('Running action add for op %s', op)

    amount_to_add = tf_action_info[2]  # amount of filters to add
    assert 'conv' in op

    # updating velocity vectors
    with tf.variable_scope(op,reuse=True) as scope:
        w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)
        act = tf.get_variable(constants.TF_ACTIVAIONS_STR)

        with tf.variable_scope(TF_WEIGHTS,reuse=True) as child_scope:
            w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
            pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)
        with tf.variable_scope(TF_BIAS, reuse=True) as child_scope:
            b_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
            pool_b_vel = tf.get_variable(TF_POOL_MOMENTUM)

        # variables containing age of each parameter
        with tf.variable_scope('age', reuse=True):
            age_w, age_b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)

        # calculating new weights
        tf_reshaped_replicative_factor_vec = tf.reshape(tf_replicative_factor_vec, [1, 1, 1, -1])
        tf_new_weights = tf.div(tf.concat(axis=3, values=[w, tf_weights_this]),tf_reshaped_replicative_factor_vec)
        tf_new_biases = tf.div(tf.concat(axis=0, values=[b, tf_bias_this]),tf_replicative_factor_vec)
        tf_new_act = tf.concat(axis=0, values=[act,tf_act_this])

        tf_new_age_w = tf.concat(axis=3, values=[age_w+age_epsilon, tf_age_weights_this])
        tf_new_age_w = tf.clip_by_value(tf_new_age_w, 1.0, max_age)

        tf_new_age_b = tf.concat(axis=0, values=[age_b+age_epsilon, tf_age_bias_this])
        tf_new_age_b = tf.clip_by_value(tf_new_age_b, 1.0, max_age)

        if research_parameters['optimizer'] == 'Momentum':
            new_weight_vel = tf.concat(axis=3, values=[w_vel, tf_wvelocity_this])
            new_bias_vel = tf.concat(axis=0, values=[b_vel, tf_bvelocity_this])
            new_pool_w_vel = tf.concat(axis=3, values=[pool_w_vel, tf_wvelocity_this])
            new_pool_b_vel = tf.concat(axis=0, values=[pool_b_vel, tf_bvelocity_this])

            update_ops.append(tf.assign(w_vel, new_weight_vel, validate_shape=False))
            update_ops.append(tf.assign(b_vel, new_bias_vel, validate_shape=False))
            update_ops.append(tf.assign(pool_w_vel, new_pool_w_vel, validate_shape=False))
            update_ops.append(tf.assign(pool_b_vel, new_pool_b_vel, validate_shape=False))

        update_ops.append(tf.assign(w, tf_new_weights, validate_shape=False))
        update_ops.append(tf.assign(b, tf_new_biases, validate_shape=False))
        update_ops.append(tf.assign(act, tf_new_act, validate_shape=False))
        update_ops.append(tf.assign(age_w, tf_new_age_w, validate_shape=False))
        update_ops.append(tf.assign(age_b, tf_new_age_b, validate_shape=False))

    # ================ Changes to next_op ===============
    # Very last convolutional layer
    # this is different from other layers
    # as a change in this require changes to FC layer
    if op == last_conv_id:
        # change FC layer
        # the reshaping is required because our placeholder for weights_next is Rank 4
        with tf.variable_scope(first_fc,reuse=True) as scope:
            w = tf.get_variable(TF_WEIGHTS)
            with tf.variable_scope(TF_WEIGHTS, reuse=True):
                w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
                pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)

            tf_reshaped_replicative_factor_vec = tf.reshape(tf_replicative_factor_vec, [-1,1])
            tf_reshaped_replicative_factor_vec = tf.tile(tf_reshaped_replicative_factor_vec,[1,final_2d_width*final_2d_width])
            tf_reshaped_replicative_factor_vec = tf.reshape(tf_reshaped_replicative_factor_vec,[-1])
            tf_reshaped_replicative_factor_vec = tf.reshape(tf_reshaped_replicative_factor_vec,[-1,1])

            # variables containing age of each parameter
            with tf.variable_scope('age', reuse=True):
                age_w = tf.get_variable(TF_WEIGHTS)

            tf_weights_next = tf.squeeze(tf_weights_next)
            tf_age_weights_next = tf.squeeze(tf_age_weights_next)

            tf_new_weights = tf.div(tf.concat(axis=0, values=[w, tf_weights_next]),tf_reshaped_replicative_factor_vec)
            tf_new_age_weights = tf.concat(axis=0, values=[age_w+age_epsilon, tf_age_weights_next])
            tf_new_age_weights = tf.clip_by_value(tf_new_age_weights, 1.0, max_age)

            # updating velocity vectors
            if research_parameters['optimizer'] == 'Momentum':
                tf_wvelocity_next = tf.squeeze(tf_wvelocity_next)
                new_weight_vel = tf.concat(axis=0, values=[w_vel, tf_wvelocity_next])
                new_pool_w_vel = tf.concat(axis=0, values=[pool_w_vel, tf_wvelocity_next])
                update_ops.append(tf.assign(w_vel, new_weight_vel, validate_shape=False))
                update_ops.append(tf.assign(pool_w_vel, new_pool_w_vel, validate_shape=False))

            update_ops.append(tf.assign(w, tf_new_weights, validate_shape=False))
            update_ops.append(tf.assign(age_w, tf_new_age_weights, validate_shape=False))

    else:

        # change in hyperparameter of next conv op
        next_conv_op = [tmp_op for tmp_op in cnn_ops[cnn_ops.index(op) + 1:] if 'conv' in tmp_op][0]
        assert op != next_conv_op

        # change only the weights in next conv_op
        with tf.variable_scope(next_conv_op, reuse=True) as scope:
            w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)
            with tf.variable_scope(TF_WEIGHTS, reuse=True):
                w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
                pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)

            # variables containing age of each parameter
            with tf.variable_scope('age', reuse=True):
                age_w = tf.get_variable(TF_WEIGHTS)

            tf_reshaped_replicative_factor_vec = tf.reshape(tf_replicative_factor_vec, [1, 1, -1, 1])
            tf_new_weights = tf.div(tf.concat(axis=2, values=[w, tf_weights_next]),tf_reshaped_replicative_factor_vec)
            tf_new_age_weights = tf.concat(axis=2, values=[age_w+age_epsilon, tf_age_weights_next])
            tf_new_age_weights = tf.clip_by_value(tf_new_age_weights, 1.0, max_age)

            if research_parameters['optimizer'] == 'Momentum':
                new_weight_vel = tf.concat(axis=2, values=[w_vel, tf_wvelocity_next])
                new_pool_w_vel = tf.concat(axis=2, values=[pool_w_vel, tf_wvelocity_next])
                update_ops.append(tf.assign(w_vel, new_weight_vel, validate_shape=False))
                update_ops.append(tf.assign(pool_w_vel, new_pool_w_vel, validate_shape=False))

            update_ops.append(tf.assign(w, tf_new_weights, validate_shape=False))
            update_ops.append(tf.assign(age_w, tf_new_age_weights, validate_shape=False))

    return update_ops


def add_to_fulcon_with_action(
        op, tf_action_info, tf_fulcon_weights_this, tf_fulcon_bias_this,
        tf_fulcon_weights_next, tf_fulcon_wvelocity_this,
        tf_fulcon_bvelocity_this, tf_fulcon_wvelocity_next, tf_replicative_factor_vec,
        tf_act_this, tf_age_fulcon_weights_this, tf_age_fulcon_bias_this,
        tf_age_fulcon_weights_next
):
    global cnn_hyperparameters, cnn_ops
    global logger

    first_fc = 'fulcon_out' if 'fulcon_0' not in cnn_ops else 'fulcon_0'
    update_ops = []
    age_epsilon = 0.01
    max_age=2.0
    # find the id of the last conv operation of the net
    next_fulcon_id = None
    for tmp_op in reversed(cnn_ops):
        if 'fulcon' in tmp_op:
            if op==tmp_op:
                break
            next_fulcon_id = tmp_op

    logger.debug('Running action add for op %s', op)

    amount_to_add = tf_action_info[2]  # amount of filters to add
    assert 'fulcon' in op

    # updating velocity vectors
    with tf.variable_scope(op, reuse=True) as scope:
        w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)
        act = tf.get_variable(constants.TF_ACTIVAIONS_STR)

        with tf.variable_scope(TF_WEIGHTS, reuse=True) as child_scope:
            w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
            pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)
        with tf.variable_scope(TF_BIAS, reuse=True) as child_scope:
            b_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
            pool_b_vel = tf.get_variable(TF_POOL_MOMENTUM)

        with tf.variable_scope('age', reuse=True):
            age_w, age_b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)

        # calculating new weights
        tf_reshaped_replicative_factor_vec = tf.reshape(tf_replicative_factor_vec, [1, -1])
        tf_new_weights = tf.div(
            tf.concat(axis=1, values=[w, tf.squeeze(tf_fulcon_weights_this)]),
            tf_reshaped_replicative_factor_vec)

        tf_new_biases = tf.div(tf.concat(axis=0, values=[b, tf_fulcon_bias_this]),tf_replicative_factor_vec)
        tf_new_act = tf.concat(axis=0, values=[act,tf_act_this])

        tf_age_new_weights = tf.concat(axis=1, values=[age_w+age_epsilon, tf.squeeze(tf_age_fulcon_weights_this)])
        tf_age_new_weights = tf.clip_by_value(tf_age_new_weights, 1.0, max_age)

        tf_age_new_biases = tf.concat(axis=0, values=[b+age_epsilon, tf_age_fulcon_bias_this])
        tf_age_new_biases = tf.clip_by_value(tf_age_new_biases, 1.0, max_age)

        if research_parameters['optimizer'] == 'Momentum':

            new_weight_vel = tf.concat(axis=1, values=[w_vel, tf.squeeze(tf_fulcon_wvelocity_this)])
            new_bias_vel = tf.concat(axis=0, values=[b_vel, tf_fulcon_bvelocity_this])
            new_pool_w_vel = tf.concat(axis=1, values=[pool_w_vel, tf.squeeze(tf_fulcon_wvelocity_this)])
            new_pool_b_vel = tf.concat(axis=0, values=[pool_b_vel, tf_fulcon_bvelocity_this])

            update_ops.append(tf.assign(w_vel, new_weight_vel, validate_shape=False))
            update_ops.append(tf.assign(b_vel, new_bias_vel, validate_shape=False))
            update_ops.append(tf.assign(pool_w_vel, new_pool_w_vel, validate_shape=False))
            update_ops.append(tf.assign(pool_b_vel, new_pool_b_vel, validate_shape=False))

        update_ops.append(tf.assign(w, tf_new_weights, validate_shape=False))
        update_ops.append(tf.assign(b, tf_new_biases, validate_shape=False))
        update_ops.append(tf.assign(act, tf_new_act, validate_shape=False))
        update_ops.append(tf.assign(age_w, tf_age_new_weights, validate_shape=False))
        update_ops.append(tf.assign(age_b, tf_age_new_biases, validate_shape=False))

    # ================ Changes to next_op ===============
    # change FC layer
    # the reshaping is required because our placeholder for weights_next is Rank 4
    with tf.variable_scope(next_fulcon_id, reuse=True):
        w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)
        with tf.variable_scope(TF_WEIGHTS, reuse=True) as child_scope:
            w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
            pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)

        with tf.variable_scope('age', reuse=True):
            age_w = tf.get_variable(TF_WEIGHTS)

        tf_reshaped_replicative_factor_vec = tf.reshape(tf_replicative_factor_vec,[-1,1])
        tf_weights_next = tf.squeeze(tf_fulcon_weights_next)
        tf_new_weights = tf.div(tf.concat(axis=0, values=[w, tf_weights_next]),tf_reshaped_replicative_factor_vec)

        tf_age_new_weights = tf.concat(axis=0, values=[age_w + age_epsilon, tf.squeeze(tf_age_fulcon_weights_next)])
        tf_age_new_weights = tf.clip_by_value(tf_age_new_weights, 1.0, max_age)

        # updating velocity vectors
        tf_wvelocity_next = tf.squeeze(tf_fulcon_wvelocity_next)
        new_w_vel = tf.concat(axis=0, values=[w_vel, tf.squeeze(tf_wvelocity_next)])
        new_pool_w_vel = tf.concat(axis=0, values=[pool_w_vel, tf.squeeze(tf_wvelocity_next)])
        update_ops.append(tf.assign(w_vel, new_w_vel, validate_shape=False))
        update_ops.append(tf.assign(pool_w_vel, new_pool_w_vel, validate_shape=False))

        update_ops.append(tf.assign(w, tf_new_weights, validate_shape=False))
        update_ops.append(tf.assign(age_w, tf_age_new_weights,validate_shape=False))
    return update_ops



def remove_with_action(op, tf_action_info, tf_cnn_hyperparameters, tf_indices_to_rm):
    global cnn_hyperparameters, cnn_ops
    global logger

    age_epsilon = 0.02
    max_age = 2.0
    first_fc = 'fulcon_out' if 'fulcon_0' not in cnn_ops else 'fulcon_0'
    update_ops = []

    for tmp_op in reversed(cnn_ops):
        if 'conv' in tmp_op:
            last_conv_id = tmp_op
            break

    # this is trickier than adding weights
    # We remove the given number of filters
    # which have the least rolling mean activation averaged over whole map
    amount_to_rmv = tf_action_info[2]
    assert 'conv' in op

    with tf.variable_scope(op, reuse=True) as scope:

        tf_indices_to_rm = tf.reshape(tf_indices_to_rm, shape=[amount_to_rmv, 1],
                                      name='indices_to_rm')
        tf_rm_ind_scatter = tf.scatter_nd(tf_indices_to_rm, tf.ones(shape=[amount_to_rmv], dtype=tf.int32),
                                          shape=[tf_cnn_hyperparameters[op][TF_CONV_WEIGHT_SHAPE_STR][3]])

        tf_indices_to_keep_boolean = tf.equal(tf_rm_ind_scatter, tf.constant(0, dtype=tf.int32))
        tf_indices_to_keep = tf.reshape(tf.where(tf_indices_to_keep_boolean), shape=[-1, 1], name='indices_to_keep')

        # currently no way to generally slice using gather
        # need to do a transoformation to do this.
        # change both weights and biase in the current op
        w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)
        act = tf.get_variable(constants.TF_ACTIVAIONS_STR)

        with tf.variable_scope(TF_WEIGHTS, reuse=True):
            w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
            pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)
        with tf.variable_scope(TF_BIAS, reuse=True):
            b_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
            pool_b_vel = tf.get_variable(TF_POOL_MOMENTUM)

        with tf.variable_scope('age', reuse=True):
            age_w, age_b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)

        tf_new_weights = tf.transpose(w, [3, 0, 1, 2])
        tf_new_weights = tf.gather_nd(tf_new_weights, tf_indices_to_keep)
        tf_new_weights = tf.transpose(tf_new_weights, [1, 2, 3, 0], name='new_weights')

        tf_age_new_weights = tf.transpose(age_w, [3,0,1,2])
        tf_age_new_weights = tf.gather_nd(tf_age_new_weights, tf_indices_to_keep)
        tf_age_new_weights = tf.transpose(tf_age_new_weights, [1,2,3,0],name='new_age_weights')

        # make sure that all the elements are >= 1
        tf_age_new_weights = tf_age_new_weights - age_epsilon
        tf_age_new_weights = tf.clip_by_value(tf_age_new_weights, 1.0, max_age)

        update_ops.append(tf.assign(w, tf_new_weights, validate_shape=False))
        update_ops.append(tf.assign(age_w, tf_age_new_weights, validate_shape=False))

        tf_new_biases = tf.reshape(tf.gather(b, tf_indices_to_keep),[-1])
        tf_age_new_biases = tf.reshape(tf.gather(age_b, tf_indices_to_keep),[-1])

        # make sure that all the elements are >= 1
        tf_age_new_biases = tf_age_new_biases - age_epsilon
        tf_age_new_biases = tf.clip_by_value(tf_age_new_biases,1.0,max_age)

        update_ops.append(tf.assign(b, tf_new_biases, validate_shape=False))
        update_ops.append(tf.assign(age_b, tf_age_new_biases, validate_shape=False))

        tf_new_acts = tf.reshape(tf.gather(act, tf_indices_to_keep),[-1])
        update_ops.append(tf.assign(act, tf_new_acts, validate_shape=False))

        if research_parameters['optimizer'] == 'Momentum':
            new_weight_vel = tf.transpose(w_vel, [3, 0, 1, 2])
            new_weight_vel = tf.gather_nd(new_weight_vel, tf_indices_to_keep)
            new_weight_vel = tf.transpose(new_weight_vel, [1, 2, 3, 0])

            new_pool_w_vel = tf.transpose(pool_w_vel, [3, 0, 1, 2])
            new_pool_w_vel = tf.gather_nd(new_pool_w_vel, tf_indices_to_keep)
            new_pool_w_vel = tf.transpose(new_pool_w_vel, [1, 2, 3, 0])

            new_bias_vel = tf.reshape(tf.gather(b_vel, tf_indices_to_keep), [-1])
            new_pool_b_vel = tf.reshape(tf.gather(pool_b_vel, tf_indices_to_keep), [-1])

            update_ops.append(tf.assign(w_vel, new_weight_vel, validate_shape=False))
            update_ops.append(tf.assign(pool_w_vel, new_pool_w_vel, validate_shape=False))
            update_ops.append(tf.assign(b_vel, new_bias_vel, validate_shape=False))
            update_ops.append(tf.assign(pool_b_vel, new_pool_b_vel, validate_shape=False))

    # Processing the 1st fully connected layer
    if op == last_conv_id:

        with tf.variable_scope(first_fc, reuse=True) as scope:
            w = tf.get_variable(TF_WEIGHTS)
            with tf.variable_scope(TF_WEIGHTS, reuse=True) as child_scope:
                w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
                pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)

            with tf.variable_scope('age', reuse=True):
                age_w = tf.get_variable(TF_WEIGHTS)

            # Making fulcon indices from the last convolution filter indices to be removed
            # by having for each index i, (i*final_w*final_w)-> ((i+1)*final_w*final_w)
            tf_offset = tf.cast(tf.reshape(tf.range(0, final_2d_width*final_2d_width),[1,-1]),dtype=tf.int64)
            tf_fulcon_indices_to_keep = tf.reshape(tf_indices_to_keep,[-1,1])
            tf_fulcon_indices_to_keep = tf.tile(tf_fulcon_indices_to_keep,[1,final_2d_width*final_2d_width])
            tf_fulcon_indices_to_keep = tf_fulcon_indices_to_keep + tf_offset
            tf_fulcon_indices_to_keep = tf.reshape(tf_fulcon_indices_to_keep,[-1,1])

            tf_new_weights = tf.gather_nd(w, tf_fulcon_indices_to_keep)

            tf_age_new_weights = tf.gather_nd(age_w, tf_fulcon_indices_to_keep)
            tf_age_new_weights = tf_age_new_weights - age_epsilon
            tf_age_new_weights = tf.clip_by_value(tf_age_new_weights,1.0,max_age)

            update_ops.append(tf.assign(w, tf_new_weights, validate_shape=False))
            update_ops.append(tf.assign(age_w, tf_age_new_weights, validate_shape=False))

            if research_parameters['optimizer'] == 'Momentum':

                new_weight_vel = tf.gather_nd(w_vel, tf_fulcon_indices_to_keep)
                new_pool_w_vel = tf.gather_nd(pool_w_vel, tf_fulcon_indices_to_keep)

                update_ops.append(tf.assign(w_vel, new_weight_vel, validate_shape=False))
                update_ops.append(tf.assign(pool_w_vel, new_pool_w_vel, validate_shape=False))

    else:
        # change in hyperparameter of next conv op
        next_conv_op = [tmp_op for tmp_op in cnn_ops[cnn_ops.index(op) + 1:] if 'conv' in tmp_op][0]
        assert op != next_conv_op

        # change only the weights in next conv_op
        with tf.variable_scope(next_conv_op) as scope:
            w = tf.get_variable(TF_WEIGHTS)
            with tf.variable_scope(TF_WEIGHTS) as child_scope:
                w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
                pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)

            with tf.variable_scope('age', reuse=True):
                age_w = tf.get_variable(TF_WEIGHTS)

            tf_new_weights = tf.transpose(w, [2, 0, 1, 3])
            tf_new_weights = tf.gather_nd(tf_new_weights, tf_indices_to_keep)
            tf_new_weights = tf.transpose(tf_new_weights, [1, 2, 0, 3])

            tf_age_new_weights = tf.transpose(age_w, [2,0,1,3])
            tf_age_new_weights = tf.gather_nd(tf_age_new_weights, tf_indices_to_keep)
            tf_age_new_weights = tf.transpose(tf_age_new_weights, [1,2,0,3])

            tf_age_new_weights = tf_age_new_weights - age_epsilon
            tf_age_new_weights = tf.clip_by_value(tf_age_new_weights, 1.0, max_age)

            update_ops.append(tf.assign(w, tf_new_weights, validate_shape=False))
            update_ops.append(tf.assign(age_w, tf_age_new_weights, validate_shape=False))
            if research_parameters['optimizer'] == 'Momentum':
                new_weight_vel = tf.transpose(w_vel, [2, 0, 1, 3])
                new_weight_vel = tf.gather_nd(new_weight_vel, tf_indices_to_keep)
                new_weight_vel = tf.transpose(new_weight_vel, [1, 2, 0, 3])

                new_pool_w_vel = tf.transpose(pool_w_vel, [2, 0, 1, 3])
                new_pool_w_vel = tf.gather_nd(new_pool_w_vel, tf_indices_to_keep)
                new_pool_w_vel = tf.transpose(new_pool_w_vel, [1, 2, 0, 3])

                update_ops.append(tf.assign(w_vel, new_weight_vel, validate_shape=False))
                update_ops.append(tf.assign(pool_w_vel, new_pool_w_vel, validate_shape=False))

    return update_ops, tf_indices_to_rm


def remove_from_fulcon(op, tf_action_info, tf_cnn_hyperparameters, tf_indices_to_rm):
    global cnn_hyperparameters, cnn_ops
    global logger

    age_epsilon = 0.02
    max_age = 2.0
    update_ops = []

    for tmp_op in reversed(cnn_ops):
        if 'conv' in tmp_op:
            last_conv_id = tmp_op
            break

    # this is trickier than adding weights
    # We remove the given number of filters
    # which have the least rolling mean activation averaged over whole map
    amount_to_rmv = tf_action_info[2]
    assert 'fulcon' in op

    with tf.variable_scope(op) as scope:

        tf_indices_to_rm = tf.reshape(tf.slice(tf_indices_to_rm, [0], [amount_to_rmv]), shape=[amount_to_rmv, 1],
                                      name='indices_to_rm')
        tf_rm_ind_scatter = tf.scatter_nd(tf_indices_to_rm, tf.ones(shape=[amount_to_rmv], dtype=tf.int32),
                                          shape=[tf_cnn_hyperparameters[op][TF_FC_WEIGHT_OUT_STR]])

        tf_indices_to_keep_boolean = tf.equal(tf_rm_ind_scatter, tf.constant(0, dtype=tf.int32))
        tf_indices_to_keep = tf.reshape(tf.where(tf_indices_to_keep_boolean), shape=[-1, 1], name='indices_to_keep')

        # currently no way to generally slice using gather
        # need to do a transoformation to do this.
        # change both weights and biase in the current op
        w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)
        act = tf.get_variable(constants.TF_ACTIVAIONS_STR)

        with tf.variable_scope(TF_WEIGHTS) as child_scope:
            w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
            pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)
        with tf.variable_scope(TF_BIAS) as child_scope:
            b_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
            pool_b_vel = tf.get_variable(TF_POOL_MOMENTUM)

        with tf.variable_scope('age', reuse=True):
            age_w, age_b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)

        tf_new_weights = tf.transpose(w)
        tf_new_weights = tf.gather_nd(tf_new_weights, tf_indices_to_keep)
        tf_new_weights = tf.transpose(tf_new_weights, name='new_weights')

        tf_age_new_weights = tf.transpose(age_w)
        tf_age_new_weights = tf.gather_nd(tf_age_new_weights, tf_indices_to_keep)
        tf_age_new_weights = tf.transpose(tf_age_new_weights, name='new_age_weights')

        tf_age_new_weights = tf_age_new_weights - age_epsilon
        tf_age_new_weights = tf.clip_by_value(tf_age_new_weights, 1.0, max_age)

        update_ops.append(tf.assign(w, tf_new_weights, validate_shape=False))
        update_ops.append(tf.assign(age_w, tf_age_new_weights, validate_shape=False))

        tf_new_biases = tf.reshape(tf.gather(b, tf_indices_to_keep), shape=[-1], name='new_bias')

        tf_age_new_biases = tf.reshape(tf.gather(age_b, tf_indices_to_keep), shape=[-1], name = 'new_age_bias')
        tf_age_new_biases = tf_age_new_biases - age_epsilon
        tf_age_new_biases = tf.clip_by_value(tf_age_new_biases,1.0,max_age)

        update_ops.append(tf.assign(b, tf_new_biases, validate_shape=False))
        update_ops.append(tf.assign(age_b, tf_age_new_biases, validate_shape=False))

        tf_new_acts = tf.reshape(tf.gather(act, tf_indices_to_keep),[-1])
        update_ops.append(tf.assign(act,tf_new_acts, validate_shape=False))

        new_weight_vel = tf.transpose(w_vel)
        new_weight_vel = tf.gather_nd(new_weight_vel, tf.reshape(tf_indices_to_keep,[-1,1]))
        new_weight_vel = tf.transpose(new_weight_vel)

        new_pool_w_vel = tf.transpose(pool_w_vel)
        new_pool_w_vel = tf.gather_nd(new_pool_w_vel, tf_indices_to_keep)
        new_pool_w_vel = tf.transpose(new_pool_w_vel)

        new_bias_vel = tf.reshape(tf.gather(b_vel, tf_indices_to_keep), [-1])
        new_pool_b_vel = tf.reshape(tf.gather(pool_b_vel, tf_indices_to_keep), [-1])

        update_ops.append(tf.assign(w_vel, new_weight_vel, validate_shape=False))
        update_ops.append(tf.assign(pool_w_vel, new_pool_w_vel, validate_shape=False))
        update_ops.append(tf.assign(b_vel, new_bias_vel, validate_shape=False))
        update_ops.append(tf.assign(pool_b_vel, new_pool_b_vel, validate_shape=False))


    # change in hyperparameter of next conv op
    next_fulcon_op = [tmp_op for tmp_op in cnn_ops[cnn_ops.index(op) + 1:] if 'fulcon' in tmp_op][0]
    assert op != next_fulcon_op

    # change only the weights in next conv_op
    with tf.variable_scope(next_fulcon_op) as scope:
        w = tf.get_variable(TF_WEIGHTS)
        with tf.variable_scope(TF_WEIGHTS) as child_scope:
            w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
            pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)

        with tf.variable_scope('age', reuse=True):
            age_w = tf.get_variable(TF_WEIGHTS)

        tf_new_weights = tf.identity(w)
        tf_new_weights = tf.gather_nd(tf_new_weights, tf_indices_to_keep)

        tf_age_new_weights = tf.gather_nd(tf_new_weights, tf_indices_to_keep)
        tf_age_new_weights = tf_age_new_weights - age_epsilon
        tf_age_new_weights = tf.clip_by_value(tf_age_new_weights,1.0,max_age)

        update_ops.append(tf.assign(w, tf_new_weights, validate_shape=False))
        update_ops.append(tf.assign(age_w, tf_age_new_weights, validate_shape=False))

        new_weight_vel = tf.gather_nd(w_vel, tf_indices_to_keep)

        new_pool_w_vel = tf.gather_nd(pool_w_vel, tf_indices_to_keep)

        update_ops.append(tf.assign(w_vel, new_weight_vel, validate_shape=False))
        update_ops.append(tf.assign(pool_w_vel, new_pool_w_vel, validate_shape=False))

    return update_ops, tf_indices_to_rm


def update_tf_hyperparameters(op,tf_weight_shape,tf_in_size, tf_out_size):
    global cnn_ops, cnn_hyperparameters
    update_ops = []
    if 'conv' in op:
        with tf.variable_scope(op,reuse=True):
            update_ops.append(tf.assign(tf.get_variable(TF_CONV_WEIGHT_SHAPE_STR,dtype=tf.int32),tf_weight_shape))
    if 'fulcon' in op:
        with tf.variable_scope(op,reuse=True):
            update_ops.append(tf.assign(tf.get_variable(TF_FC_WEIGHT_IN_STR,dtype=tf.int32),tf_in_size))
            update_ops.append(tf.assign(tf.get_variable(TF_FC_WEIGHT_OUT_STR, dtype=tf.int32), tf_out_size))
    return update_ops

