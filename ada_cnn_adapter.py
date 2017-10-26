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
        tf_bvelocity_this, tf_wvelocity_next, tf_replicative_factor_vec, layer_feature_map_sizes
):
    global cnn_hyperparameters, cnn_ops
    global logger

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
    with tf.variable_scope(op) as scope:
        w = tf.get_variable(TF_WEIGHTS)

        with tf.variable_scope(constants.TF_BN_SCOPE):
            mu, sigma = tf.get_variable(constants.TF_BN_POP_MU_STR), tf.get_variable(constants.TF_BN_POP_SIGMA_STR)
            gamma, beta = tf.get_variable(constants.TF_BN_GAMMA_STR), tf.get_variable(constants.TF_BN_BETA_STR)

        with tf.variable_scope(TF_WEIGHTS) as child_scope:
            w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
            pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)

        # calculating new weights
        tf_reshaped_replicative_factor_vec = tf.reshape(tf_replicative_factor_vec, [1, 1, 1, -1])
        tf_new_weights = tf.div(tf.concat(axis=3, values=[w, tf_weights_this]),tf_reshaped_replicative_factor_vec)

        if research_parameters['optimizer'] == 'Momentum':
            new_weight_vel = tf.concat(axis=3, values=[w_vel, tf_wvelocity_this])

            new_pool_w_vel = tf.concat(axis=3, values=[pool_w_vel, tf_wvelocity_this])


            update_ops.append(tf.assign(w_vel, new_weight_vel, validate_shape=False))

            update_ops.append(tf.assign(pool_w_vel, new_pool_w_vel, validate_shape=False))


        update_ops.append(tf.assign(w, tf_new_weights, validate_shape=False))

        # Batch normalization, mu sigma gamma and beta updates
        update_ops.append(
            tf.assign(mu, tf.concat(
                values=[mu, tf.zeros(shape=[layer_feature_map_sizes[op],layer_feature_map_sizes[op],amount_to_add],
                                     dtype=tf.float32)],axis=2),
                      validate_shape=False)
        )
        update_ops.append(
            tf.assign(sigma, tf.concat(
                values=[sigma, tf.ones(shape=[layer_feature_map_sizes[op], layer_feature_map_sizes[op], amount_to_add],
                                     dtype=tf.float32)], axis=2),
                      validate_shape=False)
        )
        update_ops.append(
            tf.assign(gamma, tf.concat(
                values=[gamma, tf.ones(shape=[amount_to_add],
                                     dtype=tf.float32)], axis=0),
                      validate_shape=False)
        )
        update_ops.append(
            tf.assign(beta, tf.concat(
                values=[beta, tf.zeros(shape=[amount_to_add],
                                     dtype=tf.float32)], axis=0),
                      validate_shape=False)
        )

    # ================ Changes to next_op ===============
    # Very last convolutional layer
    # this is different from other layers
    # as a change in this require changes to FC layer
    if op == last_conv_id:
        # change FC layer
        # the reshaping is required because our placeholder for weights_next is Rank 4
        with tf.variable_scope(first_fc) as scope:
            w = tf.get_variable(TF_WEIGHTS)
            with tf.variable_scope(TF_WEIGHTS) as child_scope:
                w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
                pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)

            tf_reshaped_replicative_factor_vec = tf.reshape(tf_replicative_factor_vec, [-1,1])
            tf_reshaped_replicative_factor_vec = tf.tile(tf_reshaped_replicative_factor_vec,[1,final_2d_width*final_2d_width])
            tf_reshaped_replicative_factor_vec = tf.reshape(tf_reshaped_replicative_factor_vec,[-1])
            tf_reshaped_replicative_factor_vec = tf.reshape(tf_reshaped_replicative_factor_vec,[-1,1])

            tf_weights_next = tf.squeeze(tf_weights_next)
            tf_new_weights = tf.div(tf.concat(axis=0, values=[w, tf_weights_next]),tf_reshaped_replicative_factor_vec)

            # updating velocity vectors
            if research_parameters['optimizer'] == 'Momentum':
                tf_wvelocity_next = tf.squeeze(tf_wvelocity_next)
                new_weight_vel = tf.concat(axis=0, values=[w_vel, tf_wvelocity_next])
                new_pool_w_vel = tf.concat(axis=0, values=[pool_w_vel, tf_wvelocity_next])
                update_ops.append(tf.assign(w_vel, new_weight_vel, validate_shape=False))
                update_ops.append(tf.assign(pool_w_vel, new_pool_w_vel, validate_shape=False))

            update_ops.append(tf.assign(w, tf_new_weights, validate_shape=False))

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

            tf_reshaped_replicative_factor_vec = tf.reshape(tf_replicative_factor_vec, [1, 1, -1, 1])
            tf_new_weights = tf.div(tf.concat(axis=2, values=[w, tf_weights_next]),tf_reshaped_replicative_factor_vec)

            if research_parameters['optimizer'] == 'Momentum':
                new_weight_vel = tf.concat(axis=2, values=[w_vel, tf_wvelocity_next])
                new_pool_w_vel = tf.concat(axis=2, values=[pool_w_vel, tf_wvelocity_next])
                update_ops.append(tf.assign(w_vel, new_weight_vel, validate_shape=False))
                update_ops.append(tf.assign(pool_w_vel, new_pool_w_vel, validate_shape=False))

            update_ops.append(tf.assign(w, tf_new_weights, validate_shape=False))

    return update_ops


def add_to_fulcon_with_action(
        op, tf_action_info, tf_fulcon_weights_this, tf_fulcon_bias_this,
        tf_fulcon_weights_next, tf_fulcon_wvelocity_this,
        tf_fulcon_bvelocity_this, tf_fulcon_wvelocity_next, tf_replicative_factor_vec
):
    global cnn_hyperparameters, cnn_ops
    global logger

    first_fc = 'fulcon_out' if 'fulcon_0' not in cnn_ops else 'fulcon_0'
    update_ops = []

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
    with tf.variable_scope(op) as scope:
        w = tf.get_variable(TF_WEIGHTS)

        with tf.variable_scope(constants.TF_BN_SCOPE):
            mu, sigma = tf.get_variable(constants.TF_BN_POP_MU_STR), tf.get_variable(constants.TF_BN_POP_SIGMA_STR)
            gamma, beta = tf.get_variable(constants.TF_BN_GAMMA_STR), tf.get_variable(constants.TF_BN_BETA_STR)

        with tf.variable_scope(TF_WEIGHTS) as child_scope:
            w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
            pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)

        # calculating new weights
        tf_reshaped_replicative_factor_vec = tf.reshape(tf_replicative_factor_vec, [1, -1])
        tf_new_weights = tf.div(tf.concat(axis=1, values=[w, tf.squeeze(tf_fulcon_weights_this)]),tf_reshaped_replicative_factor_vec)
        if op=='fulcon_out':
            with tf.variable_scope(TF_BIAS) as child_scope:
                b_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
                pool_b_vel = tf.get_variable(TF_POOL_MOMENTUM)

            b = tf.get_variable(TF_BIAS)
            tf_new_biases = tf.div(tf.concat(axis=0, values=[b, tf_fulcon_bias_this]),tf_replicative_factor_vec)


        if research_parameters['optimizer'] == 'Momentum':
            new_weight_vel = tf.concat(axis=1, values=[w_vel, tf.squeeze(tf_fulcon_wvelocity_this)])
            new_pool_w_vel = tf.concat(axis=1, values=[pool_w_vel, tf.squeeze(tf_fulcon_wvelocity_this)])
            if op == 'fulcon_out':
                new_bias_vel = tf.concat(axis=0, values=[b_vel, tf_fulcon_bvelocity_this])
                new_pool_b_vel = tf.concat(axis=0, values=[pool_b_vel, tf_fulcon_bvelocity_this])

            update_ops.append(tf.assign(w_vel, new_weight_vel, validate_shape=False))
            update_ops.append(tf.assign(pool_w_vel, new_pool_w_vel, validate_shape=False))

            if op == 'fulcon_out':
                update_ops.append(tf.assign(b_vel, new_bias_vel, validate_shape=False))
                update_ops.append(tf.assign(pool_b_vel, new_pool_b_vel, validate_shape=False))

                update_ops.append(tf.assign(b, tf_new_biases, validate_shape=False))

        update_ops.append(tf.assign(w, tf_new_weights, validate_shape=False))


        # Batch normalization, mu sigma gamma and beta updates
        update_ops.append(
            tf.assign(mu, tf.concat(
                values=[mu, tf.zeros(shape=[amount_to_add],
                                     dtype=tf.float32)], axis=0),
                      validate_shape=False)
        )
        update_ops.append(
            tf.assign(sigma, tf.concat(
                values=[sigma, tf.ones(shape=[amount_to_add],
                                       dtype=tf.float32)], axis=0),
                      validate_shape=False)
        )
        update_ops.append(
            tf.assign(gamma, tf.concat(
                values=[gamma, tf.ones(shape=[amount_to_add],
                                        dtype=tf.float32)], axis=0),
                      validate_shape=False)
        )
        update_ops.append(
            tf.assign(beta, tf.concat(
                values=[beta, tf.zeros(shape=[amount_to_add],
                                       dtype=tf.float32)], axis=0),
                      validate_shape=False)
        )

    # ================ Changes to next_op ===============
    # change FC layer
    # the reshaping is required because our placeholder for weights_next is Rank 4
    with tf.variable_scope(next_fulcon_id) as scope:
        w = tf.get_variable(TF_WEIGHTS)
        with tf.variable_scope(TF_WEIGHTS) as child_scope:
            w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
            pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)

        tf_reshaped_replicative_factor_vec = tf.reshape(tf_replicative_factor_vec,[-1,1])
        tf_weights_next = tf.squeeze(tf_fulcon_weights_next)
        tf_new_weights = tf.div(tf.concat(axis=0, values=[w, tf_weights_next]),tf_reshaped_replicative_factor_vec)

        # updating velocity vectors
        if research_parameters['optimizer'] == 'Momentum':
            tf_wvelocity_next = tf.squeeze(tf_fulcon_wvelocity_next)
            new_weight_vel = tf.concat(axis=0, values=[w_vel, tf.squeeze(tf_wvelocity_next)])
            new_pool_w_vel = tf.concat(axis=0, values=[pool_w_vel, tf.squeeze(tf_wvelocity_next)])
            update_ops.append(tf.assign(w_vel, new_weight_vel, validate_shape=False))
            update_ops.append(tf.assign(pool_w_vel, new_pool_w_vel, validate_shape=False))

        update_ops.append(tf.assign(w, tf_new_weights, validate_shape=False))

    return update_ops

def get_rm_indices_with_distance(op, tf_action_info, tf_cnn_hyperparameters):
    amount_to_rmv = tf_action_info[2]
    with tf.variable_scope(op) as scope:
        w = tf.get_variable(TF_WEIGHTS)  # h x w x in x out
        reshaped_weight = tf.transpose(w, [3, 0, 1, 2])
        reshaped_weight = tf.reshape(w, [tf_cnn_hyperparameters[op]['weights'][3],
                                         tf_cnn_hyperparameters[op]['weights'][0] *
                                         tf_cnn_hyperparameters[op]['weights'][1] *
                                         tf_cnn_hyperparameters[op]['weights'][2]]
                                     )
    cos_sim_weights = tf.matmul(reshaped_weight, tf.transpose(reshaped_weight), name='dot_prod_cos_sim') / tf.matmul(
        tf.sqrt(tf.reduce_sum(reshaped_weight ** 2, axis=1, keep_dims=True)),
        tf.sqrt(tf.transpose(tf.reduce_sum(reshaped_weight ** 2, axis=1, keep_dims=True)))
        , name='norm_cos_sim')

    upper_triang_cos_sim = tf.matrix_band_part(cos_sim_weights, 0, -1, name='upper_triang_cos_sim')
    zero_diag_triang_cos_sim = tf.matrix_set_diag(upper_triang_cos_sim,
                                                  tf.zeros(shape=[tf_cnn_hyperparameters[op]['weights'][3]]),
                                                  name='zero_diag_upper_triangle')
    flattened_cos_sim = tf.reshape(zero_diag_triang_cos_sim, shape=[-1], name='flattend_cos_sim')

    # we are finding top amount_to_rmv + epsilon amount because
    # to avoid k_values = {...,(83,1)(139,94)(139,83),...} like incidents
    # above case will ignore both indices of (139,83) resulting in a reduction < amount_to_rmv
    [high_sim_values, high_sim_indices] = tf.nn.top_k(flattened_cos_sim,
                                                      k=tf.minimum(amount_to_rmv + 10,
                                                                   tf_cnn_hyperparameters[op]['weights'][3]),
                                                      name='top_k_indices')

    tf_indices_to_remove_1 = tf.reshape(tf.mod(high_sim_indices, tf_cnn_hyperparameters[op]['weights'][3]), shape=[-1],
                                        name='mod_indices')
    tf_indices_to_remove_2 = tf.reshape(tf.floor_div(high_sim_indices, tf_cnn_hyperparameters[op]['weights'][3]),
                                        shape=[-1], name='floor_div_indices')
    # concat both mod and floor_div indices
    tf_indices_to_rm = tf.reshape(tf.stack([tf_indices_to_remove_1, tf_indices_to_remove_2], name='all_rm_indices'),
                                  shape=[-1])
    # return both values and indices of unique values (discard indices)
    tf_unique_rm_ind, _ = tf.unique(tf_indices_to_rm, name='unique_rm_indices')

    return tf_unique_rm_ind


def remove_with_action(op, tf_action_info, tf_cnn_hyperparameters, tf_indices_to_rm):
    global cnn_hyperparameters, cnn_ops
    global logger

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

    with tf.variable_scope(op) as scope:

        tf_indices_to_rm = tf.reshape(tf.slice(tf_indices_to_rm, [0], [amount_to_rmv]), shape=[amount_to_rmv, 1],
                                      name='indices_to_rm')
        tf_rm_ind_scatter = tf.scatter_nd(tf_indices_to_rm, tf.ones(shape=[amount_to_rmv], dtype=tf.int32),
                                          shape=[tf_cnn_hyperparameters[op][TF_CONV_WEIGHT_SHAPE_STR][3]])

        tf_indices_to_keep_boolean = tf.equal(tf_rm_ind_scatter, tf.constant(0, dtype=tf.int32))
        tf_indices_to_keep = tf.reshape(tf.where(tf_indices_to_keep_boolean), shape=[-1, 1], name='indices_to_keep')

        # currently no way to generally slice using gather
        # need to do a transoformation to do this.
        # change both weights and biase in the current op
        w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)
        with tf.variable_scope(TF_WEIGHTS) as child_scope:
            w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
            pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)
        with tf.variable_scope(TF_BIAS) as child_scope:
            b_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
            pool_b_vel = tf.get_variable(TF_POOL_MOMENTUM)

        tf_new_weights = tf.transpose(w, [3, 0, 1, 2])
        tf_new_weights = tf.gather_nd(tf_new_weights, tf_indices_to_keep)
        tf_new_weights = tf.transpose(tf_new_weights, [1, 2, 3, 0], name='new_weights')

        update_ops.append(tf.assign(w, tf_new_weights, validate_shape=False))

        tf_new_biases = tf.reshape(tf.gather(b, tf_indices_to_keep), shape=[-1], name='new_bias')
        update_ops.append(tf.assign(b, tf_new_biases, validate_shape=False))

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

    if op == last_conv_id:

        with tf.variable_scope(first_fc) as scope:
            w = tf.get_variable(TF_WEIGHTS)
            with tf.variable_scope(TF_WEIGHTS) as child_scope:
                w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
                pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)

            if final_2d_width > 1:
                tf_new_weights = tf.transpose(w, [1, 0])
                tf_new_weights = tf.reshape(tf_new_weights, [
                    tf.floordiv(tf_cnn_hyperparameters[first_fc][TF_FC_WEIGHT_IN_STR], final_2d_width ** 2),
                    final_2d_width, final_2d_width, tf_cnn_hyperparameters[first_fc][TF_FC_WEIGHT_OUT_STR]])
                tf_new_weights = tf.gather_nd(tf_new_weights, tf_indices_to_keep)
                tf_new_weights = tf.reshape(tf_new_weights,
                                            [-1, tf_cnn_hyperparameters[first_fc][TF_FC_WEIGHT_OUT_STR]])
            else:
                tf_new_weights = tf.gather_nd(w_vel, tf_indices_to_keep)

            update_ops.append(tf.assign(w, tf_new_weights, validate_shape=False))

            if research_parameters['optimizer'] == 'Momentum':
                if final_2d_width > 1:
                    new_weight_vel = tf.transpose(w_vel, [1, 0])
                    new_weight_vel = tf.reshape(new_weight_vel, [
                        tf.floordiv(tf_cnn_hyperparameters[first_fc][TF_FC_WEIGHT_IN_STR], final_2d_width ** 2),
                        final_2d_width, final_2d_width, tf_cnn_hyperparameters[first_fc][TF_FC_WEIGHT_OUT_STR]])
                    new_weight_vel = tf.gather_nd(new_weight_vel, tf_indices_to_keep)
                    new_weight_vel = tf.reshape(new_weight_vel,
                                                [-1, tf_cnn_hyperparameters[first_fc][TF_FC_WEIGHT_OUT_STR]])

                    new_pool_w_vel = tf.transpose(pool_w_vel, [1, 0])
                    new_pool_w_vel = tf.reshape(new_pool_w_vel, [
                        tf.floordiv(tf_cnn_hyperparameters[first_fc][TF_FC_WEIGHT_IN_STR], final_2d_width ** 2),
                        final_2d_width, final_2d_width, tf_cnn_hyperparameters[first_fc][TF_FC_WEIGHT_OUT_STR]])
                    new_pool_w_vel = tf.gather_nd(new_pool_w_vel, tf_indices_to_keep)
                    new_pool_w_vel = tf.reshape(new_pool_w_vel,
                                                [-1, tf_cnn_hyperparameters[first_fc][TF_FC_WEIGHT_OUT_STR]])
                else:
                    new_weight_vel = tf.gather_nd(w_vel, tf_indices_to_keep)
                    new_pool_w_vel = tf.gather_nd(pool_w_vel, tf_indices_to_keep)

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

            tf_new_weights = tf.transpose(w, [2, 0, 1, 3])
            tf_new_weights = tf.gather_nd(tf_new_weights, tf_indices_to_keep)
            tf_new_weights = tf.transpose(tf_new_weights, [1, 2, 0, 3])

            update_ops.append(tf.assign(w, tf_new_weights, validate_shape=False))

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
        with tf.variable_scope(TF_WEIGHTS) as child_scope:
            w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
            pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)
        with tf.variable_scope(TF_BIAS) as child_scope:
            b_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
            pool_b_vel = tf.get_variable(TF_POOL_MOMENTUM)

        tf_new_weights = tf.transpose(w, [1, 0])
        tf_new_weights = tf.gather_nd(tf_new_weights, tf_indices_to_keep)
        tf_new_weights = tf.transpose(tf_new_weights, [1, 0], name='new_weights')

        update_ops.append(tf.assign(w, tf_new_weights, validate_shape=False))

        tf_new_biases = tf.reshape(tf.gather(b, tf_indices_to_keep), shape=[-1], name='new_bias')
        update_ops.append(tf.assign(b, tf_new_biases, validate_shape=False))

        if research_parameters['optimizer'] == 'Momentum':
            new_weight_vel = tf.transpose(w_vel, [1,0])
            new_weight_vel = tf.gather_nd(new_weight_vel, tf_indices_to_keep)
            new_weight_vel = tf.transpose(new_weight_vel, [1, 0])

            new_pool_w_vel = tf.transpose(pool_w_vel, [1, 0])
            new_pool_w_vel = tf.gather_nd(new_pool_w_vel, tf_indices_to_keep)
            new_pool_w_vel = tf.transpose(new_pool_w_vel, [1, 0])

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

        tf_new_weights = tf.transpose(w, [1, 0])
        tf_new_weights = tf.gather_nd(tf_new_weights, tf_indices_to_keep)
        tf_new_weights = tf.transpose(tf_new_weights, [1, 0])

        update_ops.append(tf.assign(w, tf_new_weights, validate_shape=False))

        if research_parameters['optimizer'] == 'Momentum':
            new_weight_vel = tf.transpose(w_vel, [1, 0])
            new_weight_vel = tf.gather_nd(new_weight_vel, tf_indices_to_keep)
            new_weight_vel = tf.transpose(new_weight_vel, [1, 0])

            new_pool_w_vel = tf.transpose(pool_w_vel, [1, 0])
            new_pool_w_vel = tf.gather_nd(new_pool_w_vel, tf_indices_to_keep)
            new_pool_w_vel = tf.transpose(new_pool_w_vel, [1, 0])

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

