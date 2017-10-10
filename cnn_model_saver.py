import os
import pickle
import constants
import tensorflow as tf
import logging
import functools

WEIGHT_SAVE_DIR = 'model_weights'

saver_logger = None
main_dir = None
def set_from_main(m_dir):
    global main_dir,saver_logger
    main_dir = m_dir

    if WEIGHT_SAVE_DIR and not os.path.exists(main_dir + os.sep + WEIGHT_SAVE_DIR):
        os.mkdir(main_dir + os.sep + WEIGHT_SAVE_DIR)

    saver_logger = logging.getLogger('model_saver_logger')
    saver_logger.propagate = False
    saver_logger.setLevel(logging.INFO)
    saveHandler = logging.FileHandler(main_dir + os.sep + WEIGHT_SAVE_DIR + os.sep + 'model_saver.log', mode='w')
    saveHandler.setFormatter(logging.Formatter('%(message)s'))
    saver_logger.addHandler(saveHandler)
    saver_logger.info('#Weight saving information')

def save_cnn_hyperparameters(cnn_ops, final_2d_width, hyp_dict, hypeparam_filename):
    global main_dir,saver_logger

    hyperparam_dict = {'layers': cnn_ops}
    hyperparam_dict['final_2d_width'] = final_2d_width

    for scope in cnn_ops:
        saver_logger.info('Logging information for %s',scope)
        if 'fulcon' not in scope:
            hyperparam_dict[scope] = dict(hyp_dict[scope])
        else:
            hyperparam_dict[scope] = {'in': hyp_dict[scope]['in'], 'out': hyp_dict[scope]['out']}

        saver_logger.info(hyperparam_dict[scope])

    pickle.dump(hyperparam_dict, open(main_dir + os.sep + WEIGHT_SAVE_DIR + os.sep + hypeparam_filename, "wb"))


def save_cnn_weights(cnn_ops, sess, model_filename):
    global main_dir, saver_logger
    var_dict = {}

    if WEIGHT_SAVE_DIR and not os.path.exists(main_dir + os.sep + WEIGHT_SAVE_DIR):
        os.mkdir(main_dir + os.sep + WEIGHT_SAVE_DIR)

    saver_logger.info('Saving weights to disk')
    for scope in cnn_ops:
        saver_logger.info('Processing %s',scope)
        if 'pool' not in scope:

            weights_name = constants.TF_GLOBAL_SCOPE + constants.TF_SCOPE_DIVIDER + \
                           scope + constants.TF_SCOPE_DIVIDER + constants.TF_WEIGHTS

            bias_name = constants.TF_GLOBAL_SCOPE + constants.TF_SCOPE_DIVIDER + \
                        scope + constants.TF_SCOPE_DIVIDER + constants.TF_BIAS
            with tf.variable_scope(constants.TF_GLOBAL_SCOPE, reuse=True):
                with tf.variable_scope(scope,reuse=True):
                    var_dict[weights_name] = tf.get_variable(constants.TF_WEIGHTS)
                    var_dict[bias_name] = tf.get_variable(constants.TF_BIAS)

            saver_logger.info(weights_name)
            saver_logger.info(bias_name)

    saver = tf.train.Saver(var_dict)
    saver.save(sess,main_dir + os.sep + WEIGHT_SAVE_DIR + os.sep +  model_filename)

def cnn_create_variables_for_restore(hyperparam_filepath):
    hyperparam_dict = pickle.load(open(hyperparam_filepath, 'rb'))
    scope_list = hyperparam_dict['layers']

    for scope in scope_list:
        with tf.variable_scope(constants.TF_GLOBAL_SCOPE):
            with tf.variable_scope(scope,reuse=False):
                if 'conv' in scope:
                    tf.get_variable(constants.TF_WEIGHTS, shape=hyperparam_dict[scope]['weights'],
                                    initializer=tf.constant_initializer(0.0, dtype=tf.float32), trainable=False)
                    tf.get_variable(constants.TF_BIAS, hyperparam_dict[scope]['weights'][-1],
                                    initializer=tf.constant_initializer(0.0, dtype=tf.float32), trainable=False)
                if 'fulcon' in scope:
                    tf.get_variable(constants.TF_WEIGHTS, shape=[hyperparam_dict[scope]['in'],hyperparam_dict[scope]['out']],
                                    initializer=tf.constant_initializer(0.0, dtype=tf.float32),trainable=False)
                    tf.get_variable(constants.TF_BIAS, hyperparam_dict[scope]['out'],
                                    initializer=tf.constant_initializer(0.0, dtype=tf.float32),trainable=False)


def get_cnn_ops_and_hyperparameters(hyperparam_filepath):

    print('File got: ',hyperparam_filepath)
    hyperparam_dict = pickle.load(open(hyperparam_filepath, 'rb'))
    scope_list = hyperparam_dict['layers']
    final_2d_width = hyperparam_dict['final_2d_width']

    return scope_list, hyperparam_dict, final_2d_width

def get_weight_vector_length(hyperparam_filepath):
    hyperparam_dict = pickle.load(open(hyperparam_filepath, 'rb'))

    scope_list = hyperparam_dict['layers']
    n = 0
    for scope in scope_list:
        if 'conv' in scope:
            n += functools.reduce(lambda x,y:x*y,hyperparam_dict[scope]['weights'])
        elif 'fulcon' in scope:
            n += hyperparam_dict[scope]['in']*hyperparam_dict[scope]['out']

    return n

def create_and_restore_cnn_weights(sess,hyperparam_filepath, weights_filepath):

    cnn_create_variables_for_restore(hyperparam_filepath)
    saver = tf.train.Saver()
    print('Restoring weights from file: ', weights_filepath)
    saver.restore(sess, weights_filepath)
    print('Restoring successful.')


def restore_cnn_weights(sess,hyperparam_filepath, weights_filepath):

    saver = tf.train.Saver()
    print('Restoring weights from file: ', weights_filepath)
    saver.restore(sess, weights_filepath)
    print('Restoring successful.')


def set_shapes_for_all_weights(cnn_ops, cnn_hyperparameters):
    print('Setting shape information for all variables')
    with tf.variable_scope(constants.TF_GLOBAL_SCOPE, reuse=True):
        for op in cnn_ops:
            if 'conv' in op:
                with tf.variable_scope(op, reuse=True):
                    w = tf.get_variable(constants.TF_WEIGHTS)
                    b = tf.get_variable(constants.TF_BIAS)
                    w.set_shape(cnn_hyperparameters[op]['weights'])
                    b.set_shape([cnn_hyperparameters[op]['weights'][3]])
            elif 'fulcon' in op:
                with tf.variable_scope(op, reuse=True):
                    w = tf.get_variable(constants.TF_WEIGHTS)
                    b = tf.get_variable(constants.TF_BIAS)
                    w.set_shape([cnn_hyperparameters[op]['in'],cnn_hyperparameters[op]['out']])
                    b.set_shape([cnn_hyperparameters[op]['out']])

    print('\tSetting shape information successful...')