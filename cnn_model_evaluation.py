import cnn_model_saver
import tensorflow as tf
import numpy as np
import os
import constants
import functools
import h5py

def get_weight_vector_with_variables(cnn_ops,n):

    w_vector = None
    with tf.variable_scope(constants.TF_GLOBAL_SCOPE):
        for op in cnn_ops:
            if 'pool' not in cnn_ops:
                with tf.variable_scope(op):
                    w = tf.get_variable(constants.TF_WEIGHTS)
                    w_unwrap = tf.reshape(w,[-1])
                    if w_vector is None:
                        w_vector = tf.identity(w_unwrap)
                    else:
                        w_vector = tf.concat([w_vector,w_unwrap],axis=0)

    assert n == w_vector.get_shape().as_list()[0],'n (%d) and weight vector length (%d) do not match'%(n,w_vector.get_shape().as_list()[0])
    return tf.reshape(w_vector,[-1,1])

def update_weight_variables_from_vector(cnn_ops, cnn_hyps, w_vec, n):

    assign_ops = []
    begin_index = 0
    with tf.variable_scope(constants.TF_GLOBAL_SCOPE):
        for op in cnn_ops:
            if 'conv' in op:
                w_size = functools.reduce(lambda x,y:x*y, cnn_hyps[op]['weights'])
                w_tensor = tf.reshape(tf.slice(w_vec,begin=begin_index,size=w_size),cnn_hyps[op]['weights'])
                begin_index += w_size
                with tf.variable_scope(op):
                    assign_ops.append(tf.assign(tf.get_variable(constants.TF_WEIGHTS),w_tensor))

            elif 'fulcon' in op:
                w_size = cnn_hyps[op]['in']*cnn_hyps[op]['out']
                w_tensor = tf.reshape(tf.slice(w_vec,begin=begin_index,size=w_size), [cnn_hyps[op]['in'],cnn_hyps[op]['out']])
                begin_index += w_size
                with tf.variable_scope(op):
                    assign_ops.append(tf.assign(tf.get_variable(constants.TF_WEIGHTS),w_tensor))

    return assign_ops

def read_data_file(datatype):

    print('Reading data from HDF5 File')
    # Datasets
    if datatype == 'cifar-10':
        dataset_file = h5py.File("data" + os.sep + "cifar_10_dataset.hdf5", "r")
        train_dataset, train_labels = dataset_file['/train/images'], dataset_file['/train/labels']


    elif datatype == 'cifar-100':
        dataset_file = h5py.File("data" + os.sep + "cifar_100_dataset.hdf5", "r")
        train_dataset, train_labels = dataset_file['/train/images'], dataset_file['/train/labels']

    elif datatype == 'imagenet-250':
        dataset_file = h5py.File(
            ".." + os.sep + "PreprocessingBenchmarkImageDatasets" + os.sep + "imagenet_small_test" + os.sep + 'imagenet_250_dataset.hdf5',
            'r')
        train_dataset, train_labels = dataset_file['/train/images'], dataset_file['/train/labels']

    return train_dataset, train_labels

if __name__ == '__main__':

    cnn_hyperparam_file = 'test' + os.sep + 'model_weights' + os.sep + 'cnn-hyperparameters-0.pickle'
    cnn_weights_file = 'test' + os.sep + 'model_weights' + os.sep + 'cnn-model-0.ckpt'

    cnn_ops, cnn_hyperparameters = cnn_model_saver.get_cnn_ops_and_hyperparameters(cnn_hyperparam_file)

    session = tf.InteractiveSession()
    cnn_model_saver.load_cnn_weights(session,cnn_hyperparam_file,cnn_weights_file)

    tf_rand_seed_ph = tf.placeholder(shape=None,dtype=tf.int32)
    n = cnn_model_saver.get_weight_vector_length(cnn_hyperparam_file)
    p = 100
    print('Found weight vector length: ',n)
    A = tf.random_uniform(shape=[n,p],minval=-1.0, maxval=1.0)
    A_plus = tf.py_func(np.linalg.pinv, [A], tf.float32)

    print('Creating a weight vector from weight tensors')
    W = get_weight_vector_with_variables(cnn_ops,n)
    print('Successfully created the weight vector')

    n_sample_within_box = 25

    epsilon = 1e-3
    lower_bound_vec = -epsilon * (tf.abs(tf.matmul(A_plus,W)) + 1)
    upper_bound_vec = epsilon * (tf.abs(tf.matmul(A_plus,W)) + 1)

    z = tf.map_fn(lambda x:tf.random_uniform(shape=None,minval=x[0],maxval=x[1],seed=tf_rand_seed_ph),
                  (lower_bound_vec,upper_bound_vec),dtype=tf.float32)

    W_corrupt = W + tf.matmul(A,z)

    tf_restore_weights_with_w_corrupt = update_weight_variables_from_vector(cnn_ops,cnn_hyperparameters,W_corrupt,n)

    data_gen = data_generator.DataGenerator(batch_size, num_labels, dataset_info['train_size'],
                                            dataset_info['n_slices'],
                                            image_size, dataset_info['n_channels'], dataset_info['resize_to'],
                                            dataset_info['dataset_name'], session)

    n_iterations = 0
    if datatype=='cifar-10':
        n_iterations = 400