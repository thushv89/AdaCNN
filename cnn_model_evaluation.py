import cnn_model_saver
import tensorflow as tf
import numpy as np
import os
import constants
import functools
import h5py
import data_generator
import ada_cnn
import cnn_intializer
import sys
import getopt

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

    allow_growth = False
    use_multiproc = False
    fake_tasks = False
    noise_label_rate = None
    noise_image_rate = None
    adapt_randomly = False

    try:
        opts, args = getopt.getopt(
            sys.argv[1:], "", ["output_dir=", "num_gpus=", "memory=", 'allow_growth=',
                               'dataset_type=',
                               'adapt_structure=', 'rigid_pooling=', 'rigid_pool_type='])
    except getopt.GetoptError as err:
        print(err.with_traceback())
        print('<filename>.py --output_dir= --num_gpus= --memory= --pool_workers=')

    if len(opts) != 0:
        for opt, arg in opts:
            if opt == '--output_dir':
                output_dir = arg
            if opt == '--num_gpus':
                num_gpus = int(arg)
            if opt == '--memory':
                mem_frac = float(arg)
            if opt == '--allow_growth':
                allow_growth = bool(arg)
            if opt == '--dataset_type':
                datatype = str(arg)
            if opt == '--adapt_structure':
                adapt_structure = bool(int(arg))
            if opt == '--rigid_pooling':
                rigid_pooling = bool(int(arg))
            if opt == '--rigid_pool_type':
                rigid_pool_type = str(arg)

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

    batch_size = 128
    n_iterations = 0
    dataset_info = {}

    if datatype=='cifar-10':
        image_size = 24
        n_iterations = 400
        num_labels = 10
        dataset_info['dataset_name']='cifar-10'
        dataset_info['n_channels']=3
        dataset_info['resize_to'] = 0
        dataset_info['n_slices'] = 1
        dataset_info['train_size'] = 50000
    if datatype=='cifar-100':
        image_size = 24
        n_iterations = 400
        num_labels = 100
        dataset_info['dataset_name']='cifar-100'
        dataset_info['n_channels']=3
        dataset_info['resize_to'] = 0
        dataset_info['n_slices'] = 1
        dataset_info['train_size'] = 50000

    train_dataset, train_labels = read_data_file(datatype)

    data_gen = data_generator.DataGenerator(batch_size, num_labels, dataset_info['train_size'],
                                            dataset_info['n_slices'],
                                            image_size, dataset_info['n_channels'], dataset_info['resize_to'],
                                            dataset_info['dataset_name'], session)

    if datatype != 'imagenet-250':
        tf_train_images = tf.placeholder(tf.float32,
                                         shape=(batch_size, image_size,
                                                image_size, dataset_info['n_channels']),
                                         name='TrainDataset')
    else:
        train_train_images = tf.placeholder(tf.float32,
                                            shape=(batch_size, dataset_info['resize_to'],
                                                   dataset_info['resize_to'], dataset_info['n_channels']),
                                            name='TrainDataset')

    tf_cnn_hyperparameters = cnn_intializer.init_tf_hyperparameters(cnn_ops,cnn_hyperparameters)

    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='TrainLabels')

    tf_data_weights = tf.placeholder(tf.float32, shape=(batch_size), name='TrainWeights')

    tf_logits_train = ada_cnn.inference(tf_train_images, tf_cnn_hyperparameters, True)


    tf_loss_train = ada_cnn.tower_loss(tf_train_images, tf_train_labels, True,
                               tf_data_weights, tf_cnn_hyperparameters)
    for iteration in range(n_iterations):

        b_d, b_l = data_gen.generate_data_ordered(train_dataset, train_labels, dataset_info)