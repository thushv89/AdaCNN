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
import utils
import dask as da

def inference(batch_size, cnn_ops, dataset, cnn_hyperparameters):
    global logger


    first_fc = 'fulcon_out' if 'fulcon_0' not in cnn_ops else 'fulcon_0'

    last_conv_id = ''
    for op in cnn_ops:
        if 'conv' in op:
            last_conv_id = op

    print('Defining the logit calculation ...')
    print('\tCurrent set of operations: %s' % cnn_ops)
    activation_ops = []

    x = dataset

    print('\tReceived data for X(%s)...' % x.get_shape().as_list())

    # need to calculate the output according to the layers we have
    for op in cnn_ops:
        if 'conv' in op:
            with tf.variable_scope(op, reuse=True) as scope:
                print('\tConvolving (%s) With Weights:%s Stride:%s' % (
                    op, cnn_hyperparameters[op]['weights'], cnn_hyperparameters[op]['stride']))
                print('\t\tWeights: %s', tf.shape(tf.get_variable(constants.TF_WEIGHTS)).eval())
                w, b = tf.get_variable(constants.TF_WEIGHTS), tf.get_variable(constants.TF_BIAS)

                x = tf.nn.conv2d(x, w, cnn_hyperparameters[op]['stride'],
                                 padding=cnn_hyperparameters[op]['padding'])
                x = utils.lrelu(x + b, name=scope.name + '/top')

        if 'pool' in op:
            print('\tPooling (%s) with Kernel:%s Stride:%s Type:%s' % (
                op, cnn_hyperparameters[op]['kernel'], cnn_hyperparameters[op]['stride'], cnn_hyperparameters[op]['type']))
            if cnn_hyperparameters[op]['type']=='max':
                x = tf.nn.max_pool(x, ksize=cnn_hyperparameters[op]['kernel'],
                                   strides=cnn_hyperparameters[op]['stride'],
                                   padding=cnn_hyperparameters[op]['padding'])
            elif cnn_hyperparameters[op]['type']=='avg':
                x = tf.nn.avg_pool(x, ksize=cnn_hyperparameters[op]['kernel'],
                                   strides=cnn_hyperparameters[op]['stride'],
                                   padding=cnn_hyperparameters[op]['padding'])
            else:
                raise NotImplementedError

        if 'fulcon' in op:
            with tf.variable_scope(op, reuse=True) as scope:
                w, b = tf.get_variable(constants.TF_WEIGHTS), tf.get_variable(constants.TF_BIAS)

                if first_fc == op:
                    # we need to reshape the output of last subsampling layer to
                    # convert 4D output to a 2D input to the hidden layer
                    # e.g subsample layer output [batch_size,width,height,depth] -> [batch_size,width*height*depth]

                    print('Input size of fulcon_out : ', cnn_hyperparameters[op]['in'])
                    print('Current input size ', x.get_shape().as_list())
                    # Transpose x (b,h,w,d) to (b,d,w,h)
                    # This help us to do adaptations more easily
                    x = tf.transpose(x, [0, 3, 1, 2])
                    x = tf.reshape(x, [batch_size, cnn_hyperparameters[op]['in']])
                    x = utils.lrelu(tf.matmul(x, w) + b, name=scope.name + '/top')

                elif 'fulcon_out' == op:
                    x = tf.matmul(x, w) + b

                else:
                    x = utils.lrelu(tf.matmul(x, w) + b, name=scope.name + '/top')

    return x

def get_weight_vector_with_variables(cnn_ops,n):

    w_vector = None
    with tf.variable_scope(constants.TF_GLOBAL_SCOPE, reuse = True):
        for op in cnn_ops:
            if 'pool' not in op:
                with tf.variable_scope(op, reuse=True):
                    w = tf.get_variable(constants.TF_WEIGHTS)
                    w_unwrap = tf.reshape(w,[-1])
                    if w_vector is None:
                        w_vector = tf.identity(w_unwrap)
                    else:
                        w_vector = tf.concat([w_vector,w_unwrap],axis=0)

    assert n == w_vector.get_shape().as_list()[0],'n (%d) and weight vector length (%d) do not match'%(n,w_vector.get_shape().as_list()[0])
    return tf.reshape(w_vector,[-1])

def update_weight_variables_from_vector(cnn_ops, cnn_hyps, w_vec, n):

    assign_ops = []
    begin_index = 0
    print('Updating weight Tensors from the vector')
    with tf.variable_scope(constants.TF_GLOBAL_SCOPE, reuse=True):
        for op in cnn_ops:
            if 'conv' in op:
                w_size = functools.reduce(lambda x,y:x*y, cnn_hyps[op]['weights'])
                print('Updating the weight tensor for ', op, ' with a vector slice of size ',w_size)
                w_tensor = tf.reshape(tf.slice(w_vec,begin=[begin_index],size=[w_size]),cnn_hyps[op]['weights'])
                begin_index += w_size
                with tf.variable_scope(op, reuse= True):
                    assign_ops.append(tf.assign(tf.get_variable(constants.TF_WEIGHTS),w_tensor))

            elif 'fulcon' in op:
                w_size = cnn_hyps[op]['in']*cnn_hyps[op]['out']
                print('Updating the weight tensor for ', op, ' with a vector slice of size ', w_size)
                w_tensor = tf.reshape(tf.slice(w_vec,begin=[begin_index],size=[w_size]), [cnn_hyps[op]['in'],cnn_hyps[op]['out']])
                begin_index += w_size
                with tf.variable_scope(op, reuse=True):
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

def loss(batch_size, cnn_ops, dataset, labels, cnn_hyperparameters):

    logits = inference(batch_size, cnn_ops, dataset, cnn_hyperparameters)
    # use weighted loss

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    beta = 0.0005
    fulcons = []
    for op in cnn_ops:
        if 'fulcon' in op and op != 'fulcon_out':
            fulcons.append(op)

    fc_weights = []
    for op in fulcons:
        with tf.variable_scope(op):
            fc_weights.append(tf.get_variable(constants.TF_WEIGHTS))

    loss = tf.reduce_sum([loss, beta * tf.reduce_sum([tf.nn.l2_loss(w) for w in fc_weights])])

    total_loss = loss

    return total_loss

def callback_loss(loss):
    global x_loss
    print('x_loss: ',x_loss)
    print('x_plus_Ay loss: ',loss)
    print('epsilon-sharpness:', (loss-x_loss)*100.0/(1+x_loss))

x_loss = None

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

    cnn_hyperparam_file = output_dir + os.sep + 'model_weights' + os.sep + 'cnn-hyperparameters-0.pickle'
    cnn_weights_file = output_dir + os.sep + 'model_weights' + os.sep + 'cnn-model-0.ckpt'

    cnn_ops, cnn_hyperparameters, final_2d_width = cnn_model_saver.get_cnn_ops_and_hyperparameters(cnn_hyperparam_file)
    print(cnn_hyperparameters)

    session = tf.InteractiveSession()
    cnn_model_saver.create_and_restore_cnn_weights(session,cnn_hyperparam_file,cnn_weights_file)
    cnn_model_saver.set_shapes_for_all_weights(cnn_ops,cnn_hyperparameters)
    tf_rand_seed_ph = tf.placeholder(shape=None,dtype=tf.int32)
    n = cnn_model_saver.get_weight_vector_length(cnn_hyperparam_file)

    p = 100

    #print('Creating files to store n-by-p manifold matrix')
    #hdf5_file = h5py.File(output_dir + os.sep + 'model_weights' + os.sep + 'tmp_manifold.hdf5', "w")
    #print('\tSuccessfully created the file')

    print('Found weight vector length: ',n, '\n')
    A = tf.random_uniform(shape=[n,p],minval=-100.0, maxval=100.0)
    A_plus = tf.py_func(np.linalg.pinv, [A], tf.float32)

    #hdf5_A = hdf5_file.create_dataset('A', (n, p), dtype='f')
    #hdf5_A_plus = hdf5_file.create_dataset('A_plus', (p, n), dtype='f')

    print('Creating a weight vector from weight tensors')
    W = get_weight_vector_with_variables(cnn_ops,n)
    print('\tSuccessfully created the weight vector (shape: %s)\n'%W.get_shape().as_list())

    epsilon = 1e-3
    tf_lower_bound_vec = -epsilon * (tf.abs(tf.matmul(A_plus,tf.reshape(W,[-1,1]))) + 1)
    tf_upper_bound_vec = epsilon * (tf.abs(tf.matmul(A_plus,tf.reshape(W,[-1,1]))) + 1)


    lower_bound = tf_lower_bound_vec.eval().ravel()
    upper_bound = tf_upper_bound_vec.eval().ravel()

    z_init = list(map(lambda x: np.random.uniform(low=x[0],high=x[1],size=(1))[0],
                      zip(lower_bound.tolist(),upper_bound.tolist())))
    z = tf.Variable(dtype=tf.float32, initial_value=z_init)

    W_corrupt = W + tf.reshape(tf.matmul(A,tf.reshape(z,[-1,1])),[-1])

    tf_restore_weights_with_w_corrupt = update_weight_variables_from_vector(cnn_ops,cnn_hyperparameters,W_corrupt,n)

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

    batch_size = dataset_info['train_size']//20
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

    # tf_cnn_hyperparameters = cnn_intializer.init_tf_hyperparameters(cnn_ops,cnn_hyperparameters)

    print('Running global variable initializer')
    init_op = tf.global_variables_initializer()
    _ = session.run(init_op)
    print('\tVariable initialization successful\n')

    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='TrainLabels')

    print(lower_bound.shape)
    print(upper_bound.shape)
    print(z.get_shape().as_list())
    with tf.variable_scope(constants.TF_GLOBAL_SCOPE, reuse=True):
        tf_logits_train = inference(batch_size, cnn_ops,tf_train_images, cnn_hyperparameters)
        tf_loss_train = loss(batch_size, cnn_ops, tf_train_images, tf_train_labels, cnn_hyperparameters)

        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            -tf_loss_train, var_to_bounds={z:(lower_bound.tolist(),upper_bound.tolist())},
            method = 'L-BFGS-B',options={'maxiter': 10}
        )

    #cnn_model_saver.restore_cnn_weights(session,cnn_hyperparam_file,cnn_weights_file)
    all_d, all_l = data_gen.generate_data_ordered(train_dataset, train_labels, dataset_info)

    x_loss = session.run(tf_loss_train, feed_dict={tf_train_images: all_d, tf_train_labels: all_l})

    session.run(tf_restore_weights_with_w_corrupt)
    optimizer.minimize(session, fetches=[tf_loss_train], feed_dict={tf_train_images: all_d, tf_train_labels: all_l}, loss_callback=callback_loss)

    #session.run(tf_restore_weights_with_w_corrupt,feed_dict={tf_rand_seed_ph:np.random.randint(low=0,high=23492095)})



