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
import dask.array as da
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from functools import partial
import logging
import time

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

    #print('\tReceived data for X(%s)...' % x.get_shape().as_list())

    # need to calculate the output according to the layers we have
    for op in cnn_ops:
        if 'conv' in op:
            with tf.variable_scope(op, reuse=True) as scope:
                #print('\tConvolving (%s) With Weights:%s Stride:%s' % (
                #    op, cnn_hyperparameters[op]['weights'], cnn_hyperparameters[op]['stride']))
                #print('\t\tWeights: %s', tf.shape(tf.get_variable(constants.TF_WEIGHTS)).eval())
                w, b = tf.get_variable(constants.TF_WEIGHTS), tf.get_variable(constants.TF_BIAS)

                x = tf.nn.conv2d(x, w, cnn_hyperparameters[op]['stride'],
                                 padding=cnn_hyperparameters[op]['padding'])
                x = utils.lrelu(x + b, name=scope.name + '/top')

        if 'pool' in op:
            #print('\tPooling (%s) with Kernel:%s Stride:%s Type:%s' % (
            #    op, cnn_hyperparameters[op]['kernel'], cnn_hyperparameters[op]['stride'], cnn_hyperparameters[op]['type']))
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

                    #print('Input size of fulcon_out : ', cnn_hyperparameters[op]['in'])
                    #print('Current input size ', x.get_shape().as_list())
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

def read_data_file(datatype,load_train_data):

    print('Reading data from HDF5 File')
    # Datasets
    if datatype == 'cifar-10':
        dataset_file = h5py.File("data" + os.sep + "cifar_10_dataset.hdf5", "r")
        if load_train_data:
            dataset, labels = dataset_file['/train/images'], dataset_file['/train/labels']
        else:
            dataset, labels = dataset_file['/test/images'], dataset_file['/test/labels']

    elif datatype == 'cifar-100':
        dataset_file = h5py.File("data" + os.sep + "cifar_100_dataset.hdf5", "r")
        if load_train_data:
            dataset, labels = dataset_file['/train/images'], dataset_file['/train/labels']
        else:
            dataset, labels = dataset_file['/test/images'], dataset_file['/test/labels']

    elif datatype == 'imagenet-250':
        dataset_file = h5py.File(
            ".." + os.sep + "PreprocessingBenchmarkImageDatasets" + os.sep + "imagenet_small_test" + os.sep + 'imagenet_250_dataset.hdf5',
            'r')
        if load_train_data:
            dataset, labels = dataset_file['/train/images'], dataset_file['/train/labels']
        else:
            dataset, labels = dataset_file['/valid/images'], dataset_file['/valid/labels']
    return dataset, labels

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


def calc_epsilon_sharpness(x_loss,x_max_loss):
    print('x_loss: ',x_loss)
    print('x_plus_Ay loss: ',loss)
    print('epsilon-sharpness:', (x_max_loss-x_loss)*100.0/(1+x_loss))
    return (x_max_loss-x_loss)*100.0/(1+x_loss)


def callback_loss(z,W,A,tf_corrupt_weights_op, tf_loss_op, tf_loss_feed_dict):

    global W_corrupt_placeholder

    W_corrupt = W + np.reshape(da.dot(A, np.reshape(z, [-1, 1])), [-1])

    session.run(tf_corrupt_weights_op,feed_dict={W_corrupt_placeholder:W_corrupt})

    l = session.run(tf_loss_op, feed_dict=tf_loss_feed_dict)

    # use -l since we need to maximize the loss

    return -l

x_loss = None
W_corrupt_placeholder = None

minimize_iter = 1
start_time = 0

def callback_iteration(xk):
    global minimize_iter, start_time
    print('\tSingle iteration (%d) finished (%d Secs)'%(minimize_iter,(time.time()-start_time)))
    minimize_iter += 1

if __name__ == '__main__':

    allow_growth = False
    use_multiproc = False
    fake_tasks = False
    noise_label_rate = None
    noise_image_rate = None
    adapt_randomly = False

    try:
        opts, args = getopt.getopt(
            sys.argv[1:], "", ["output_dir=", "file_id=", "memory=", 'allow_growth=',
                               'dataset_type=',
                               'adapt_structure=', 'rigid_pooling=', 'rigid_pool_type='])
    except getopt.GetoptError as err:
        print(err.with_traceback())
        print('<filename>.py --output_dir= --num_gpus= --memory= --pool_workers=')

    if len(opts) != 0:
        for opt, arg in opts:
            if opt == '--output_dir':
                output_dir = arg
            if opt == '--file_id':
                file_id = int(arg)
            if opt == '--memory':
                mem_frac = float(arg)
            if opt == '--allow_growth':
                allow_growth = bool(int(arg))
            if opt == '--dataset_type':
                datatype = str(arg)


    cnn_hyperparam_file = output_dir + os.sep + 'model_weights' + os.sep + 'cnn-hyperparameters-%d.pickle'%file_id
    cnn_weights_file = output_dir + os.sep + 'model_weights' + os.sep + 'cnn-model-%d.ckpt'%file_id

    cnn_ops, cnn_hyperparameters, final_2d_width = cnn_model_saver.get_cnn_ops_and_hyperparameters(cnn_hyperparam_file)
    print(cnn_hyperparameters)

    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = mem_frac
    config.gpu_options.allow_growth = allow_growth
    config.log_device_placement = False
    session = tf.InteractiveSession(config=config)

    cnn_model_saver.create_and_restore_cnn_weights(session,cnn_hyperparam_file,cnn_weights_file)
    cnn_model_saver.set_shapes_for_all_weights(cnn_ops,cnn_hyperparameters)

    print('Calcluating total number of  CNN parameters.')
    n = cnn_model_saver.get_weight_vector_length(cnn_hyperparam_file)
    print('\tSuccessfully calcluated total number of  CNN parameters.')

    p = 50

    #print('Creating files to store n-by-p manifold matrix')
    #hdf5_file = h5py.File(output_dir + os.sep + 'model_weights' + os.sep + 'tmp_manifold.hdf5', "w")
    #print('\tSuccessfully created the file')

    print('Found weight vector length: ',n, '\n')
    A_columns = np.random.uniform(-1e8,1e8,size=(p)).astype(np.float32).tolist()
    A = np.stack([np.ones(shape=(n),dtype=np.float32)*col for col in A_columns],axis=1)
    #A = np.random.uniform(-1e-5,1e-5,size=(n,p))
    assert A.shape==(n,p), 'Shape of A %s'%str(A.shape)
    A = da.from_array(A, chunks=(n//100,p))
    print(A[:10,:10].compute())
    # A_plus = (A'A)−1A' (https://pythonhosted.org/algopy/examples/moore_penrose_pseudoinverse.html)
    print('Calculating the psuedo inverse of A')
    start_time = time.time()
    A_plus = da.from_array(np.linalg.pinv(A),chunks=(p,n//100))
    print(A_plus[:10, :10].compute())
    start_time = time.time()
    print('\tSuccessfully calculated the psuedo inverse of A (%d Secs)\n'%(time.time()-start_time))
    #hdf5_A = hdf5_file.create_dataset('A', (n, p), dtype='f')
    #hdf5_A_plus = hdf5_file.create_dataset('A_plus', (p, n), dtype='f')

    print('Creating a weight vector from weight tensors')
    W = session.run(get_weight_vector_with_variables(cnn_ops,n))
    W_corrupt_placeholder = tf.placeholder(shape=[n],dtype=tf.float32,name='W_corrupt_ph')
    print('\tSuccessfully created the weight vector (shape: %s)\n'%W.shape)

    print('Calculating lower and upper bound of the box')
    start_time = time.time()
    epsilon = 1e-3
    lower_bound = -epsilon * (np.abs(da.dot(A_plus,np.reshape(W,(-1,1)))) + 1)
    #upper_bound = epsilon * (np.abs(da.dot(A_plus,np.reshape(W,(-1,1)))) + 1)
    upper_bound = -np.asarray(lower_bound)

    print('\t Successfully calculated lower and upper bound of the box (%d Secs)\n'%(time.time()-start_time))

    del A_plus

    z_init = list(map(lambda x: np.random.uniform(low=x[0],high=x[1],size=(1))[0],
                      zip(lower_bound.tolist(),upper_bound.tolist())))

    #W_corrupt = W + np.reshape(da.dot(A,tf.reshape(z,[-1,1])),[-1])

    tf_restore_weights_with_w_corrupt = update_weight_variables_from_vector(cnn_ops,cnn_hyperparameters,W_corrupt_placeholder,n)

    n_iterations = 0
    dataset_info = {}

    logger = logging.getLogger('error_logger')
    logger.propagate = False
    logger.setLevel(logging.INFO)
    errHandler = logging.FileHandler(output_dir + os.sep + 'eps-sharpness.log', mode='w')
    errHandler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(errHandler)
    logger.info('#Output Dir: %s', output_dir)


    if datatype=='cifar-10':
        image_size = 24
        num_labels = 10
        dataset_info['dataset_name']='cifar-10'
        dataset_info['n_channels']=3
        dataset_info['resize_to'] = 0
        dataset_info['n_slices'] = 1
        dataset_info['train_size'] = 50000
    if datatype=='cifar-100':
        image_size = 24
        num_labels = 100
        dataset_info['dataset_name']='cifar-100'
        dataset_info['n_channels']=3
        dataset_info['resize_to'] = 0
        dataset_info['n_slices'] = 1
        dataset_info['train_size'] = 50000

    batch_size = dataset_info['train_size']//20
    dataset, labels = read_data_file(datatype,load_train_data=True)

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

    #print(lower_bound.shape)
    #print(upper_bound.shape)
    #print(z.get_shape().as_list())
    print('Defining ops for logits and loss')
    with tf.variable_scope(constants.TF_GLOBAL_SCOPE, reuse=True):
        tf_logits_train = inference(batch_size, cnn_ops,tf_train_images, cnn_hyperparameters)
        tf_loss_train = loss(batch_size, cnn_ops, tf_train_images, tf_train_labels, cnn_hyperparameters)
    print('\t Successfully defined\n')

    all_d, all_l = data_gen.generate_data_ordered(dataset, labels, dataset_info)
    print('Retrieved data \n')
    # Original loss
    start_time = time.time()
    x_loss = session.run(tf_loss_train, feed_dict={tf_train_images: all_d, tf_train_labels: all_l})
    print('Calculated loss (%d Secs)\n'%(time.time()-start_time))

    part_loss_callback = partial(callback_loss,W=W,A=A,tf_corrupt_weights_op=tf_restore_weights_with_w_corrupt, tf_loss_op=tf_loss_train,
            tf_loss_feed_dict={tf_train_images: all_d, tf_train_labels: all_l})

    print('Lower and Upper bounds for z')
    print(list(zip(lower_bound.ravel().tolist(), upper_bound.ravel().tolist()))[:10])
    print('\n')
    print('L-BFGS Optimization started')

    opt_res = minimize(fun=part_loss_callback,x0=z_init,method='L-BFGS-B',
                       bounds=list(zip(lower_bound.ravel().tolist(),upper_bound.ravel().tolist())),
                       options={'eps':1e-3,'maxfun':100,'maxiter':10,'ftol':0.001}, callback=callback_iteration)

    #opt_res = basinhopping(func=part_loss_callback, x0=z_init,
    #                       minimizer_kwargs={'method':'L-BFGS-B','options':{'maxfun':25,'maxiter':10},
    #                                         'bounds':list(zip(lower_bound.ravel().tolist(),upper_bound.ravel().tolist()))}
    #                       )
    z_max = opt_res['x']
    print('Maximum z is given by')
    print(z_max,'\n')

    z_max_loss = - opt_res['fun']
    print('Maximum loss')
    print(z_max_loss)
    print('Manually calc max loss')
    print(-part_loss_callback(z_max))

    print('\tL-BFGS Optimization finished \n')

    logger.info('Epsilon-shapness: %.5f', calc_epsilon_sharpness(x_loss, z_max_loss))

    print('='*80)
    print('Epsilon-sharpness: ',calc_epsilon_sharpness(x_loss,z_max_loss))
    print('=' * 80)
    #cnn_model_saver.restore_cnn_weights(session,cnn_hyperparam_file,cnn_weights_file)


    #session.run(tf_restore_weights_with_w_corrupt,feed_dict={tf_rand_seed_ph:np.random.randint(low=0,high=23492095)})



