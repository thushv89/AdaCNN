__author__ = 'Thushan Ganegedara'

import sys, argparse
import pickle
import struct
import random
import math
import numpy as np
import os
from collections import defaultdict
import logging
from collections import Counter
import utils
import matplotlib.pyplot as plt


logging_level = logging.INFO
logging_format = '[%(funcName)s] %(message)s'

# Produce a covariance matrix
def kernel(a, b):
    """ Squared exponential kernel """
    sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
    return np.exp(-0.5 * sqdist)


# generate 'size' samples for a given distribution
def sample_from_distribution(dist,size):
    global logger

    def get_label_sequence(dist,size):
        dist_cumsum = np.cumsum(dist)
        label_sequence = []
        label_found = False
        # take each element of the dist
        # generate a random number and
        # if number < cumulative sum of distribution at i th point
        # add i as a label other wise don't
        # if a label is not found we add the last label
        logger.debug('Sampling %d from the given distribution of size(%s)',size,str(dist.shape))
        for i in range(size):
            r = np.random.random()
            for j in range(dist.size):
                if r<dist_cumsum[j]:
                    label_sequence.append(j)
                    label_found = True
                    break
                label_found = False
            if not label_found:
                # if any label not found we add the last label
                label_sequence.append(dist.size-1)

        assert len(label_sequence)==size
        np.random.shuffle(label_sequence)
        return label_sequence

    label_sequence = get_label_sequence(dist,size)
    cnt = Counter(label_sequence)

    euc_distance = 0
    euc_threshold = (0.02**2)*dist.size
    for li in range(dist.size):
        if li in cnt:
            euc_distance += ((cnt[li]*1.0/size)-dist[li])**2
        else:
            euc_distance += dist[li]**2

    if euc_distance>euc_threshold:
        logger.debug('Distribution:')
        logger.debug(dist)
        logger.debug('='*80)
        logger.debug('Label Sequence Counts')
        norm_counts = []
        for li in range(dist.size):
            if li in cnt:
                norm_counts.append(cnt[li]*1.0/size)
            else:
                norm_counts.append(0)
        logger.debug(norm_counts)
        logger.debug('='*80)
        logger.debug('')

        logger.debug('Regenerating Label Sequence ...')
        label_sequence = get_label_sequence(dist,size)
        for li in range(dist.size):
            if li in cnt:
                euc_distance += ((cnt[li] * 1.0 / size) - dist[li]) ** 2
            else:
                euc_distance += dist[li] ** 2

    assert euc_distance<euc_threshold
    return label_sequence


# generate gaussian priors
def generate_gaussian_priors_for_labels(batch_size,elements,chunk_size,num_labels, fluctuation_normalizer):
    chunk_count = int(elements/chunk_size)

    # the upper bound of x defines the number of peaks
    # smaller bound => less peaks
    # larger bound => more peaks
    x = np.linspace(0,max(num_labels,100), chunk_count).reshape(-1, 1)
    # 1e-6 * is for numerical stibility
    L = np.linalg.cholesky(kernel(x, x) + 1e-6 * np.eye(chunk_count))

    # single chunk represetn a single point on x axis of the gaussian curve
    # f_prior represent how the data distribution looks like at each chunk
    # f_prior [chunk_count,num_labels] size
    # e.g. f_prior[0] is class distribution at first chunk of elements
    f_prior = np.dot(L, np.random.normal(size=(chunk_count, num_labels)))
    # normalization
    f_prior -= f_prior.min()
    # use below line to control the scale of gaussians higher the power, higher the fluctuations are
    # original was sqrt(num_labels)
    f_prior = f_prior ** fluctuation_normalizer
    f_prior /= np.sum(f_prior, axis=1).reshape(-1, 1)

    return f_prior

def sample_label_sequence_with_distribution(dist_logger, dataset_info, data_dir, f_prior, save_directory):

    elements, batch_size = dataset_info['elements'], dataset_info['batch_size']
    num_chunks = elements / chunk_size
    resize_to = dataset_info['resize_to']
    dataset_type, image_size = dataset_info['dataset_type'], dataset_info['image_size']
    num_channels = dataset_info['num_channels']

    for i, dist in enumerate(f_prior):
        label_sequence = sample_from_distribution(dist, chunk_size)

        cnt = Counter(label_sequence)
        dist_str = ''
        for li in range(num_labels):
            dist_str += str(cnt[li] / len(label_sequence)) + ',' if li in cnt else str(0) + ','
        dist_logger.info('%d,%s', i, dist_str)


if __name__ == '__main__':

    global logger
    logger = logging.getLogger('label_sequence_generator')
    logger.setLevel(logging_level)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(logging_format))
    console.setLevel(logging_level)
    logger.addHandler(console)

    persist_dir = 'label_sequence_dir' # various things we persist related to ConstructorRL

    dataset_type = 'imagenet-250'  # 'cifar-10 imagenet-250
    distribution_type = 'stationary'

    batch_size = 128

    elements = int(batch_size * 10000)  # number of elements in the whole dataset
    # there are elements/chunk_size points in the gaussian curve for each class
    chunk_size = int(batch_size * 10)  # number of samples sampled for each instance of the gaussian curve

    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    class_distribution_logger = logging.getLogger('label-sequence-'+dataset_type+'-'+distribution_type)
    class_distribution_logger.setLevel(logging.INFO)
    cdfileHandler = logging.FileHandler(persist_dir + os.sep + 'label-sequence-'+dataset_type+'-'+distribution_type + '.log',
                                        mode='w')
    cdfileHandler.setFormatter(logging.Formatter('%(message)s'))
    class_distribution_logger.addHandler(cdfileHandler)


    dataset_folder = ''
    dataset_sizes_file = dataset_folder+os.sep+'dataset_sizes.xml'
    dataset_sizes = utils.retrive_dictionary_from_xml(dataset_sizes_file)

    if dataset_type == 'cifar-10':
        dataset_file = dataset_folder + os.sep + None
        train_size, test_size = None,None
        image_size = 32
        num_labels = 10
        num_channels = 3

        dataset_size = 50000
        test_size = 10000
        fluctuation_normalizer = num_labels*5
        dataset_info = {'dataset_type':dataset_type,'elements':elements,'chunk_size':chunk_size,'image_size':image_size,
                        'num_channels':num_channels,'num_labels':num_labels,'dataset_size':dataset_size,'test_size':test_size}

    if dataset_type == 'cifar-100':
        image_size = 32
        num_labels = 100
        num_channels = 3

        fluctuation_normalizer = num_labels
        dataset_info = {'dataset_type':dataset_type,'elements':elements,'chunk_size':chunk_size,'image_size':image_size,
                        'num_channels':num_channels,'num_labels':num_labels,'dataset_size':dataset_size,'test_size':test_size}

    elif dataset_type == 'svhn-10':
        image_size = 32
        num_labels = 10
        num_channels = 3

        image_use_counter = {}
        dataset_info = {'dataset_type': dataset_type, 'elements': elements, 'chunk_size': chunk_size,
                        'image_size': image_size,
                        'num_channels': num_channels, 'num_labels': num_labels, 'dataset_size': dataset_size,
                        'test_size': test_size}

    elif dataset_type == 'imagenet-250':

        train_size, test_size = dataset_sizes['train_dataset'],dataset_sizes['valid_dataset']

        image_size = 128
        num_labels = 250
        num_channels = 3

        chunk_size = int(batch_size * 50)  # number of samples sampled for each instance of the gaussian curve


        image_use_counter = {}
        dataset_info = {'dataset_type':dataset_type,'elements':elements,'chunk_size':chunk_size,'image_size':image_size,'num_labels':num_labels,'dataset_size':train_size,
                        'num_channels':num_channels,'data_in_memory':chunk_size,'resize_to':resize_to,'test_size':test_size}

    logger.info('='*60)
    logger.info('Dataset Information')
    logger.info('\t%s',str(dataset_info))
    logger.info('='*60)

    logger.info('Generating gaussian priors')
    if distribution_type=='non-stationary':
        priors = generate_gaussian_priors_for_labels(batch_size,elements,chunk_size,num_labels, fluctuation_normalizer)
    elif distribution_type=='stationary':
        priors = np.ones((elements//chunk_size,num_labels))*(1.0/num_labels)
    else:
        raise NotImplementedError

    logger.info('\tGenerated priors of size: %d,%d',priors.shape[0],priors.shape[1])


def plot_distribution(priors):
    '''
    Plot the priors as stacked lines
    :param priors: A (num_chunks, num_labels) sized numpy array
    :return:
    '''

    cuml_priors = np.asarray(priors)
    sum_col = np.zeros((priors.shape[0],),dtype=np.float32)
    x = np.arange(priors.shape[0])

    for col_i in range(priors.shape[1]):
        col = priors[:, col_i]
        if col_i==0:
            sum_col = col
        else:
            cuml_priors[:,col_i] += sum_col
            sum_col = priors[:,col_i]

        plt.plot(x,cuml_priors[:,col_i])

    plt.show()

def quick_test():

    # =============== Quick Test =====================
    sample_size = 1000
    num_labels = 10
    x = np.linspace(0, 50, sample_size).reshape(-1, 1)
    #x = np.random.random(size=[1,sample_size])

    # 1e-6 * is for numerical stibility
    L = np.linalg.cholesky(kernel(x, x) + 1e-6 * np.eye(sample_size))

    # massage the data to get a good distribution
    # len(data) is number of labels
    # f_prior are the samples
    f_prior = np.dot(L, np.random.normal(size=(sample_size, num_labels)))
    # make minimum zero
    f_prior -= f_prior.min()
    f_prior = f_prior ** math.ceil(math.sqrt(num_labels))
    f_prior /= np.sum(f_prior, axis=1).reshape(-1, 1)

    x_axis = np.arange(sample_size)


    for i in range(num_labels):
        plt.plot(x_axis,f_prior[:,i])

    #print(np.sum(f_prior,axis=1).shape)
    #plt.plot(x_axis,np.sum(f_prior,axis=1))
    plt.show()

