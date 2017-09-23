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

logger = logging.getLogger('label_sequence_generator_logger')
logger.setLevel(logging_level)
console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter(logging_format))
console.setLevel(logging_level)
logger.addHandler(console)

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
    logger.debug('Class distribution')
    logger.debug(cnt)
    euc_distance = 0
    euc_threshold = (0.1**2)*dist.size
    for li in range(dist.size):
        if li in cnt:
            euc_distance += ((cnt[li]*1.0/size)-dist[li])**2
        else:
            euc_distance += dist[li]**2

    if euc_distance>euc_threshold:
        logger.debug('Distribution:')
        logger.debug(dist)
        logger.debug('='*80)
        logger.debug('Label Sequence Counts (Normalized)')
        norm_counts = []
        for li in range(dist.size):
            if li in cnt:
                norm_counts.append(cnt[li]*1.0/size)
            else:
                norm_counts.append(0)
        logger.debug(norm_counts)
        logger.debug('='*80)
        logger.debug('')

        while euc_distance>euc_threshold:
            euc_distance = 0
            logger.info('Regenerating Label Sequence ...')
            logger.info('Euc distance: %.3f',euc_distance)
            label_sequence = get_label_sequence(dist,size)
            cnt = Counter(label_sequence)
            for li in range(dist.size):
                if li in cnt:
                    euc_distance += ((cnt[li] * 1.0 / size) - dist[li]) ** 2
                else:
                    euc_distance += dist[li] ** 2

    assert euc_distance<euc_threshold
    return label_sequence


# generate gaussian priors
def generate_gaussian_priors_for_labels(full_size, batch_size, fluctuation_normalizer,num_labels):
    chunk_count = int(full_size//batch_size)

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


batch_index = 0
def sample_label_sequence_for_batch(n_iterations, f_prior, batch_size, num_labels,freeze_index_increment=False):
    '''
    Sample data for a single batch using the prior we generated
    :param n_iterations:
    :param f_prior:
    :param batch_size:
    :param num_labels:
    :param freeze_index_increment: Freezing index increment will cause the index not to increase
    (Used for getting next unseen training batch for validation purpose)
    :return:
    '''
    global batch_index

    dist = f_prior[batch_index]
    label_sequence = sample_from_distribution(dist, batch_size)

    if not freeze_index_increment:
        batch_index = (batch_index+1)%n_iterations

    return label_sequence

def create_prior(n_iterations, distribution_type, num_labels, fluctuation_normalizer):

    batch_size = 128

    if distribution_type=='non-stationary':
        priors = generate_gaussian_priors_for_labels(n_iterations*batch_size,batch_size,fluctuation_normalizer,num_labels)
    elif distribution_type=='stationary':
        priors = np.ones((n_iterations,num_labels))*(1.0/num_labels)

    return priors

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

