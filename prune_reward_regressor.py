
import numpy as np
import json
import random
import logging
import sys
from math import ceil, floor
from six.moves import cPickle as pickle
import os
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.base import clone
import tensorflow as tf
import constants


class PruneRewardRegressor(object):

    def __init__(self, **params):
        self.session = params['session']
        self.input_size = params['n_tasks'] + 1
        self.n_tasks = params['n_tasks']
        self.hidden_layers = [32,16]
        self.hidden_scopes = ['fulcon_1','fulcon_2','fulcon_out']
        self.layer_info = None
        self.output_size = 1

        self.layer_info = [self.input_size]
        for hidden in self.hidden_layers:
            self.layer_info.append(hidden)  # 128,64,32
        self.layer_info.append(self.output_size)

        self.batch_size = 10
        self.tf_weights, self.tf_bias = [], []

        self.momentum = 0.9  # 0.9
        self.learning_rate = 0.01  # 0.01

        self.tf_input = tf.placeholder(tf.float32, shape=(None, self.input_size), name='PruneRewardInput')
        self.tf_output = tf.placeholder(tf.float32, shape=(None, self.output_size), name='PruneRewardOutput')

        all_params = self.tf_define_network()
        init_op = tf.variables_initializer(all_params)

        #_ = self.session.run(init_op)

        self.tf_net_out = self.tf_calc_output(self.tf_input)
        self.tf_loss = self.tf_sqr_loss(self.tf_net_out, self.tf_output)
        self.tf_optimize_op = self.tf_momentum_optimizer(self.tf_loss)

        all_vars_in_scope = tf.get_collection(tf.GraphKeys.VARIABLES,scope=constants.TF_PRUNE_REWARD_SCOPE)
        _ = self.session.run(tf.variables_initializer(all_vars_in_scope))

        self.prune_predict_logger = logging.getLogger('prune_predict_logger')
        self.prune_predict_logger.propagate = False
        self.prune_predict_logger.setLevel(logging.INFO)
        handler = logging.FileHandler(params['persist_dir'] + os.sep + 'predicted_prune' + '.log',
                                            mode='w')
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.prune_predict_logger.addHandler(handler)


    def tf_define_network(self):
        '''
        Initialize the variables for neural network used for q learning
        :return:
        '''
        all_params = []
        with tf.variable_scope(constants.TF_PRUNE_REWARD_SCOPE):
            for li in range(len(self.layer_info) - 1):
                with tf.variable_scope(self.hidden_scopes[li],reuse=False):
                    all_params.append(tf.get_variable(constants.TF_WEIGHTS, initializer=tf.truncated_normal(
                        [self.layer_info[li], self.layer_info[li + 1]], stddev=2. / self.layer_info[li]
                    )))

                    all_params.append(tf.get_variable(constants.TF_BIAS, initializer= tf.truncated_normal(
                        [self.layer_info[li + 1]], stddev=2. / self.layer_info[li]
                    )))
        return all_params

    def tf_calc_output(self, tf_input):
        tf_net_out = tf_input
        with tf.variable_scope(constants.TF_PRUNE_REWARD_SCOPE):
            for scope in self.hidden_scopes[:-1]:
                with tf.variable_scope(scope,reuse=True):
                    w, b = tf.get_variable(constants.TF_WEIGHTS), tf.get_variable(constants.TF_BIAS)
                    tf_net_out = tf.nn.relu(tf.matmul(tf_net_out,w)+b)

            with tf.variable_scope(self.hidden_scopes[-1],reuse=True):
                w, b = tf.get_variable(constants.TF_WEIGHTS), tf.get_variable(constants.TF_BIAS)
                tf_net_out = tf.nn.tanh(tf.matmul(tf_net_out,w)+b)

        return tf_net_out

    def tf_sqr_loss(self,net_out, output):

        return tf.reduce_mean((net_out-output)**2)

    def tf_momentum_optimizer(self,loss):

        return tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum).minimize(loss)

    def train_mlp_with_data(self,in_data, out_data):

        _ = self.session.run(self.tf_optimize_op,feed_dict={self.tf_input:in_data, self.tf_output:out_data})

    def predict_best_prune_factor(self,task_id):

        values_to_try = np.zeros((10,self.n_tasks),dtype=np.float32)
        values_to_try[:,task_id]=1.0
        prune_factors = np.asarray([0.1 * pi for pi in range(10)], dtype=np.float32).reshape(-1, 1)
        values_to_try = np.append(values_to_try,prune_factors, axis=1)

        predicted_values = self.session.run(self.tf_net_out,feed_dict={self.tf_input:values_to_try})

        self.prune_predict_logger.info('Predicting for task %d',task_id)
        predict_str = ''
        for p in predicted_values.ravel().tolist():
            predict_str += str(p) + ','
        self.prune_predict_logger.info(predict_str)
        return prune_factors[np.asscalar(np.argmax(predicted_values))]

