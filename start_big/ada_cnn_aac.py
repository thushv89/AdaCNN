
__author__ = 'Thushan Ganegedara'

from enum import IntEnum
from collections import defaultdict
import numpy as np
import json
import random
import logging
import sys
from math import ceil, floor
import os
import tensorflow as tf
sys.path.append('../AdaCNN')
import utils
import constants

logging_level = logging.DEBUG
logging_format = '[%(name)s] [%(funcName)s] %(message)s'


class AdaCNNAdaptingAdvantageActorCritic(object):
    def __init__(self, **params):



        self.binned_data_dist_length = params['binned_data_dist_length']

        # RL Agent Specifict Hyperparameters
        self.discount_rate = params['discount_rate']


        self.batch_size = params['batch_size']
        self.add_amount = params['adapt_max_amount']
        self.add_fulcon_amount = params['adapt_fulcon_max_amount']
        self.epsilon = params['epsilon']
        self.min_epsilon = 0.1

        self.num_classes = params['num_classes']
        self.persit_dir = params['persist_dir']

        # CNN specific Hyperparametrs
        self.net_depth = params['net_depth'] # the depth of the network (counting pooling + convolution + fulcon layers)
        self.n_conv = params['n_conv']  # number of convolutional layers
        self.n_fulcon = params['n_fulcon'] # number of fully connected layers
        self.conv_ids = params['conv_ids'] # ids of the convolution layers
        self.fulcon_ids = params['fulcon_ids'] # ids of the fulcon layers
        self.filter_bound_vec = params['filter_vector'] # a vector that gives the upper bound for each convolution and pooling layer ( use 0 for pooling)
        assert len(self.filter_bound_vec) == self.net_depth, 'net_depth (%d) parameter does not match the size of the filter bound vec (%d)'%(self.net_depth,len(self.filter_bound_vec))
        self.min_filter_threshold = params['filter_min_threshold'] # The minimum bound for each convolution layer
        self.min_fulcon_threshold = params['fulcon_min_threshold'] # The minimum neurons in a fulcon layer

        # Action related Hyperparameters
        self.actions = []
        self.actions.extend([('adapt', 0.0, conv_id) for conv_id in self.conv_ids])
        self.actions.extend([('adapt', 0.0, fc_id) for fc_id in self.fulcon_ids])
        self.actions.extend([('finetune', 0)])

        # Time steps in RL
        self.local_time_stamp = 0
        self.global_time_stamp = 0

        # Of format {s1,a1,s2,a2,s3,a3}
        # NOTE that this doesnt hold the current state
        self.state_history_length = params['state_history_length']

        # Loggers
        self.verbose_logger, self.q_logger, self.reward_logger, self.action_logger = None, None, None, None
        self.setup_loggers()

        # RL Agent Input/Output sizes
        self.local_actions, self.global_actions = 1, 3
        self.output_size = len(self.actions)
        self.actor_input_size = self.calculate_input_size(constants.TF_ACTOR_SCOPE)
        self.critic_input_size = self.calculate_input_size(constants.TF_CRITIC_SCOPE)

        # RL Agent netowrk sizes
        self.actor_layer_info, self.critic_layer_info = [],[]
        self.layer_scopes = []

        # Behavior of the RL Agent

        # Experience related hyperparameters
        self.q_length = 25 * self.output_size # length of the experience
        self.state_history_collector = []
        self.state_history_dumped = False
        self.experience_per_action = 25
        self.exp_clean_interval = 25

        self.current_state_history = []
        # Format of {phi(s_t),a_t,r_t,phi(s_t+1)}
        self.experience = []

        self.previous_reward = 0
        self.prev_prev_pool_accuracy = 0

        # Accumulating random states for q_rand^eval metric
        self.rand_state_accum_rate = 0.25
        self.rand_state_length = params['rand_state_length']
        self.rand_state_list = []

        # Stoping adaptaion criteria related things
        self.threshold_stop_adapting = 25  # fintune should not increase for this many steps
        self.ft_saturated_count = 0
        self.max_q_ft = -1000
        self.stop_adapting = False
        self.current_q_for_actions = None

        # Trial Phase (Usually the first epoch related)
        self.trial_phase = 0
        self.trial_phase_threshold = params['trial_phase_threshold'] # After this threshold all actions will be taken deterministically (e-greedy)

        # Tensorflow ops for function approximators (neural nets) for q-learning
        self.TAU = 0.001
        self.session = params['session']
        self.learning_rate = params['learning_rate']
        self.momentum = params['momentum']

        self.tf_state_input, self.tf_action_input, self.tf_y_i_targets, self.tf_q_mask = None,None,None,None
        self.tf_critic_out_op, self.tf_actor_out_op = None, None
        self.tf_critic_target_out_op, self.tf_actor_target_out_op = None, None
        self.tf_critic_loss_op = None
        self.tf_actor_optimize_op, self.tf_critic_optimize_op = None, None
        self.tf_actor_target_update_op, self.tf_critic_target_update_op = None, None

        self.prev_action, self.prev_state = None, None

        self.top_k_accuracy = params['top_k_accuracy']

        self.setup_tf_network_and_ops(params)


    def setup_tf_network_and_ops(self,params):
        '''
        Setup Tensorflow based Multi-Layer Perceptron and TF Operations
        :param params:
        :return:
        '''

        self.actor_layer_info = [self.actor_input_size]
        self.critic_layer_info = [self.critic_input_size]

        self.input_size = self.net_depth+self.binned_data_dist_length

        for h_i,hidden in enumerate(params['hidden_layers']):
            self.layer_scopes.append('fulcon_%d'%h_i)
            self.actor_layer_info.append(hidden)  # 128,64,32
            self.critic_layer_info.append(hidden)
        self.layer_scopes.append('fulcon_out')
        self.actor_layer_info.append(self.output_size)
        self.critic_layer_info.append(1)

        self.verbose_logger.info('Target Network Layer sizes: %s', self.actor_layer_info)

        self.momentum = params['momentum']  # 0.9
        self.learning_rate = params['learning_rate']  # 0.005

        # Initialize both actor and critic networks
        self.tf_init_actor_and_critic()

        # Input and output placeholders
        self.tf_state_input = tf.placeholder(tf.float32, shape=(None, self.input_size), name='InputDataset')
        self.tf_action_input = tf.placeholder(tf.float32, shape=(None, self.output_size), name='InputDataset')
        self.tf_y_i_targets = tf.placeholder(tf.float32, shape=(None, self.output_size), name='TargetDataset')

        # output of each network
        self.tf_critic_out_op = self.tf_calc_critic_output(self.tf_state_input,self.tf_action_input)
        self.tf_actor_out_op = self.tf_calc_actor_output(self.tf_state_input)
        self.tf_critic_target_out_op = self.tf_calc_actor_critic_target_output(self.tf_state_input, self.tf_action_input,
                                                           constants.TF_CRITIC_SCOPE)
        self.tf_actor_target_out_op = self.tf_calc_actor_critic_target_output(self.tf_state_input, self.tf_action_input,
                                                         constants.TF_ACTOR_SCOPE)

        self.tf_critic_loss_op = self.tf_mse_loss_of_critic(self.tf_y_i_targets, self.tf_critic_out_op)
        self.tf_critic_optimize_op = self.tf_momentum_optimize(self.tf_critic_loss_op)

        self.tf_actor_optimize_op = self.tf_policy_gradient_optimize()
        self.tf_actor_target_update_op = self.tf_train_actor_or_critic_target(constants.TF_ACTOR_SCOPE)
        self.tf_critic_target_update_op = self.tf_train_actor_or_critic_target(constants.TF_CRITIC_SCOPE)

        all_variables = []
        for w, b, wt, bt in zip(self.tf_weights, self.tf_bias, self.tf_target_weights, self.tf_target_biase):
            all_variables.extend([w, b, wt, bt])
        init_op = tf.variables_initializer(all_variables)
        _ = self.session.run(init_op)

    def setup_loggers(self):
        '''
        Setting up loggers
        verbose_logger - Log general information for viewing purposes
        q_logger - Log predicted q values at each time step
        reward_logger - Log the reward, action for a given time stamp
        action_logger - Actions taken every step
        :return:
        '''

        self.verbose_logger = logging.getLogger('verbose_q_learner_logger')
        self.verbose_logger.propagate = False
        self.verbose_logger.setLevel(logging.DEBUG)
        vHandler = logging.FileHandler(self.persit_dir + os.sep + 'ada_cnn_qlearner.log', mode='w')
        vHandler.setLevel(logging.INFO)
        vHandler.setFormatter(logging.Formatter('%(message)s'))
        self.verbose_logger.addHandler(vHandler)
        v_console = logging.StreamHandler(sys.stdout)
        v_console.setFormatter(logging.Formatter(logging_format))
        v_console.setLevel(logging_level)
        self.verbose_logger.addHandler(v_console)

        self.q_logger = logging.getLogger('q_logger')
        self.q_logger.propagate = False
        self.q_logger.setLevel(logging.INFO)
        qHandler = logging.FileHandler(self.persit_dir + os.sep + 'q_logger.log', mode='w')
        qHandler.setFormatter(logging.Formatter('%(message)s'))
        self.q_logger.addHandler(qHandler)
        self.q_logger.info(self.get_action_string_for_logging())

        self.reward_logger = logging.getLogger('reward_logger')
        self.reward_logger.propagate = False
        self.reward_logger.setLevel(logging.INFO)
        rewarddistHandler = logging.FileHandler(self.persit_dir + os.sep + 'action_reward_.log', mode='w')
        rewarddistHandler.setFormatter(logging.Formatter('%(message)s'))
        self.reward_logger.addHandler(rewarddistHandler)
        self.reward_logger.info('#global_time_stamp:batch_id:action_list:prev_pool_acc:pool_acc:reward')

        self.action_logger = logging.getLogger('action_logger')
        self.action_logger.propagate = False
        self.action_logger.setLevel(logging.INFO)
        actionHandler = logging.FileHandler(self.persit_dir + os.sep + 'actions_.log', mode='w')
        actionHandler.setFormatter(logging.Formatter('%(message)s'))
        self.action_logger.addHandler(actionHandler)

        self.advantage_logger = logging.getLogger('adavantage_logger')
        self.advantage_logger.propagate = False
        self.advantage_logger.setLevel(logging.INFO)
        actionHandler = logging.FileHandler(self.persit_dir + os.sep + 'advantage.log',mode='w')
        actionHandler.setFormatter(logging.Formatter('%(message)s'))
        self.action_logger.addHandler(actionHandler)

    def calculate_input_size(self, actor_or_critic_scope):
        '''
        Calculate input size for MLP (depends on the length of the history)
        :return:
        '''
        dummy_state = [0 for _ in range(self.net_depth+self.binned_data_dist_length)]

        if actor_or_critic_scope==constants.TF_ACTOR_SCOPE:
            return len(self.phi(dummy_state))
        elif actor_or_critic_scope==constants.TF_CRITIC_SCOPE:
            return len(self.phi(dummy_state)) + len(self.actions)

    def phi(self, si):
        '''
        Takes a state history [(s_t-2,a_t-2),(s_t-1,a_t-1),(s_t,a_t),s_t+1] and convert it to
        [s_t-2,a_t-2,a_t-1,a_t,s_t+1]
        a_n is a one-hot-encoded vector
        :param state_history:
        :return:
        '''

        self.verbose_logger.debug('Converting state history to phi')
        return si

    # ==================================================================
    # All neural network related TF operations

    def tf_init_actor_and_critic(self):
        '''
        Initialize the variables for neural network used for q learning
        :return:
        '''
        with tf.variable_scope(constants.TF_ACTOR_SCOPE):
            # Defining actor network
            for li in range(len(self.actor_layer_info) - 1):
                with tf.variable_scope(self.layer_scopes[li]):
                    tf.get_variable(initializer=tf.truncated_normal([self.actor_layer_info[li], self.actor_layer_info[li + 1]],
                                                                           stddev=2. / self.actor_layer_info[li]),
                                                       name=constants.TF_WEIGHTS)
                    tf.get_variable(initializer=tf.zeros([self.actor_layer_info[li + 1]]),
                                    name=constants.TF_BIAS)

                    with tf.variable_scope(constants.TF_TARGET_NET_SCOPE):

                        tf.get_variable(initializer=tf.zeros([self.actor_layer_info[li], self.actor_layer_info[li + 1]],
                                                        dtype=tf.float32),name=constants.TF_WEIGHTS)
                        tf.get_variable(initializer=tf.zeros([self.actor_layer_info[li + 1]]), name=constants.TF_BIAS)

        with tf.variable_scope(constants.TF_CRITIC_SCOPE):
            # Defining critic network
            for li in range(len(self.critic_layer_info) - 1):
                with tf.variable_scope(self.layer_scopes[li]):
                    tf.get_variable(initializer=tf.truncated_normal([self.critic_layer_info[li], self.critic_layer_info[li + 1]],
                                                                           stddev=2. / self.actor_layer_info[li]),
                                                       name=constants.TF_WEIGHTS)
                    tf.get_variable(initializer=tf.zeros([self.critic_layer_info[li + 1]]), name=constants.TF_BIAS)

                    with tf.variable_scope(constants.TF_TARGET_NET_SCOPE):
                        tf.get_variable(initializer=tf.zeros(shape=[self.critic_layer_info[li], self.critic_layer_info[li + 1]],
                                                        dtype=tf.float32),name=constants.TF_WEIGHTS)
                        tf.get_variable(initializer=tf.zeros([self.critic_layer_info[li + 1]]), name=constants.TF_BIAS)



    def tf_calc_critic_output(self, tf_state_input, tf_action_input):
        '''
        Calculate the output of the actor/critic network (quickly updated one)
        '''
        x = tf.concat([tf_state_input,tf_action_input],axis=1)

        with tf.variable_scope(constants.TF_CRITIC_SCOPE, reuse=True):
            for scope in self.layer_scopes:
                with tf.variable_scope(scope, reuse=True):
                    if scope != self.layer_scopes[-1]:
                        x = utils.lrelu(tf.matmul(x, tf.get_variable(constants.TF_WEIGHTS)) + tf.get_variable(
                            constants.TF_BIAS))
                    else:
                        x = tf.matmul(x, tf.get_variable(constants.TF_WEIGHTS)) + tf.get_variable(constants.TF_BIAS)

        return x

    def tf_calc_actor_output(self, tf_state_input):
        '''
        Calculate the output of the actor/critic network (quickly updated one)
        '''

        x = tf_state_input
        with tf.variable_scope(constants.TF_ACTOR_SCOPE, reuse=True):
            for scope in self.layer_scopes:
                with tf.variable_scope(scope, reuse=True):
                    if scope != self.layer_scopes[-1]:
                        x = utils.lrelu(tf.matmul(x, tf.get_variable(constants.TF_WEIGHTS)) + tf.get_variable(
                            constants.TF_BIAS))
                    else:
                        x = tf.nn.tanh(tf.matmul(x, tf.get_variable(constants.TF_WEIGHTS)) + tf.get_variable(
                            constants.TF_BIAS))
        return x

    def tf_calc_actor_critic_target_output(self, tf_state_input, tf_action_input, actor_or_critic_scope):
        '''
        Calculate the output of the target actor/critic network (slowly updated one)
        '''
        if actor_or_critic_scope==constants.TF_ACTOR_SCOPE:
            x = tf_state_input
        elif actor_or_critic_scope==constants.TF_CRITIC_SCOPE:
            x = tf.concat([tf_state_input,tf_action_input],axis=1)

        with tf.variable_scope(actor_or_critic_scope, reuse=True):
            for scope in self.layer_scopes:
                with tf.variable_scope(scope, reuse=True):
                    with tf.variable_scope(constants.TF_TARGET_NET_SCOPE, reuse = True):
                        if scope != self.layer_scopes[-1]:
                            x = utils.lrelu(tf.matmul(x, tf.get_variable(constants.TF_WEIGHTS))+tf.get_variable(constants.TF_BIAS))
                        else:
                            if actor_or_critic_scope==constants.TF_CRITIC_SCOPE:
                                x = tf.matmul(x, tf.get_variable(constants.TF_WEIGHTS)) + tf.get_variable(constants.TF_BIAS)
                            elif actor_or_critic_scope==constants.TF_ACTOR_SCOPE:
                                x = tf.nn.tanh(tf.matmul(x, tf.get_variable(constants.TF_WEIGHTS)) + tf.get_variable(constants.TF_BIAS))
                            else:
                                raise NotImplementedError
        return x

    def tf_mse_loss_of_critic(self, tf_y_i_output, tf_q_given_s_i):
        '''
        Calculate the squared loss between target and output
        :param tf_output:
        :param tf_targets:
        :return:
        '''
        return tf.reduce_mean((tf_y_i_output - tf_q_given_s_i) ** 2)


    def tf_momentum_optimize(self, loss):
        '''
        Optimizes critic
        :param loss:
        :return:
        '''
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                               momentum=self.momentum).minimize(loss)
        return optimizer

    def tf_policy_gradient_optimize(self):

        mu_s = self.tf_calc_actor_output(self.tf_state_input)
        theta_mu = self.get_all_variables(constants.TF_ACTOR_SCOPE, False)
        q_grad = tf.gradients(ys=self.tf_calc_critic_output(self.tf_state_input, self.tf_action_input),
                                   xs= mu_s)
        print(q_grad)
        negative_grads = []
        for g,v in q_grad:
            negative_grads.append((-g,v))
        # grad_ys acts as a way of chaining multiple gradients
        # more info: https://stackoverflow.com/questions/42399401/use-of-grads-ys-parameter-in-tf-gradients-tensorflow
        mu_grad = tf.gradients(ys= mu_s,
                     xs= theta_mu,
                     grad_ys = negative_grads)
        grads = zip(mu_grad,theta_mu)

        grad_apply_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                              momentum=self.momentum).apply_gradients(grads)

        return grad_apply_op

    def get_all_variables(self,actor_or_critic_scope, is_target_network):
        vars = []
        with tf.variable_scope(actor_or_critic_scope, reuse=True):
            for scope in self.layer_scopes:
                with tf.variable_scope(scope, reuse=True):
                    if not is_target_network:
                        vars.extend([tf.get_variable(constants.TF_WEIGHTS),
                                     tf.get_variable(constants.TF_BIAS)])
                    else:
                        with tf.variable_scope(constants.TF_TARGET_NET_SCOPE,reuse = True):
                            vars.extend([tf.get_variable(constants.TF_WEIGHTS),
                                         tf.get_variable(constants.TF_BIAS)])

        return vars

    def tf_target_weight_copy_op(self):
        '''
        Copy the weights from frequently updated RL agent to Target Network
        :return:
        '''
        update_ops = []
        for li, (w, b) in enumerate(zip(self.tf_weights, self.tf_bias)):
            update_ops.append(tf.assign(self.tf_target_weights[li], w))
            update_ops.append(tf.assign(self.tf_target_biase[li], b))

        return update_ops

    # ============================================================================

    def clean_experience(self):
        '''
        Delete past experience to free memory
        :return:
        '''
        self.verbose_logger.info('Cleaning Q values (removing old ones)')
        self.verbose_logger.debug('\tSize of Q before: %d', len(self.q))
        if len(self.q) > self.q_length:
            for _ in range(len(self.q) - self.q_length):
                self.q.popitem(last=True)
            self.verbose_logger.debug('\tSize of Q after: %d', len(self.q))

    def get_action_string_for_logging(self):
        '''
        Action string for logging purposes (predicted_q.log)
        :return:
        '''
        action_str = 'Time-Stamp,'

        for ci in self.conv_ids + self.fulcon_ids:
            action_str += 'Remove-%d,'%ci

        action_str+='Finetune,'
        action_str+='NaiveTrain'

        return action_str

    def update_trial_phase(self, trial_phase):
        self.trial_phase = trial_phase

    def get_current_q_vector(self):
        return self.current_q_for_actions


    def get_action_string(self, layer_action_list):
        act_string = ''
        for li, la in enumerate(layer_action_list):
            if la is None:
                continue
            else:
                act_string += la[0] + str(la[1])

        return act_string

    def normalize_state(self, s):
        '''
        Normalize the layer filter count to [-1, 1]
        :param s: current state
        :return:
        '''
        # state looks like [distMSE, filter_count_1, filter_count_2, ...]
        norm_state = np.zeros((1, self.net_depth))
        self.verbose_logger.debug('Before normalization: %s', s)
        # enumerate only the depth related part of the state
        for ii, item in enumerate(s[:self.net_depth]):
            if self.filter_bound_vec[ii] > 0:
                norm_state[0, ii] = item * 1.0 - (self.filter_bound_vec[ii]/2.0)
                norm_state[0, ii] /= (self.filter_bound_vec[ii]/2.0)
            else:
                norm_state[0, ii] = -1.0

        # concatenate binned distributions and normalized layer depth
        norm_state = np.append(norm_state,np.reshape(s[self.net_depth:],(1,-1)),axis=1)
        self.verbose_logger.debug('\tNormalized state: %s\n', norm_state)
        return tuple(norm_state.flatten())

    def get_ohe_state_ndarray(self, s):
        return np.asarray(self.normalize_state(s)).reshape(1, -1)

    def clean_experience(self):
        '''
        Clean experience to reduce the memory requirement
        We keep a
        :return:
        '''
        exp_action_count = {}
        for e_i, [_, ai, _, _, time_stamp] in enumerate(self.experience):
            # phi_t, a_idx, reward, phi_t_plus_1
            a_idx = ai
            if a_idx not in exp_action_count:
                exp_action_count[a_idx] = [(time_stamp, e_i)]
            else:
                exp_action_count[a_idx].append((time_stamp, e_i))

        indices_to_remove = []
        for k, v in exp_action_count.items():
            sorted_v = sorted(v, key=lambda item: item[0])
            if len(v) > self.experience_per_action:
                indices_to_remove.extend(sorted_v[:len(sorted_v) - self.experience_per_action])

        indices_to_remove = sorted(indices_to_remove, reverse=True)

        self.verbose_logger.info('Indices of experience that will be removed')
        self.verbose_logger.info('\t%s', indices_to_remove)

        for _, r_i in indices_to_remove:  # each element in indices to remove are tuples (time_stamp,exp_index)
            self.experience.pop(r_i)

        exp_action_count = {}
        for e_i, [_, ai, _, _, _] in enumerate(self.experience):
            # phi_t, a_idx, reward, phi_t_plus_1
            a_idx = ai
            if a_idx not in exp_action_count:
                exp_action_count[a_idx] = [e_i]
            else:
                exp_action_count[a_idx].append(e_i)

        # np.random.shuffle(self.experience) # decorrelation

        self.verbose_logger.debug('Action count after removal')
        self.verbose_logger.debug(exp_action_count)

    def get_s_a_r_s_with_experince(self, experience_slice):

        x, y, rewards, sj = None, None, None, None

        for [si, ai, reward, s_i_plus_1, time_stamp] in experience_slice:
            # phi_t, a_idx, reward, phi_t_plus_1
            if x is None:
                x = np.asarray(self.phi(si)).reshape((1, -1))
            else:
                x = np.append(x, np.asarray(self.phi(si)).reshape((1, -1)), axis=0)

            if y is None:
                y = np.asarray(ai).reshape(1, -1)
            else:
                y = np.append(y, np.asarray(ai).reshape(1, -1), axis=0)

            if rewards is None:
                rewards = np.asarray(reward).reshape(1, -1)
            else:
                rewards = np.append(rewards, np.asarray(reward).reshape(1, -1), axis=0)

            if sj is None:
                sj = np.asarray(self.phi(s_i_plus_1)).reshape(1, -1)
            else:
                sj = np.append(sj, np.asarray(self.phi(s_i_plus_1)).reshape(1, -1), axis=0)

        return x, y, rewards, sj

    def exploration_noise_OU(self, a, mu, theta, sigma):
        return theta * (mu - a) + sigma * np.random.randn(1.0)

    def get_action_with_exploration(self):
        '''
        Returns a(t) = mu(s(t)|theta_mu) + N(t) where in is the exploration policy
        N(t) =
        :return:
        '''

    def get_complexity_penalty(self, curr_comp, prev_comp, filter_bound_vec):


        # total gain should be negative for taking add action before half way througl a layer
        # total gain should be positve for taking add action after half way througl a layer
        total = 0
        split_factor = 0.6
        for l_i,(c_depth, p_depth, up_dept) in enumerate(zip(curr_comp,prev_comp,filter_bound_vec)):
            if up_dept>0 and abs(c_depth-p_depth) > 0:
                total += (((up_dept*split_factor)-c_depth)/(up_dept*split_factor))

        if sum(curr_comp)-sum(prev_comp)>0.0:
            return - total * (self.top_k_accuracy/self.num_classes)
        elif sum(curr_comp)-sum(prev_comp)<0.0:
            return total * (self.top_k_accuracy/self.num_classes)
        else:
            return 0.0

    def tf_train_actor_or_critic_target(self,actor_or_critic_scope):
        target_assign_ops = []
        with tf.variable_scope(actor_or_critic_scope, reuse=True):
            for scope in self.layer_scopes:
                with tf.variable_scope(scope, reuse=True):
                    w_dash,b_dash = tf.get_variable(constants.TF_WEIGHTS), tf.get_variable(constants.TF_BIAS)
                    with tf.variables_scope(constants.TF_TARGET_NET_SCOPE, reuse=True):
                        w, b = tf.get_variable(constants.TF_WEIGHTS), tf.get_variable(constants.TF_BIAS)
                        target_assign_ops.append(tf.assign(w, self.TAU * w + (1-self.TAU)* w_dash))
                        target_assign_ops.append(tf.assign(b, self.TAU * b + (1 - self.TAU) * b_dash))

        return target_assign_ops

    def train_critic(self, experience_batch):
        # data['prev_state'], data['curr_state']

        # sample a batch from experience
        s_i, a_i , r, s_i_plus_1 = self.get_s_a_r_s_with_experince(experience_batch)

        self.verbose_logger.debug('Summary of Experience data')
        self.verbose_logger.debug('\ts(t):%s', s_i.shape)
        self.verbose_logger.debug('\ta(t):%s', a_i.shape)
        self.verbose_logger.debug('\tr:%s', r.shape)
        self.verbose_logger.debug('\ts(t+1):%s', s_i_plus_1.shape)

        # predicte Q(s,a|theta_Q) with the critic
        mu_s_i_plus_1 = self.session.run(self.tf_actor_target_out_op, feed_dict={self.tf_state_input: s_i_plus_1})
        q_given_s_i_plus_1_mu_s_i_plus_1 = self.session.run(self.tf_critic_target_out_op,
                                                            feed_dict={self.tf_state_input: s_i_plus_1,
                                                                       self.tf_action_input: mu_s_i_plus_1})
        y_i = r + self.discount_rate*q_given_s_i_plus_1_mu_s_i_plus_1

        _ = self.session.run(self.tf_critic_optimize_op,
                             feed_dict={self.tf_state_input:s_i,self.tf_action_input:a_i,
                                        self.tf_y_i_targets:y_i})

    def train_actor(self,experience_batch):
        '''
        Train the actor with a batch sampled from the experience
        Gradient update
        1/N * Sum(d Q(s,a|theta_Q)/d mu(s_i)* d mu(s_i|theta_mu)/d theta_mu)
        :param experience_batch:
        :return:
        '''
        s_i, a_i, r, s_i_plus_1 = self.get_s_a_r_s_with_experince(experience_batch)
        _ = self.session.run(self.tf_actor_optimize_op,feed_dict={
            self.tf_state_input:s_i
        })

    def sample_action_stochastic_from_actor(self,data):

        state = []
        state.extend(data['filter_counts_list'])
        state.extend(data['binned_data_dist'])

        self.verbose_logger.info('Data for (Depth Index,DistMSE,Filter Count) %s\n' % str(state))

        self.verbose_logger.debug('Epsilons: %.3f\n', self.epsilon)
        self.verbose_logger.info('Trial phase: %.3f\n', self.trial_phase)

        self.verbose_logger.debug('Getting new action according the the Actor')
        s_i = np.asarray(self.phi(state)).reshape(1, -1)
        cont_actions_all_layers = self.session.run(self.tf_actor_out_op, feed_dict={self.tf_state_input: s_i})
        cont_actions_all_layers = cont_actions_all_layers.flatten()

        cont_actions_all_layers += self.exploration_noise_OU(cont_actions_all_layers,0.0,0.3,1.0)

        valid_action = self.get_new_valid_action_when_stochastic(
             cont_actions_all_layers, data
        )

        return valid_action

    def sample_action_deterministic_from_actor(self,data):

        state = []
        state.extend(data['filter_counts_list'])
        state.extend(data['binned_data_dist'])

        self.verbose_logger.info('Data for (Depth Index,DistMSE,Filter Count) %s\n' % str(state))
        self.verbose_logger.debug('Epsilons: %.3f\n', self.epsilon)
        self.verbose_logger.info('Trial phase: %.3f\n', self.trial_phase)

        self.verbose_logger.debug('Getting new action according the the Actor')
        s_i = np.asarray(self.phi(state)).reshape(1, -1)
        cont_actions_all_layers = self.session.run(self.tf_actor_out_op, feed_dict={self.tf_state_input: s_i})
        cont_actions_all_layers = cont_actions_all_layers.flatten()

        valid_action = self.get_new_valid_action_when_stochastic(
             cont_actions_all_layers, data
        )

        return state, valid_action

    def get_new_valid_action_when_stochastic(self, action, data):

        self.verbose_logger.debug('Getting new valid action (stochastic)')

        valid_action = action.tolist()

        for a_idx, a in enumerate(valid_action):
            layer_id_for_action = None

            # For Convolution layers
            if a_idx < len(self.conv_ids):
                layer_id_for_action = self.conv_ids[a_idx]

                next_layer_complexity = data['filter_counts_list'][layer_id_for_action] + ceil(a * self.add_amount)

                if next_layer_complexity < self.min_filter_threshold or next_layer_complexity > self.filter_bound_vec[layer_id_for_action]:
                    valid_action[a_idx] = 0.0

            # For fully-connected layers
            elif a_idx < len(self.conv_ids) + len(self.fulcon_ids):
                layer_id_for_action = self.fulcon_ids[a_idx - len(self.conv_ids)]

                next_layer_complexity = data['filter_counts_list'][layer_id_for_action] + ceil(a * self.add_fulcon_amount)

                if next_layer_complexity < self.min_fulcon_threshold or \
                                next_layer_complexity > self.filter_bound_vec[layer_id_for_action]:

                    valid_action[a_idx] = 0.0
            # For finetune action there is no invalid state
            else:
                continue

        return valid_action

    def get_new_valid_action_when_deterministic(self, action, data):

        self.verbose_logger.debug('Getting new valid action (stochastic)')

        valid_action = action.tolist()

        for a_idx, a in enumerate(valid_action):
            layer_id_for_action = None

            # For Convolution layers
            if a_idx < len(self.conv_ids):
                layer_id_for_action = self.conv_ids[a_idx]

                next_layer_complexity = data['filter_counts_list'][layer_id_for_action] + ceil(a * self.add_amount)

                if next_layer_complexity < self.min_filter_threshold or next_layer_complexity > self.filter_bound_vec[layer_id_for_action]:
                    valid_action[a_idx] = 0.0

            # For fully-connected layers
            elif a_idx < len(self.conv_ids) + len(self.fulcon_ids):
                layer_id_for_action = self.fulcon_ids[a_idx - len(self.conv_ids)]

                next_layer_complexity = data['filter_counts_list'][layer_id_for_action] + ceil(a * self.add_fulcon_amount)

                if next_layer_complexity < self.min_fulcon_threshold or \
                                next_layer_complexity > self.filter_bound_vec[layer_id_for_action]:
                    if valid_action[a_idx]>0.0:
                        valid_action[a_idx] = np.random.random(-0.1,0.0)
                    elif valid_action[a_idx]<0.0:
                        valid_action[a_idx] = np.random.random(0.0, 0.1)
                        valid_action[a_idx] = np.random.random(0.0, 0.1)
            # For finetune action there is no invalid state
            else:
                continue

        return valid_action


    def train_actor_critic(self, data):
        # data['prev_state']
        # data['prev_action']
        # data['curr_state']
        # data['next_accuracy']
        # data['prev_accuracy']
        # data['batch_id']

        if self.global_time_stamp > 0 and len(self.experience) > 0:
            self.verbose_logger.info('Training the Q Approximator with Experience...')
            self.verbose_logger.debug('(Q) Total experience data: %d', len(self.experience))

            # =====================================================
            # Returns a batch of experience
            # ====================================================
            if len(self.experience) > self.batch_size:
                exp_indices = np.random.randint(0, len(self.experience), (self.batch_size,))
                self.verbose_logger.debug('Experience indices: %s', exp_indices)
                self.train_actor([self.experience[ei] for ei in exp_indices])
                self.train_critic([self.experience[ei] for ei in exp_indices])
            else:
                self.train_actor(self.experience)
                self.train_critic(self.experience)

            # Removing old experience to save memory
            if self.global_time_stamp > 0 and self.global_time_stamp % self.exp_clean_interval == 0:
                self.clean_experience()

        si, ai, sj = data['prev_state'], data['prev_action'], data['curr_state']
        self.verbose_logger.debug('Si,Ai,Sj: %s,%s,%s', si, ai, sj)

        comp_gain = self.get_complexity_penalty(data['curr_state'], data['prev_state'], self.filter_bound_vec)

        accuracy_push_reward = self.top_k_accuracy/self.num_classes if (data['prev_pool_accuracy'] - data['pool_accuracy'])/100.0<= self.top_k_accuracy/self.num_classes \
            else (data['prev_pool_accuracy'] - data['pool_accuracy'])/100.0

        mean_accuracy = accuracy_push_reward if data['pool_accuracy'] > data['max_pool_accuracy'] else -accuracy_push_reward
        #immediate_mean_accuracy = (1.0 + ((data['unseen_valid_accuracy'] + data['prev_unseen_valid_accuracy'])/200.0))*\
        #                          (data['unseen_valid_accuracy'] - data['prev_unseen_valid_accuracy']) / 100.0

        self.verbose_logger.info('Complexity penalty: %.5f', comp_gain)
        self.verbose_logger.info('Pool Accuracy: %.5f ', mean_accuracy)
        self.verbose_logger.info('Max Pool Accuracy: %.5f ', data['max_pool_accuracy'])

        reward = mean_accuracy - comp_gain #+ 0.5*immediate_mean_accuracy # new
        # if complete_do_nothing:
        #    reward = -1e-3# * max(self.same_action_count+1,5)

        self.reward_logger.info("%d:%d:%s:%.3f:%.3f:%.5f", self.global_time_stamp, data['batch_id'], ai,
                                data['prev_pool_accuracy'], data['pool_accuracy'], reward)

        self.update_experience()

        self.previous_reward = reward
        self.prev_prev_pool_accuracy = data['prev_pool_accuracy']

        self.local_time_stamp += 1
        self.global_time_stamp += 1

        self.verbose_logger.info('Global/Local time step: %d/%d\n', self.global_time_stamp, self.local_time_stamp)


    def update_experience(self,si,ai,reward,sj):


        # update experience

        self.experience.append([si, ai, reward, sj, self.global_time_stamp])

        if self.global_time_stamp < 3:
            self.verbose_logger.debug('Latest Experience: ')
            self.verbose_logger.debug('\t%s\n', self.experience[-1])

        self.verbose_logger.info('Update Summary ')
        self.verbose_logger.info('\tState: %s', si)
        self.verbose_logger.info('\tAction:%s', ai)
        self.verbose_logger.info('\tReward: %.3f', reward)


    def get_average_Q(self):
        x = None
        if len(self.rand_state_list) == self.rand_state_length:
            for s_t in self.rand_state_list:
                s_t = np.asarray(s_t).reshape(1, -1)
                if x is None:
                    x = s_t
                else:
                    x = np.append(x, s_t, axis=0)

            self.verbose_logger.debug('Shape of x: %s', x.shape)
            q_pred = self.session.run(self.tf_actor_out_op, feed_dict={self.tf_state_input: x})
            self.verbose_logger.debug('Shape of q_pred: %s', q_pred.shape)
            return np.mean(np.max(q_pred, axis=1))
        else:
            return 0.0

    def reset_loggers(self):
        self.verbose_logger.handlers = []
        self.action_logger.handlers = []
        self.reward_logger.handlers = []
        self.q_logger.handlers = []

    def get_stop_adapting_boolean(self):
        return self.stop_adapting
