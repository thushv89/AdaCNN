__author__ = 'Thushan Ganegedara'

from enum import IntEnum
from collections import defaultdict
# from sklearn.gaussian_process import GaussianProcessRegressor
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

from collections import OrderedDict

logging_level = logging.INFO
logging_format = '[%(name)s] [%(funcName)s] %(message)s'


class AdaCNNAdaptingQLearner(object):
    def __init__(self, **params):

        self.discount_rate = params['discount_rate']

        self.num_classes = params['num_classes']

        self.fit_interval = params['fit_interval']  # GP calculating interval (10)
        self.target_update_rate = params['target_update_rate']

        self.persit_dir = params['persist_dir']
        self.q_logger = logging.getLogger('pred_q_logger')
        self.q_logger.setLevel(logging.INFO)
        q_distHandler = logging.FileHandler(self.persit_dir + os.sep + 'predicted_q.log', mode='w')
        q_distHandler.setFormatter(logging.Formatter('%(message)s'))
        self.q_logger.addHandler(q_distHandler)

        self.reward_logger = logging.getLogger('reward_logger')
        self.reward_logger.setLevel(logging.INFO)
        rewarddistHandler = logging.FileHandler(self.persit_dir + os.sep + 'reward.log', mode='w')
        rewarddistHandler.setFormatter(logging.Formatter('%(message)s'))
        self.reward_logger.addHandler(rewarddistHandler)
        self.reward_logger.info('#global_time_stamp:batch_id:action_list:prev_pool_acc:pool_acc:reward')

        self.action_logger = logging.getLogger('action_logger')
        self.action_logger.setLevel(logging.INFO)
        actionHandler = logging.FileHandler(self.persit_dir + os.sep + 'actions.log', mode='w')
        actionHandler.setFormatter(logging.Formatter('%(message)s'))
        self.action_logger.addHandler(actionHandler)

        self.batch_size = params['batch_size']

        self.net_depth = params['net_depth']
        self.n_conv = params['n_conv']  # number of convolutional layers
        self.conv_ids = params['conv_ids']
        self.random_mode = params['random_mode']

        # self.filter_bound_vec = self.make_filter_bound_vector(self.filter_upper_bound, self.net_depth,self.conv_ids,params['impose_pyramid_structure'])
        self.filter_bound_vec = params['filter_vector']

        self.rl_logger = logging.getLogger('Adapting Policy Logger')
        self.rl_logger.setLevel(logging_level)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(logging_format))
        console.setLevel(logging_level)
        self.rl_logger.addHandler(console)
        self.rl_logger.debug('Conv ids: %s', self.conv_ids)
        self.rl_logger.debug('Filter vector %s', self.filter_bound_vec)
        self.local_time_stamp = 0
        self.global_time_stamp = 0

        self.actions = [
            ('do_nothing', 0), ('finetune', 0),
            ('add', params['add_amount']), ('remove', params['remove_amount'])
        ]
        self.add_amount = params['add_amount']

        self.q_logger.info('#%s', self.actions)

        self.q_length = 25 * len(self.actions)

        self.past_mean_accuracy = 0

        self.prev_action, self.prev_state = None, None
        self.same_action_count = [0 for _ in range(self.net_depth)]
        self.epsilon = params['epsilon']

        self.same_action_threshold = 50

        # Of format {s1,a1,s2,a2,s3,a3} NOTE that this doesnt hold the current state
        self.state_history_length = params['state_history_length']

        self.session = params['session']

        self.local_actions, self.global_actions = 2, 2
        self.output_size = self.calculate_output_size()
        self.input_size = self.calculate_input_size()

        self.explore_tries = self.output_size * params['exploratory_tries_factor']
        self.explore_interval = params['exploratory_interval']
        self.stop_exploring_after = params['stop_exploring_after']

        self.layer_info = [self.input_size]
        for hidden in params['hidden_layers']:
            self.layer_info.append(hidden)  # 128,64,32
        self.layer_info.append(self.output_size)

        self.rl_logger.info('Target Network Layer sizes: %s', self.layer_info)

        self.current_state_history = []
        # Format of {phi(s_t),a_t,r_t,phi(s_t+1)}
        self.experience = []

        self.tf_weights, self.tf_bias = [], []
        self.tf_target_weights, self.tf_target_biase = [], []

        self.momentum = params['momentum']  # 0.9
        self.learning_rate = params['learning_rate']  # 0.005

        self.tf_init_mlp()
        self.tf_state_input = tf.placeholder(tf.float32, shape=(None, self.input_size), name='InputDataset')
        self.tf_q_targets = tf.placeholder(tf.float32, shape=(None, self.output_size), name='TargetDataset')

        self.tf_out_op = self.tf_calc_output(self.tf_state_input)
        self.tf_out_target_op = self.tf_calc_output_target(self.tf_state_input)
        self.tf_loss_op = self.tf_sqr_loss(self.tf_out_op, self.tf_q_targets)
        self.tf_optimize_op = self.tf_momentum_optimize(self.tf_loss_op)

        self.tf_target_update_ops = self.tf_target_weight_copy_op()

        all_variables = []
        for w, b, wt, bt in zip(self.tf_weights, self.tf_bias, self.tf_target_weights, self.tf_target_biase):
            all_variables.extend([w, b, wt, bt])
        init_op = tf.variables_initializer(all_variables)
        _ = self.session.run(init_op)

        self.state_history_collector = []
        self.state_history_dumped = False
        self.experience_per_action = 25
        self.exp_clean_interval = 25

        self.min_filter_threshold = params['filter_min_threshold']

        self.min_epsilon = 0.1
        self.previous_reward = 0
        self.prev_prev_pool_accuracy = 0

        self.rand_state_accum_rate = 0.25
        self.rand_state_length = params['rand_state_length']
        self.rand_state_list = []

        self.trial_phase = 0

        self.threshold_stop_adapting = 25  # fintune should not increase for this many steps
        self.ft_saturated_count = 0
        self.max_q_ft = -1000
        self.stop_adapting = False

        self.trial_phase_threshold = params['trial_phase_threshold']

    def calculate_output_size(self):
        total = 0
        for _ in range(self.local_actions):  # add and remove actions
            total += self.n_conv

        total += self.global_actions  # finetune and donothing
        return total

    def calculate_input_size(self):
        dummy_state = [0 for _ in range(self.n_conv)]
        dummy_state = tuple(dummy_state)

        dummy_action = tuple([0 for _ in range(self.output_size)])
        dummy_history = []
        for _ in range(self.state_history_length - 1):
            dummy_history.append([dummy_state, dummy_action])
        dummy_history.append([dummy_state])
        self.rl_logger.debug('Dummy history')
        self.rl_logger.debug('\t%s\n', dummy_history)
        self.rl_logger.debug('Input Size: %d', len(self.phi(dummy_history)))
        return len(self.phi(dummy_history))

    def phi(self, state_history):
        '''
        Takes a state history [(s_t-2,a_t-2),(s_t-1,a_t-1),(s_t,a_t),s_t+1] and convert it to
        [s_t-2,a_t-2,a_t-1,a_t,s_t+1]
        a_n is a one-hot-encoded vector
        :param state_history:
        :return:
        '''
        self.rl_logger.debug('Got: %s', state_history)
        preproc_input = []
        for iindex, item in enumerate(state_history):
            if iindex == 0:  # first state
                preproc_input.extend(list(self.normalize_state(item[0])))
                preproc_input.extend(item[1])
            elif iindex != len(state_history) - 1:
                preproc_input.extend(item[1])
            else:  # last state
                preproc_input.extend(list(self.normalize_state(item[0])))

        self.rl_logger.debug('Returning: %s\n', preproc_input)
        assert len(state_history) == self.state_history_length
        return preproc_input

    def tf_init_mlp(self):
        for li in range(len(self.layer_info) - 1):
            self.tf_weights.append(tf.Variable(tf.truncated_normal([self.layer_info[li], self.layer_info[li + 1]],
                                                                   stddev=2. / self.layer_info[li]),
                                               name='weights_' + str(li) + '_' + str(li + 1)))
            self.tf_target_weights.append(
                tf.Variable(tf.truncated_normal([self.layer_info[li], self.layer_info[li + 1]],
                                                stddev=2. / self.layer_info[li]),
                            name='target_weights_' + str(li) + '_' + str(li + 1)))
            self.tf_bias.append(
                tf.Variable(tf.zeros([self.layer_info[li + 1]]), name='bias_' + str(li) + '_' + str(li + 1)))
            self.tf_target_biase.append(
                tf.Variable(tf.zeros([self.layer_info[li + 1]]), name='target_bias_' + str(li) + '_' + str(li + 1)))

    def tf_calc_output(self, tf_state_input):
        x = tf_state_input
        for li, (w, b) in enumerate(zip(self.tf_weights[:-1], self.tf_bias[:-1])):
            x = tf.nn.relu(tf.matmul(x, w) + b)

        return tf.matmul(x, self.tf_weights[-1]) + self.tf_bias[-1]

    def tf_calc_output_target(self, tf_state_input):
        x = tf_state_input
        for li, (w, b) in enumerate(zip(self.tf_target_weights[:-1], self.tf_target_biase[:-1])):
            x = tf.nn.relu(tf.matmul(x, w) + b)

        return tf.matmul(x, self.tf_weights[-1]) + self.tf_bias[-1]

    def tf_sqr_loss(self, tf_output, tf_targets):
        return tf.reduce_mean((tf_output - tf_targets) ** 2)

    def tf_momentum_optimize(self, loss):
        optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                               momentum=self.momentum).minimize(loss)
        return optimizer

    def tf_target_weight_copy_op(self):
        update_ops = []
        for li, (w, b) in enumerate(zip(self.tf_weights, self.tf_bias)):
            update_ops.append(tf.assign(self.tf_target_weights[li], w))
            update_ops.append(tf.assign(self.tf_target_biase[li], b))

        return update_ops

    def make_filter_bound_vector(self, n_fil, n_layer, conv_ids, pyramid):
        fil_vec = []
        if pyramid:
            curr_fil = n_fil / 2 ** (len(conv_ids) - 1)
            for li in range(n_layer):
                if li in conv_ids:
                    fil_vec.append(max(self.filter_min_bound, int(curr_fil)))
                    curr_fil *= 2
                else:
                    fil_vec.append(0)

            assert len(fil_vec) == n_layer
        else:
            for li in range(n_layer):
                if li in conv_ids:
                    fil_vec.append(int(n_fil))
                else:
                    fil_vec.append(0)

            assert len(fil_vec) == n_layer
        return fil_vec

    def restore_policy(self, **restore_data):
        # use this to restore from saved data
        self.q = restore_data['q']
        self.regressor = restore_data['lrs']

    def clean_Q(self):
        self.rl_logger.debug('Cleaning Q values (removing old ones)')
        self.rl_logger.debug('\tSize of Q before: %d', len(self.q))
        if len(self.q) > self.q_length:
            for _ in range(len(self.q) - self.q_length):
                self.q.popitem(last=True)
            self.rl_logger.debug('\tSize of Q after: %d', len(self.q))

    # how action_idx turned into action list
    # convert the idx to binary representation (example 10=> 0b1010)
    # get text [2:] to discard first two letters
    # prepend 0s to it so the length is equal to number of conv layers

    def action_list_with_index(self, action_idx):
        self.rl_logger.debug('Got: %d\n', action_idx)
        layer_actions = [None for _ in range(self.net_depth)]
        self.rl_logger.debug('Layer actions: %s\n', layer_actions)

        if action_idx < self.output_size - self.global_actions:
            primary_action = action_idx // self.n_conv  # action
            secondary_action = action_idx % self.n_conv  # the layer the action will be executed
            if primary_action == 0:
                tmp_a = self.actions[3]
            elif primary_action == 1:
                tmp_a = self.actions[2]
            # elif primary_action==2:
            #    tmp_a = self.actions[4]

            for ci, c_id in enumerate(self.conv_ids):
                if ci == secondary_action:
                    layer_actions[c_id] = tmp_a
                else:
                    layer_actions[c_id] = self.actions[0]

        elif action_idx == self.output_size - 2:
            layer_actions = [self.actions[0] if li in self.conv_ids else None for li in range(self.net_depth)]
        elif action_idx == self.output_size - 1:
            layer_actions = [self.actions[1] if li in self.conv_ids else None for li in range(self.net_depth)]

        assert len(layer_actions) == self.net_depth
        self.rl_logger.debug('Return: %s\n', layer_actions)
        return layer_actions

    def index_from_action_list(self, action_list):
        self.rl_logger.debug('Got: %s\n', action_list)
        if self.get_action_string(action_list) == \
                self.get_action_string(
                    [self.actions[0] if li in self.conv_ids else None for li in range(self.net_depth)]):
            self.rl_logger.debug('Return: %d\n', self.output_size - 2)
            return self.output_size - 2
        elif self.get_action_string(action_list) == \
                self.get_action_string(
                    [self.actions[1] if li in self.conv_ids else None for li in range(self.net_depth)]):
            self.rl_logger.debug('Return: %d\n', self.output_size - 1)
            return self.output_size - 1

        else:
            conv_id = 0
            for li, la in enumerate(action_list):
                if la is None:
                    continue

                if la == self.actions[2]:
                    secondary_idx = conv_id
                    primary_idx = 1
                elif la == self.actions[3]:
                    secondary_idx = conv_id
                    primary_idx = 0
                # elif la==self.actions[4]:
                #    secondary_idx = conv_id
                #    primary_idx= 2
                conv_id += 1

            action_idx = primary_idx * self.n_conv + secondary_idx
            self.rl_logger.debug('Primary %d Secondary %d', primary_idx, secondary_idx)
            self.rl_logger.debug('Return: %d\n', action_idx)
            return action_idx

    def update_trial_phase(self, trial_phase):
        self.trial_phase = trial_phase

    def output_action(self, data):
        invalid_actions = []
        # data => ['distMSE']['filter_counts']
        # ['filter_counts'] => depth_index : filter_count
        # State => Layer_Depth (w.r.t net), dist_MSE, number of filters in layer

        action_type = None  # for logging purpose
        state = []  # removed distMSE (believe doesn't have much value)
        state.extend(data['filter_counts_list'])

        self.rl_logger.info('Data for (Depth Index,DistMSE,Filter Count) %s\n' % str(state))
        history_t_plus_1 = list(self.current_state_history)
        history_t_plus_1.append([state])

        self.rl_logger.debug('Current state history: %s\n', self.current_state_history)
        self.rl_logger.debug('history_t+1:%s\n', history_t_plus_1)
        self.rl_logger.debug('Epsilons: %.3f\n', self.epsilon)
        self.rl_logger.info('Trial phase: %.3f\n', self.trial_phase)

        n_conv_first_half = int(ceil(self.n_conv * 1.0 / 2.0))
        n_conv_last_half = int(self.n_conv - n_conv_first_half)
        assert n_conv_first_half + n_conv_last_half == self.n_conv
        if self.trial_phase < self.trial_phase_threshold:

            if self.trial_phase <= self.trial_phase_threshold / 5.0:
                # more add actions
                # actions are indexed as [{remove actions},{add actions},{do_nothing,finetune}]
                trial_action_probs = [0.2 / (1.0 * self.n_conv) for _ in range(self.n_conv)]  # remove
                trial_action_probs.extend([0.6 / (1.0 * self.n_conv) for _ in range(self.n_conv)])  # add
                trial_action_probs.extend([0.05, 0.15])
            elif self.trial_phase <= self.trial_phase_threshold / 2.5:
                trial_action_probs = [0 / (1.0 * self.n_conv) for _ in range(self.n_conv)]  # remove
                trial_action_probs.extend([0 / (1.0 * self.n_conv) for _ in range(self.n_conv)])  # add
                trial_action_probs.extend([0.0, 1.0])
            elif self.trial_phase <= self.trial_phase_threshold / 2.0:
                trial_action_probs = [0.2 / (1.0 * self.n_conv) for _ in range(self.n_conv)]  # remove
                trial_action_probs.extend([0.6 / (1.0 * self.n_conv) for _ in range(self.n_conv)])  # add
                trial_action_probs.extend([0.05, 0.15])

            elif self.trial_phase <= self.trial_phase_threshold / 1.43:
                trial_action_probs = [0.7 / (1.0 * self.n_conv) for _ in range(self.n_conv)]  # remove
                trial_action_probs.extend([0.1 / (1.0 * self.n_conv) for _ in range(self.n_conv)])  # add
                trial_action_probs.extend([0.05, 0.15])
            elif self.trial_phase <= self.trial_phase_threshold / 1.11:
                trial_action_probs = [0 / (1.0 * self.n_conv) for _ in range(self.n_conv)]  # remove
                trial_action_probs.extend([0 / (1.0 * self.n_conv) for _ in range(self.n_conv)])  # add
                trial_action_probs.extend([0.0, 1.0])
            else:
                trial_action_probs = [0.7 / (1.0 * self.n_conv) for _ in range(self.n_conv)]  # remove
                trial_action_probs.extend([0.1 / (1.0 * self.n_conv) for _ in range(self.n_conv)])  # add
                trial_action_probs.extend([0.05, 0.15])

            action_idx = np.random.choice(self.output_size, p=trial_action_probs)
            layer_actions_list = self.action_list_with_index(action_idx)

            found_valid_action = False

            # while loop for checkin the validity of the action and choosing another if not
            while not found_valid_action and action_idx < self.output_size - 2:

                # if chosen action is do_nothing or finetune
                if action_idx >= self.output_size - 2:
                    found_valid_action = True
                    break

                # check for all layers if the chosen action is valid
                for li, la in enumerate(layer_actions_list):
                    if la is None:
                        continue

                    if la[0] == 'add':
                        next_filter_count = data['filter_counts_list'][li] + la[1]
                    elif la[0] == 'remove':
                        next_filter_count = data['filter_counts_list'][li] - la[1]
                    else:
                        next_filter_count = data['filter_counts_list'][li]

                    # if action is invalid, remove that from the allowed actions
                    if next_filter_count <= self.min_filter_threshold or next_filter_count > self.filter_bound_vec[
                        li]:
                        self.rl_logger.debug('\tAction %s is not valid li(%d), (Next Filter Count: %d). ' % (
                            str(la), li, next_filter_count))

                        # invalid_actions.append(action_idx)
                        found_valid_action = False
                        # udpate current action to another action
                        action_idx = np.random.choice(self.output_size, p=trial_action_probs)

                        layer_actions_list = self.action_list_with_index(action_idx)
                        self.rl_logger.debug('\tSelected new action: %s', layer_actions_list)
                        break
                    else:
                        found_valid_action = True

            if action_idx >= self.output_size - 2:
                found_valid_action = True
            assert found_valid_action

            if len(history_t_plus_1) == self.state_history_length:
                curr_x = np.asarray(self.phi(history_t_plus_1)).reshape(1, -1)
                q_for_actions = self.session.run(self.tf_out_target_op, feed_dict={self.tf_state_input: curr_x})
                q_for_actions = q_for_actions.flatten().tolist()

                q_value_strings = ''
                for q_val in q_for_actions:
                    q_value_strings += '%.5f' % q_val + ','
                self.q_logger.info("%d,%s", self.local_time_stamp, q_value_strings)
                self.rl_logger.debug('\tPredicted Q: %s', q_for_actions[:10])

            if len(self.rand_state_list) < self.rand_state_length and \
                            np.random.random() < self.rand_state_accum_rate and \
                            len(history_t_plus_1) == self.state_history_length:
                self.rand_state_list.append(self.phi(history_t_plus_1))

        # deterministic selection (if epsilon is not 1 or q is not empty)
        elif np.random.random() > self.epsilon and len(history_t_plus_1) == self.state_history_length:
            self.rl_logger.info('Choosing action deterministic...')
            # we create this copy_actions in case we need to change the order the actions processed
            # without changing the original action space (self.actions)

            curr_x = np.asarray(self.phi(history_t_plus_1)).reshape(1, -1)
            q_for_actions = self.session.run(self.tf_out_target_op, feed_dict={self.tf_state_input: curr_x})
            q_for_actions = q_for_actions.flatten().tolist()

            q_value_strings = ''
            for q_val in q_for_actions:
                q_value_strings += '%.5f' % q_val + ','
            self.q_logger.info("%d,%s", self.local_time_stamp, q_value_strings)
            self.rl_logger.debug('\tPredicted Q: %s', q_for_actions[:10])

            # Finding when to stop adapting
            # for this we choose the point the finetune operation has the
            # maximum utility compared to other actions and itself previously
            if self.trial_phase > self.trial_phase_threshold * 1.5 and np.argmax(q_for_actions) == self.output_size - 1:
                if q_for_actions[-1] > self.max_q_ft:
                    assert self.trial_phase >= 1.0
                    self.max_q_ft = q_for_actions[-1]
                    self.ft_saturated_count = 0
                else:
                    self.ft_saturated_count += 1

                if self.ft_saturated_count > self.threshold_stop_adapting:
                    self.stop_adapting = True
            else:
                self.ft_saturated_count = 0

            action_type = 'Deterministic'
            action_idx = np.asscalar(np.argmax(q_for_actions))

            if np.random.random() < 0.25:
                action_idx = np.asscalar(np.argsort(q_for_actions).flatten()[-2])

            layer_actions_list = self.action_list_with_index(action_idx)
            self.rl_logger.debug('\tChose: %s' % str(layer_actions_list))

            # ================= Look ahead 1 step (Validate Predicted Action) =========================
            # make sure predicted action stride is not larger than resulting output.
            # make sure predicted kernel size is not larger than the resulting output
            # To avoid such scenarios we create a restricted action space if this happens and chose from that

            found_valid_action = False
            allowed_actions = [tmp for tmp in range(self.output_size)]

            # while loop for checkin the validity of the action and choosing another if not
            while len(q_for_actions) > 0 and not found_valid_action and action_idx < self.output_size - 2:

                # if chosen action is do_nothing or finetune
                if action_idx >= self.output_size - 2:
                    found_valid_action = True
                    break

                # check for all layers if the chosen action is valid
                for li, la in enumerate(layer_actions_list):
                    if la is None:
                        continue

                    if la[0] == 'add':
                        next_filter_count = data['filter_counts_list'][li] + la[1]
                    elif la[0] == 'remove':
                        next_filter_count = data['filter_counts_list'][li] - la[1]
                    else:
                        next_filter_count = data['filter_counts_list'][li]

                    # if action is invalid, remove that from the allowed actions
                    if next_filter_count <= self.min_filter_threshold or next_filter_count > self.filter_bound_vec[li]:
                        self.rl_logger.debug('\tAction %s is not valid li(%d), (Next Filter Count: %d). ' % (
                        str(la), li, next_filter_count))
                        try:
                            del q_for_actions[action_idx]
                        except:
                            self.rl_logger.critical('Error Length Q (%d) Action idx (%d)', len(q_for_actions),
                                                    action_idx)
                            self.rl_logger.critical('\tAction %s is not valid li(%d), (Next Filter Count: %d). ',
                                                    str(la), li, next_filter_count)
                        allowed_actions.remove(action_idx)
                        invalid_actions.append(action_idx)
                        found_valid_action = False

                        # udpate current action to another action
                        max_idx = np.asscalar(np.argmax(q_for_actions))
                        action_idx = allowed_actions[max_idx]
                        layer_actions_list = self.action_list_with_index(action_idx)
                        self.rl_logger.debug('\tSelected new action: %s', layer_actions_list)
                        break
                    else:
                        found_valid_action = True

            if action_idx >= self.output_size - 2:
                found_valid_action = True
            assert found_valid_action

        # random selection
        else:
            self.rl_logger.info('Choosing action stochastic...')
            action_type = 'Stochastic'

            curr_x = np.asarray(self.phi(history_t_plus_1)).reshape(1, -1)
            q_for_actions = self.session.run(self.tf_out_target_op, feed_dict={self.tf_state_input: curr_x})

            # not to restrict from the beginning
            if self.global_time_stamp > self.stop_exploring_after:
                rand_indices = np.argsort(q_for_actions).flatten()[
                               ceil(2.0 * self.output_size / 4.0):]  # Only get a random index from the highest q values
                self.rl_logger.info('Allowed action indices: %s', rand_indices)
                action_idx = np.random.choice(rand_indices)
            else:
                action_idx = np.random.randint(0, self.output_size)
            layer_actions_list = self.action_list_with_index(action_idx)
            self.rl_logger.debug('\tChose: %s' % str(layer_actions_list))

            # ================= Look ahead 1 step (Validate Predicted Action) =========================
            # make sure predicted action stride is not larger than resulting output.
            # make sure predicted kernel size is not larger than the resulting output
            # To avoid such scenarios we create a
            #  action space if this happens

            # Check if the next filter count is invalid for any layer
            found_valid_action = False
            if self.global_time_stamp > self.stop_exploring_after:
                allowed_actions = np.argsort(q_for_actions).flatten()[floor(
                    1.0 * self.output_size / 4.0):]  # Only get a random index from the highest q values
                allowed_actions = allowed_actions.tolist()
            else:
                allowed_actions = [tmp for tmp in range(self.output_size)]

            while not found_valid_action and action_idx < self.output_size - 2:
                self.rl_logger.debug('Checking action validity')
                if action_idx >= self.output_size - 2:
                    found_valid_action = True
                    break

                for li, la in enumerate(layer_actions_list):
                    if la is None:
                        continue
                    elif la[0] == 'add':
                        next_filter_count = data['filter_counts_list'][li] + la[1]
                    elif la[0] == 'remove':
                        next_filter_count = data['filter_counts_list'][li] - la[1]
                    else:
                        next_filter_count = data['filter_counts_list'][li]

                    if next_filter_count <= self.min_filter_threshold or next_filter_count > self.filter_bound_vec[li]:
                        self.rl_logger.debug('\tAction %s is not valid li(%d), (Next Filter Count: %d). ', str(la), li,
                                             next_filter_count)
                        allowed_actions.remove(action_idx)
                        invalid_actions.append(action_idx)
                        found_valid_action = False

                        action_idx = np.random.choice(allowed_actions)
                        layer_actions_list = self.action_list_with_index(action_idx)
                        self.rl_logger.debug('\tSelected new action: %s', layer_actions_list)
                        break
                    else:
                        found_valid_action = True

            if action_idx >= self.output_size - 2:
                found_valid_action = True
            assert found_valid_action

        # decay epsilon
        if self.trial_phase >= self.trial_phase_threshold:
            self.epsilon = max(self.epsilon * 0.9, self.min_epsilon)

            # TODO: same action taking repeatedly
            # this is to reduce taking the same action over and over again
            # if self.same_action_count >= self.same_action_threshold:
            #    self.epsilon = min(self.epsilon*1.01,1.0)

        self.rl_logger.debug('=' * 60)
        self.rl_logger.debug('State')
        self.rl_logger.debug(state)
        self.rl_logger.debug('Action')
        self.rl_logger.debug(layer_actions_list)
        self.rl_logger.debug('=' * 60)

        if self.prev_action is not None and \
                        self.get_action_string(layer_actions_list) == self.get_action_string(self.prev_action):
            self.same_action_count += 1

        else:
            self.same_action_count = 0

        self.action_logger.info('%s,%s,%s,%.3f', action_type, state, layer_actions_list, self.epsilon)

        self.prev_action = layer_actions_list
        self.prev_state = state

        self.rl_logger.info('\tSelected action: %s\n', layer_actions_list)

        return state, layer_actions_list, invalid_actions

    def get_action_string(self, layer_action_list):
        act_string = ''
        for li, la in enumerate(layer_action_list):
            if la is None:
                continue
            else:
                act_string += la[0] + str(la[1])

        return act_string

    def normalize_state(self, s):
        # state looks like [distMSE, filter_count_1, filter_count_2, ...]
        ohe_state = np.zeros((1, self.net_depth))
        self.rl_logger.debug('Before normalization: %s', s)
        for ii, item in enumerate(s):
            if self.filter_bound_vec[ii] > 0:
                ohe_state[0, ii] = item * 1.0 / self.filter_bound_vec[ii]
            else:
                ohe_state[0, ii] = 0
        self.rl_logger.debug('\tNormalized state: %s\n', ohe_state)
        return tuple(ohe_state.flatten())

    def get_ohe_state_ndarray(self, s):
        return np.asarray(self.normalize_state(s)).reshape(1, -1)

    def clean_experience(self):
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

        self.rl_logger.info('Indices of experience that will be removed')
        self.rl_logger.info('\t%s', indices_to_remove)

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

        self.rl_logger.debug('Action count after removal')
        self.rl_logger.debug(exp_action_count)

    def get_xy_with_experince(self, experience_slice):

        x, y, rewards, sj = None, None, None, None

        for [hist_t, ai, reward, hist_t_plus_1, time_stamp] in experience_slice:
            # phi_t, a_idx, reward, phi_t_plus_1
            if x is None:
                x = np.asarray(self.phi(hist_t)).reshape((1, -1))
            else:
                x = np.append(x, np.asarray(self.phi(hist_t)).reshape((1, -1)), axis=0)

            ohe_a = [1 if ai == act else 0 for act in range(self.output_size)]
            if y is None:
                y = np.asarray(ohe_a).reshape(1, -1)
            else:
                y = np.append(y, np.asarray(ohe_a).reshape(1, -1), axis=0)

            if rewards is None:
                rewards = np.asarray(reward).reshape(1, -1)
            else:
                rewards = np.append(rewards, np.asarray(reward).reshape(1, -1), axis=0)

            if sj is None:
                sj = np.asarray(self.phi(hist_t_plus_1)).reshape(1, -1)
            else:
                sj = np.append(sj, np.asarray(self.phi(hist_t_plus_1)).reshape(1, -1), axis=0)

        return x, y, rewards, sj

    def update_policy(self, data, add_future_reward):
        # data['prev_state']
        # data['prev_action']
        # data['curr_state']
        # data['next_accuracy']
        # data['prev_accuracy']
        # data['batch_id']
        if not self.random_mode:

            if self.global_time_stamp > 0 and len(
                    self.experience) > 0 and self.global_time_stamp % self.fit_interval == 0:
                self.rl_logger.info('Training the Q Approximator with Experience...')
                self.rl_logger.debug('(Q) Total experience data: %d', len(self.experience))

                if len(self.experience) > self.batch_size:
                    exp_indices = np.random.randint(0, len(self.experience), (self.batch_size,))
                    self.rl_logger.debug('Experience indices: %s', exp_indices)
                    x, y, r, next_state = self.get_xy_with_experince([self.experience[ei] for ei in exp_indices])
                else:
                    x, y, r, next_state = self.get_xy_with_experince(self.experience)

                if self.global_time_stamp < 5:
                    assert np.max(x) <= 1.0 and np.max(x) >= -1.0 and np.max(y) <= 1.0 and np.max(y) >= -1.0

                self.rl_logger.debug('Summary of Structured Experience data')
                self.rl_logger.debug('\tX:%s', x.shape)
                self.rl_logger.debug('\tY:%s', y.shape)
                self.rl_logger.debug('\tR:%s', r.shape)
                self.rl_logger.debug('\tNextState:%s', next_state.shape)

                pred_q = self.session.run(self.tf_out_target_op, feed_dict={self.tf_state_input: x})
                self.rl_logger.debug('\tPredicted %s:', pred_q.shape)
                target_q = r.flatten() + self.discount_rate * np.max(pred_q, axis=1).flatten()

                self.rl_logger.debug('\tTarget Q %s:', target_q.shape)
                self.rl_logger.debug('\tTarget Q Values %s:', target_q[:5])
                assert target_q.size <= self.batch_size

                y = np.multiply(y, target_q.reshape(-1, 1))

                # since the state contain layer id, let us make the layer id one-hot encoded
                self.rl_logger.debug('X (shape): %s, Y (shape): %s', x.shape, y.shape)
                self.rl_logger.debug('X: \n%s, Y: \n%s', str(x[:3, :]), str(y[:3]))

                # self.regressor.partial_fit(x, y)
                _ = self.session.run([self.tf_loss_op, self.tf_optimize_op], feed_dict={
                    self.tf_state_input: x, self.tf_q_targets: y
                })

                if self.global_time_stamp % self.target_update_rate == 0 and self.local_time_stamp % self.n_conv == 0:
                    self.rl_logger.info('Coppying the Q approximator as the Target Network')
                    # self.target_network = self.regressor.partial_fit(x, y)
                    _ = self.session.run([self.tf_target_update_ops])

                if self.global_time_stamp > 0 and self.global_time_stamp % self.exp_clean_interval == 0:
                    self.clean_experience()

        mean_accuracy = (data['pool_accuracy'] - data['prev_pool_accuracy']) / 100.0
        max_accuracy = (data['pool_accuracy'] - data['max_pool_accuracy']) / 100.0

        si, ai_list, sj = data['prev_state'], data['prev_action'], data['curr_state']
        self.rl_logger.debug('Si,Ai,Sj: %s,%s,%s', si, ai_list, sj)

        aux_penalty, prev_aux_penalty = 0, 0
        for li, la in enumerate(ai_list):
            if la is None:
                continue
            if la[0] == 'add':
                assert sj[li] == si[li] + la[1]
                break
            elif la[0] == 'remove':
                assert sj[li] == si[li] - la[1]
                break
            elif la[0] == 'replace':
                break
            else:
                continue

        # if all actions are do_nothing randomly apply a penalty
        # complete_do_nothing = False
        # for li,la in enumerate(ai_list):
        #    if la is None:
        #        continue
        #    if la[0]=='do_nothing':
        #        complete_do_nothing = True
        #    else:
        #        complete_do_nothing = False
        #        break

        reward = mean_accuracy  # new
        curr_action_string = self.get_action_string(ai_list)

        # exponential magnifier to prevent from being taken consecutively
        if self.trial_phase > 1.0:
            if ('add' in curr_action_string or 'remove' in curr_action_string) and self.same_action_count >= 1:
                self.rl_logger.info('Reward before magnification: %.5f', reward)
                if reward > 0:
                    reward = reward  # /= min(self.same_action_count + 1,10)
                else:
                    reward *= min(self.same_action_count + 1, 10)
                self.rl_logger.info('Reward after magnification: %.5f', reward)

            # encourage taking finetune action consecutively
            if 'finetune' in curr_action_string and self.same_action_count >= 1:
                self.rl_logger.info('Reward before magnification: %.5f', reward)
                if reward > 0:
                    reward *= min(self.same_action_count + 1, 10)
                else:  # new
                    self.same_action_count = 0  # reset action count # new
                self.rl_logger.info('Reward after magnification: %.5f', reward)

        # if complete_do_nothing:
        #    reward = -1e-3# * max(self.same_action_count+1,5)

        self.reward_logger.info("%d:%d:%s:%.3f:%.3f:%.5f", self.global_time_stamp, data['batch_id'], ai_list,
                                data['prev_pool_accuracy'], data['pool_accuracy'], reward)
        # how the update on state_history looks like
        # t=5 (s2,a2),(s3,a3),(s4,a4)
        # t=6 (s3,a3),(s4,a4),(s5,a5)
        # add previous action (the action we know the reward for) i.e not the current action
        # as a one-hot vector

        # phi_t (s_t-3,a_t-3),(s_t-2,a_t-2),(s_t-1,a_t-1),(s_t,a_t)
        history_t = list(self.current_state_history)
        history_t.append([si])
        self.rl_logger.debug('History(t)')
        self.rl_logger.debug('%s\n', history_t)

        assert len(history_t) <= self.state_history_length

        action_idx = self.index_from_action_list(ai_list)

        # update current state history
        self.current_state_history.append([si])
        self.current_state_history[-1].append([1 if action_idx == act else 0 for act in range(self.output_size)])

        if len(self.current_state_history) > self.state_history_length - 1:
            del self.current_state_history[0]
            assert len(self.current_state_history) == self.state_history_length - 1

        self.rl_logger.debug('Current History')
        self.rl_logger.debug('%s\n', self.current_state_history)

        history_t_plus_1 = list(self.current_state_history)
        history_t_plus_1.append([sj])
        assert len(history_t_plus_1) <= self.state_history_length

        # update experience
        if len(history_t) >= self.state_history_length:
            self.experience.append([history_t, action_idx, reward, history_t_plus_1, self.global_time_stamp])

            if add_future_reward and len(self.experience) > 2:
                prev_action_string = self.get_action_string(self.action_list_with_index(self.experience[-2][1]))
                curr_action_string = self.get_action_string(ai_list)
                # this is because the reward can be delayed
                # if 'finetune' in curr_action_string and ('add' in prev_action_string or 'remove' in prev_action_string):
                #    self.rl_logger.info('Updating reward for %s', prev_action_string)
                #    self.rl_logger.info('\tFrom: %.3f',self.experience[-2][2])
                #    self.experience[-2][2] += 0.5*max(0,(data['pool_accuracy'] - self.prev_prev_pool_accuracy)/100.0)
                #    self.rl_logger.info('\tTo: %.3f\n', self.experience[-2][2])

            for invalid_a in data['invalid_actions']:
                self.rl_logger.debug('Adding the invalid action %s to experience', invalid_a)
                if 'remove' in self.get_action_string(self.action_list_with_index(invalid_a)):
                    for _ in range(3):
                        self.experience.append(
                            [history_t, invalid_a, -1.0 / (self.num_classes), history_t_plus_1, self.global_time_stamp])
                    self.reward_logger.info("%d:%d:%s:%.3f:%.3f:%.5f", self.global_time_stamp, data['batch_id'],
                                            self.action_list_with_index(invalid_a), -1, -1, -1.0 / (self.num_classes))

                    opp_action = []
                    for la in self.action_list_with_index(invalid_a):
                        if la is None or la[0] == 'do_nothing':
                            opp_action.append(la)
                        if la is not None and la[0] == 'remove':
                            opp_action.append(('add', self.add_amount))
                    for _ in range(3):
                        self.experience.append(
                            [history_t, self.index_from_action_list(opp_action), 1.0 / (self.num_classes * 10.0),
                             history_t_plus_1, self.global_time_stamp])
                    self.reward_logger.info("%d:%d:%s:%.3f:%.3f:%.5f", self.global_time_stamp, data['batch_id'],
                                            opp_action, -1, -1, 1.0 / (self.num_classes * 10.0))

                else:
                    for _ in range(3):
                        self.experience.append(
                            [history_t, invalid_a, -1.0 / (self.num_classes * 10.0), history_t_plus_1,
                             self.global_time_stamp])
                    self.reward_logger.info("%d:%d:%s:%.3f:%.3f:%.5f", self.global_time_stamp, data['batch_id'],
                                            self.action_list_with_index(invalid_a), -1, -1, -1.0 / (self.num_classes))

            if self.global_time_stamp < 3:
                self.rl_logger.debug('Latest Experience: ')
                self.rl_logger.debug('\t%s\n', self.experience[-1])

        self.rl_logger.info('Update Summary ')
        self.rl_logger.info('\tState: %s', si)
        self.rl_logger.info('\tAction: %d,%s', action_idx, ai_list)
        self.rl_logger.info('\tReward: %.3f', reward)
        self.rl_logger.info('\t\tReward (Mean Acc): %.4f', mean_accuracy)

        self.previous_reward = reward
        self.prev_prev_pool_accuracy = data['prev_pool_accuracy']

        self.local_time_stamp += 1
        self.global_time_stamp += 1

        self.rl_logger.info('Global/Local time step: %d/%d\n', self.global_time_stamp, self.local_time_stamp)

    def get_average_Q(self):
        x = None
        if len(self.rand_state_list) == self.rand_state_length:
            for s_t in self.rand_state_list:
                s_t = np.asarray(s_t).reshape(1, -1)
                if x is None:
                    x = s_t
                else:
                    x = np.append(x, s_t, axis=0)

            self.rl_logger.debug('Shape of x: %s', x.shape)
            q_pred = self.session.run(self.tf_out_target_op, feed_dict={self.tf_state_input: x})
            self.rl_logger.debug('Shape of q_pred: %s', q_pred.shape)
            return np.mean(np.max(q_pred, axis=1))
        else:
            return 0.0

    def reset_loggers(self):
        self.rl_logger.handlers = []
        self.action_logger.handlers = []
        self.reward_logger.handlers = []
        self.q_logger.handlers = []

    def get_stop_adapting_boolean(self):
        return self.stop_adapting
