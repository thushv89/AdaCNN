
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

        # Action related Hyperparameters
        self.qlearner_type = params['qlearner_type']
        if self.qlearner_type=='growth':
            self.actions = [
                ('do_nothing', 0), ('finetune', 0),('naive_train', 0),
                ('add', params['add_amount'])
            ]
        elif self.qlearner_type=='prune':
            self.actions = [
                ('do_nothing', 0), ('finetune', 0),('naive_train', 0),
                ('remove', params['remove_amount'])
            ]
        else:
            raise AttributeError
        self.binned_data_dist_length = params['binned_data_dist_length']

        # RL Agent Specifict Hyperparameters
        self.discount_rate = params['discount_rate']
        self.fit_interval = params['fit_interval']  # RL agent training interval
        self.target_update_rate = params['target_update_rate']
        self.batch_size = params['batch_size']
        self.add_amount = params['add_amount']
        self.epsilon = params['epsilon']
        self.min_epsilon = 0.1

        self.num_classes = params['num_classes']
        self.persit_dir = params['persist_dir']

        # CNN specific Hyperparametrs
        self.net_depth = params['net_depth'] # the depth of the network (counting pooling + convolution layers)
        self.n_conv = params['n_conv']  # number of convolutional layers
        self.conv_ids = params['conv_ids'] # ids of the convolution layers
        self.filter_bound_vec = params['filter_vector'] # a vector that gives the upper bound for each convolution and pooling layer ( use 0 for pooling)
        self.min_filter_threshold = params['filter_min_threshold'] # The minimum bound for each convolution layer

        # Things used for peanalizing reward
        # e.g. Taking too many add actions without a significant reward
        self.same_action_threshold = 50
        self.same_action_count = [0 for _ in range(self.net_depth)]

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
        self.output_size = self.calculate_output_size()
        self.input_size = self.calculate_input_size()

        # Behavior of the RL Agent
        self.random_mode = params['random_mode']
        self.explore_tries = self.output_size * params['exploratory_tries_factor']
        self.explore_interval = params['exploratory_interval']
        self.stop_exploring_after = params['stop_exploring_after']

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
        self.session = params['session']
        self.tf_state_input, self.tf_q_targets, self.tf_q_mask = None,None,None
        self.tf_out_op,self.tf_out_target_op = None,None
        self.tf_weights, self.tf_bias = None, None
        self.tf_loss_op,self.tf_optimize_op = None,None
        self.setup_tf_network_and_ops(params)

        self.prev_action, self.prev_state = None, None



    def setup_tf_network_and_ops(self,params):
        '''
        Setup Tensorflow based Multi-Layer Perceptron and TF Operations
        :param params:
        :return:
        '''

        self.layer_info = [self.input_size]
        for hidden in params['hidden_layers']:
            self.layer_info.append(hidden)  # 128,64,32
        self.layer_info.append(self.output_size)

        self.verbose_logger.info('Target Network Layer sizes: %s', self.layer_info)

        self.tf_weights, self.tf_bias = [], []
        self.tf_target_weights, self.tf_target_biase = [], []

        self.momentum = params['momentum']  # 0.9
        self.learning_rate = params['learning_rate']  # 0.005

        self.tf_init_mlp()
        self.tf_state_input = tf.placeholder(tf.float32, shape=(None, self.input_size), name='InputDataset')
        self.tf_q_targets = tf.placeholder(tf.float32, shape=(None, self.output_size), name='TargetDataset')
        self.tf_q_mask = tf.placeholder(tf.float32, shape=(None, self.output_size), name='TargeMask')

        self.tf_out_op = self.tf_calc_output(self.tf_state_input)
        self.tf_out_target_op = self.tf_calc_output_target(self.tf_state_input)
        self.tf_loss_op = self.tf_sqr_loss(self.tf_out_op, self.tf_q_targets, self.tf_q_mask)
        self.tf_optimize_op = self.tf_momentum_optimize(self.tf_loss_op)

        self.tf_target_update_ops = self.tf_target_weight_copy_op()

        all_variables = []
        for w, b, wt, bt in zip(self.tf_weights, self.tf_bias, self.tf_target_weights, self.tf_target_biase):
            all_variables.extend([w, b, wt, bt])
        init_op = tf.variables_initializer(all_variables)
        _ = self.session.run(init_op)

    def get_finetune_action(self,data):
        state = data['filter_counts_list'] + data['binned_data_dist']
        return state, [self.actions[1] if li in self.conv_ids else None for li in range(self.net_depth)],[]

    def setup_loggers(self):
        '''
        Setting up loggers
        verbose_logger - Log general information for viewing purposes
        q_logger - Log predicted q values at each time step
        reward_logger - Log the reward, action for a given time stamp
        action_logger - Actions taken every step
        :return:
        '''

        self.verbose_logger = logging.getLogger('verbose_q_learner_logger_'+self.qlearner_type)
        self.verbose_logger.propagate = False
        self.verbose_logger.setLevel(logging.DEBUG)
        vHandler = logging.FileHandler(self.persit_dir + os.sep + 'ada_cnn_qlearner_' + self.qlearner_type  +'.log', mode='w')
        vHandler.setLevel(logging.INFO)
        vHandler.setFormatter(logging.Formatter('%(message)s'))
        self.verbose_logger.addHandler(vHandler)
        v_console = logging.StreamHandler(sys.stdout)
        v_console.setFormatter(logging.Formatter(logging_format))
        v_console.setLevel(logging_level)
        self.verbose_logger.addHandler(v_console)

        self.q_logger = logging.getLogger('pred_q_logger_'+self.qlearner_type)
        self.q_logger.propagate = False
        self.q_logger.setLevel(logging.INFO)
        q_distHandler = logging.FileHandler(self.persit_dir + os.sep + 'predicted_q_' + self.qlearner_type +'.log', mode='w')
        q_distHandler.setFormatter(logging.Formatter('%(message)s'))
        self.q_logger.addHandler(q_distHandler)
        self.q_logger.info(self.get_action_string_for_logging())

        self.reward_logger = logging.getLogger('reward_logger'+self.qlearner_type)
        self.reward_logger.propagate = False
        self.reward_logger.setLevel(logging.INFO)
        rewarddistHandler = logging.FileHandler(self.persit_dir + os.sep + 'reward_'+ self.qlearner_type +'.log', mode='w')
        rewarddistHandler.setFormatter(logging.Formatter('%(message)s'))
        self.reward_logger.addHandler(rewarddistHandler)
        self.reward_logger.info('#global_time_stamp:batch_id:action_list:prev_pool_acc:pool_acc:reward')

        self.action_logger = logging.getLogger('action_logger'+self.qlearner_type)
        self.action_logger.propagate = False
        self.action_logger.setLevel(logging.INFO)
        actionHandler = logging.FileHandler(self.persit_dir + os.sep + 'actions_'+ self.qlearner_type + '.log', mode='w')
        actionHandler.setFormatter(logging.Formatter('%(message)s'))
        self.action_logger.addHandler(actionHandler)

    def calculate_output_size(self):
        '''
        Calculate output size for MLP (depends on the number of layers and actions)
        :return:
        '''
        total = 0
        for _ in range(self.local_actions):  # add and remove actions
            total += self.n_conv

        total += self.global_actions  # finetune and donothing
        self.verbose_logger.info('Calculated output action space size: %d',total)

        return total

    def calculate_input_size(self):
        '''
        Calculate input size for MLP (depends on the length of the history)
        :return:
        '''
        dummy_state = [0 for _ in range(self.net_depth+self.binned_data_dist_length)]
        dummy_state = tuple(dummy_state)

        dummy_action = tuple([0 for _ in range(self.output_size)])
        dummy_history = []
        for _ in range(self.state_history_length - 1):
            dummy_history.append([dummy_state, dummy_action])
        dummy_history.append([dummy_state])

        self.verbose_logger.debug('Dummy history')
        self.verbose_logger.debug('\t%s\n', dummy_history)
        self.verbose_logger.debug('Input Size: %d', len(self.phi(dummy_history)))

        return len(self.phi(dummy_history))

    def phi(self, state_history):
        '''
        Takes a state history [(s_t-2,a_t-2),(s_t-1,a_t-1),(s_t,a_t),s_t+1] and convert it to
        [s_t-2,a_t-2,a_t-1,a_t,s_t+1]
        a_n is a one-hot-encoded vector
        :param state_history:
        :return:
        '''

        self.verbose_logger.debug('Converting state history to phi')
        self.verbose_logger.debug('Got (state_history): %s', state_history)
        preproc_input = []
        for iindex, item in enumerate(state_history):
            if iindex == 0:  # first state
                preproc_input.extend(list(self.normalize_state(item[0])))
                preproc_input.extend(item[1])
            elif iindex != len(state_history) - 1:
                preproc_input.extend(item[1])
            else:  # last state
                preproc_input.extend(list(self.normalize_state(item[0])))

        self.verbose_logger.debug('Returning (phi): %s\n', preproc_input)
        assert len(state_history) == self.state_history_length
        return preproc_input

    # ==================================================================
    # All neural network related TF operations

    def tf_init_mlp(self):
        '''
        Initialize the variables for neural network used for q learning
        :return:
        '''
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
        '''
        Calculate the output till the last layer
        Middle layers have relu activation
        Last layer is a linear layer
        :param tf_state_input:
        :return:
        '''
        x = tf_state_input
        for li, (w, b) in enumerate(zip(self.tf_weights[:-1], self.tf_bias[:-1])):
            x = tf.nn.relu(tf.matmul(x, w) + b)

        return tf.matmul(x, self.tf_weights[-1]) + self.tf_bias[-1]

    def tf_calc_output_target(self, tf_state_input):
        x = tf_state_input
        for li, (w, b) in enumerate(zip(self.tf_target_weights[:-1], self.tf_target_biase[:-1])):
            x = tf.nn.relu(tf.matmul(x, w) + b)

        return tf.matmul(x, self.tf_weights[-1]) + self.tf_bias[-1]

    def tf_sqr_loss(self, tf_output, tf_targets, tf_mask):
        '''
        Calculate the squared loss between target and output
        :param tf_output:
        :param tf_targets:
        :return:
        '''
        return tf.reduce_mean(((tf_output*tf_mask) - tf_targets) ** 2)

    def tf_momentum_optimize(self, loss):
        optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                               momentum=self.momentum).minimize(loss)
        return optimizer

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

    def restore_policy(self, **restore_data):
        # use this to restore from saved data
        self.q = restore_data['q']
        self.regressor = restore_data['lrs']

    def clean_Q(self):
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

    def action_list_with_index(self, action_idx):
        '''
        self.actions [('do_nothing', 0), ('finetune', 0),('naive_train', 0),
                ('remove', params['remove_amount'])]
        How action_idx turned into action list
          convert the idx to binary representation (example 10=> 0b1010)
          get text [2:] to discard first two letters
          prepend 0s to it so the length is equal to number of conv layers
        :param action_idx:
        :return: 0...n_conv for add actions corresponding to each layer n_conv + 1 -> do_nothing, n_conv+2 -> finetune n_conv + 3 -> naive_train
        '''
        self.verbose_logger.info('Getting action list from action index')
        self.verbose_logger.debug('Got (idx): %d\n', action_idx)
        layer_actions = [None for _ in range(self.net_depth)]

        if action_idx < self.output_size - self.global_actions:
            primary_action = action_idx // self.n_conv  # action
            secondary_action = action_idx % self.n_conv  # the layer the action will be executed
            if primary_action == 0:
                tmp_a = self.actions[3]

            for ci, c_id in enumerate(self.conv_ids):
                if ci == secondary_action:
                    layer_actions[c_id] = tmp_a
                else:
                    layer_actions[c_id] = self.actions[0]

        elif action_idx == self.output_size - 3:
            layer_actions = [self.actions[0] if li in self.conv_ids else None for li in range(self.net_depth)]
        elif action_idx == self.output_size - 2:
            layer_actions = [self.actions[1] if li in self.conv_ids else None for li in range(self.net_depth)]
        elif action_idx == self.output_size - 1:
            layer_actions = [self.actions[2] if li in self.conv_ids else None for li in range(self.net_depth)]

        assert len(layer_actions) == self.net_depth
        self.verbose_logger.debug('Return (action_list): %s\n', layer_actions)
        return layer_actions

    def get_action_string_for_logging(self):
        '''
        Action string for logging purposes (predicted_q.log)
        :return:
        '''
        action_str = 'Time-Stamp,'
        if self.qlearner_type=='prune':
            for ci in self.conv_ids:
                action_str += 'Remove-%d,'%ci
        elif self.qlearner_type=='growth':
            for ci in self.conv_ids:
                action_str += 'Add-%d,' % ci
        else:
            raise AttributeError

        action_str+='DoNothing,'
        action_str+='Finetune,'
        action_str+='NaiveTrain'

        return action_str

    def index_from_action_list(self, action_list):
        '''
        Index of an action with the action list
        Eg for 7 layer net with C,C,C,P,C,C,C (C-conv and P-pool) (n_conv = 6)
        for action_list [DoNothing, DoNothing, ...] action_idx = # of actions - 2
        for action_list [Finetune, Finetune, ...] action_idx = # of actions - 1
        for action_list [DoNothing, DoNothing, Add, None, DoNothing, DoNothing]
        since "Add" is in the action, primary index = 0
        and the secondary action is the layer index of which action is Add, so secondary index = 2
        and action_idx = 0 * n_conv + 2 = 2
        :param action_list:
        :return:
        '''
        self.verbose_logger.info('Getting action index from action list')
        self.verbose_logger.debug('Got: %s\n', action_list)
        if self.get_action_string(action_list) == \
                self.get_action_string(
                    [self.actions[0] if li in self.conv_ids else None for li in range(self.net_depth)]):
            self.verbose_logger.debug('Return: %d\n', self.output_size - 2)
            return self.output_size - 3
        elif self.get_action_string(action_list) == \
                self.get_action_string(
                    [self.actions[1] if li in self.conv_ids else None for li in range(self.net_depth)]):
            self.verbose_logger.debug('Return (index): %d\n', self.output_size - 1)
            return self.output_size - 2
        elif self.get_action_string(action_list) == \
                self.get_action_string(
                    [self.actions[2] if li in self.conv_ids else None for li in range(self.net_depth)]):
            self.verbose_logger.debug('Return (index): %d\n', self.output_size - 1)
            return self.output_size - 1

        else:
            conv_id = 0
            for li, la in enumerate(action_list):
                if la is None:
                    continue

                if la == self.actions[self.global_actions]:
                    secondary_idx = conv_id
                    primary_idx = 0

                conv_id += 1

            action_idx = primary_idx * self.n_conv + secondary_idx
            self.verbose_logger.debug('Primary %d Secondary %d', primary_idx, secondary_idx)
            self.verbose_logger.debug('Return (index): %d\n', action_idx)
            return action_idx

    def update_trial_phase(self, trial_phase):
        self.trial_phase = trial_phase

    def get_new_valid_action_when_greedy(self,action_idx,found_valid_action, data, q_for_actions):
        '''
        ================= Look ahead 1 step (Validate Predicted Action) =========================
        make sure predicted action stride is not larger than resulting output.
        make sure predicted kernel size is not larger than the resulting output
        To avoid such scenarios we create a restricted action space if this happens and chose from that
        :param action_idx:
        :param found_valid_action:
        :param data:
        :param q_for_actions:
        :return:
        '''
        allowed_actions = [tmp for tmp in range(self.output_size)]
        invalid_actions = []

        layer_actions_list = self.action_list_with_index(action_idx)
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
                    self.verbose_logger.debug('\tAction %s is not valid li(%d), (Next Filter Count: %d). ' % (
                        str(la), li, next_filter_count))
                    try:
                        del q_for_actions[action_idx]
                    except:
                        self.verbose_logger.critical('Error Length Q (%d) Action idx (%d)', len(q_for_actions),
                                                action_idx)
                        self.verbose_logger.critical('\tAction %s is not valid li(%d), (Next Filter Count: %d). ',
                                                str(la), li, next_filter_count)
                    allowed_actions.remove(action_idx)
                    invalid_actions.append(action_idx)
                    found_valid_action = False

                    # udpate current action to another action
                    max_idx = np.asscalar(np.argmax(q_for_actions))
                    action_idx = allowed_actions[max_idx]
                    layer_actions_list = self.action_list_with_index(action_idx)
                    self.verbose_logger.debug('\tSelected new action: %s', layer_actions_list)
                    break
                else:
                    found_valid_action = True

        if action_idx >= self.output_size - 2:
            found_valid_action = True

        found_valid_action = True
        return layer_actions_list, found_valid_action,invalid_actions

    def check_if_should_stop_adapting(self):

        if np.argmax(self.current_q_for_actions) == self.output_size - 1:
            if self.current_q_for_actions[-1] > self.max_q_ft:
                self.max_q_ft = self.current_q_for_actions[-1]
                self.ft_saturated_count = 0
                self.stop_adapting = False
            else:
                self.ft_saturated_count += 1

            if self.ft_saturated_count > self.threshold_stop_adapting:
                self.stop_adapting = True
        else:
            self.ft_saturated_count = 0

        return self.stop_adapting

    def get_new_valid_action_when_exploring(self, action_idx, found_valid_action, data,trial_action_probs):
        '''
            ================= Look ahead 1 step (Validate Predicted Action) =========================
            make sure predicted action stride is not larger than resulting output.
            make sure predicted kernel size is not larger than the resulting output
            To avoid such scenarios we create a restricted action space if this happens and chose from that
        :param found_valid_action: Returns True when a valid action is found
        :param data:
        :param trial_action_probs:
        :return:
        '''

        invalid_actions=[]
        layer_actions_list = self.action_list_with_index(action_idx)
        # while loop for checkin the validity of the action and choosing another if not
        while not found_valid_action and action_idx < self.output_size - 2:

            # if chosen action is do_nothing or finetune
            if action_idx >= self.output_size - 2:
                found_valid_action = True
                break

            # check for all layers if the chosen action is valid
            for li, la in enumerate(layer_actions_list):
                if la is None:  # pool layer
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
                    self.verbose_logger.debug('\tAction %s is not valid li(%d), (Next Filter Count: %d). ' % (
                        str(la), li, next_filter_count))

                    # invalid_actions.append(action_idx)
                    found_valid_action = False
                    # udpate current action to another action
                    action_idx = np.random.choice(self.output_size, p=trial_action_probs)

                    layer_actions_list = self.action_list_with_index(action_idx)
                    self.verbose_logger.debug('\tSelected new action: %s', layer_actions_list)
                    break
                else:
                    found_valid_action = True

            if action_idx >= self.output_size - 2:
                found_valid_action = True

        found_valid_action = True

        return layer_actions_list,found_valid_action,invalid_actions

    def get_new_valid_action_when_stochastic(self, action_idx, found_valid_action, data, q_for_actions):

        layer_actions_list = self.action_list_with_index(action_idx)
        invalid_actions = []

        # If the qlearner type is pruning we have to be careful not to take "remove from last layer" action
        if self.qlearner_type=='prune':
            allowed_actions = np.argsort(q_for_actions).flatten()[1:]  # Only get a random index from the highest q values
        else:
            allowed_actions = np.argsort(q_for_actions).flatten()

        allowed_actions = allowed_actions.tolist()

        while not found_valid_action and action_idx < self.output_size - 2:
            self.verbose_logger.debug('Checking action validity')
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
                    self.verbose_logger.debug('\tAction %s is not valid li(%d), (Next Filter Count: %d). ', str(la), li,
                                         next_filter_count)
                    allowed_actions.remove(action_idx)
                    invalid_actions.append(action_idx)
                    found_valid_action = False

                    action_idx = np.random.choice(allowed_actions)
                    layer_actions_list = self.action_list_with_index(action_idx)
                    self.verbose_logger.debug('\tSelected new action: %s', layer_actions_list)
                    break
                else:
                    found_valid_action = True

            if action_idx >= self.output_size - 2:
                found_valid_action = True

        found_valid_action = True

        return layer_actions_list, found_valid_action, invalid_actions

    def get_explore_type_action(self,data,history_t_plus_1, explore_action_probs):

        action_idx = np.random.choice(self.output_size, p=explore_action_probs)

        found_valid_action = False
        layer_actions_list, found_valid_action, invalid_actions = self.get_new_valid_action_when_exploring(
            action_idx, found_valid_action, data, trial_action_probs=explore_action_probs
        )

        assert found_valid_action

        if len(history_t_plus_1) == self.state_history_length:
            curr_x = np.asarray(self.phi(history_t_plus_1)).reshape(1, -1)
            q_for_actions = self.session.run(self.tf_out_target_op, feed_dict={self.tf_state_input: curr_x})
            q_for_actions = q_for_actions.flatten().tolist()
            self.current_q_for_actions = q_for_actions

            q_value_strings = ''
            for q_val in q_for_actions:
                q_value_strings += '%.5f' % q_val + ','
            self.q_logger.info("%d,%s", self.local_time_stamp, q_value_strings)
            self.verbose_logger.debug('\tPredicted Q: %s', q_for_actions[:10])

        if len(self.rand_state_list) < self.rand_state_length and \
                        np.random.random() < self.rand_state_accum_rate and \
                        len(history_t_plus_1) == self.state_history_length:
            self.rand_state_list.append(self.phi(history_t_plus_1))

        return layer_actions_list, invalid_actions

    def get_greedy_type_action(self,data,history_t_plus_1):

        curr_x = np.asarray(self.phi(history_t_plus_1)).reshape(1, -1)
        q_for_actions = self.session.run(self.tf_out_target_op, feed_dict={self.tf_state_input: curr_x})
        q_for_actions = q_for_actions.flatten().tolist()
        self.current_q_for_actions = q_for_actions

        q_value_strings = ''
        for q_val in q_for_actions:
            q_value_strings += '%.5f' % q_val + ','
        self.q_logger.info("%d,%s", self.local_time_stamp, q_value_strings)
        self.verbose_logger.debug('\tPredicted Q: %s', q_for_actions[:10])

        # Finding when to stop adapting
        # for this we choose the point the finetune operation has the
        # maximum utility compared to other actions and itself previously

        action_idx = np.asscalar(np.argmax(q_for_actions))

        if np.random.random() < 0.25:
            if np.random.random() < 0.5:
                action_idx = np.asscalar(np.argsort(q_for_actions).flatten()[-2])
            else:
                action_idx = np.asscalar(np.argsort(q_for_actions).flatten()[-3])

        found_valid_action = False
        layer_actions_list, found_valid_action, invalid_actions = self.get_new_valid_action_when_greedy(
            action_idx, found_valid_action, data, q_for_actions
        )

        self.verbose_logger.debug('\tChose: %s' % str(layer_actions_list))

        assert found_valid_action

        return layer_actions_list, invalid_actions

    def get_stochastic_type_action(self,data, history_t_plus_1):

        curr_x = np.asarray(self.phi(history_t_plus_1)).reshape(1, -1)
        q_for_actions = self.session.run(self.tf_out_target_op, feed_dict={self.tf_state_input: curr_x})
        self.current_q_for_actions = q_for_actions

        # not to restrict from the beginning

        rand_indices = np.argsort(q_for_actions).flatten()[1:]  # Only get a random index from the actions except last
        self.verbose_logger.info('Allowed action indices: %s', rand_indices)
        action_idx = np.random.choice(rand_indices)

        layer_actions_list = self.action_list_with_index(action_idx)
        self.verbose_logger.debug('\tChose: %s' % str(layer_actions_list))

        # ================= Look ahead 1 step (Validate Predicted Action) =========================
        # make sure predicted action stride is not larger than resulting output.
        # make sure predicted kernel size is not larger than the resulting output
        # To avoid such scenarios we create a
        #  action space if this happens

        # Check if the next filter count is invalid for any layer
        found_valid_action = False
        layer_actions_list, found_valid_action, invalid_actions = self.get_new_valid_action_when_stochastic(action_idx,found_valid_action,data,q_for_actions)

        assert found_valid_action

        return layer_actions_list, invalid_actions

    def output_action_with_type(self, data, action_type, **kwargs):
        '''
        Output action acording to one of the below methods
        Explore: action during the exploration (network growth and netowrk shrinkage)
        Deterministic: action with highest q value
        Stochastic: action in a stochastic manner
        :param data:
        :return:
        '''

        state = []
        state.extend(data['filter_counts_list'])
        state.extend(data['binned_data_dist'])

        self.verbose_logger.info('Data for (Depth Index,DistMSE,Filter Count) %s\n' % str(state))
        history_t_plus_1 = list(self.current_state_history)
        history_t_plus_1.append([state])

        self.verbose_logger.debug('Current state history: %s\n', self.current_state_history)
        self.verbose_logger.debug('history_t+1:%s\n', history_t_plus_1)
        self.verbose_logger.debug('Epsilons: %.3f\n', self.epsilon)
        self.verbose_logger.info('Trial phase: %.3f\n', self.trial_phase)

        if action_type == 'Explore':
            layer_actions_list,invalid_actions = self.get_explore_type_action(data,history_t_plus_1,kwargs['p_action'])

        # deterministic selection (if epsilon is not 1 or q is not empty)
        elif action_type == 'Greedy':
            self.verbose_logger.info('Choosing action deterministic...')
            # we create this copy_actions in case we need to change the order the actions processed
            # without changing the original action space (self.actions)
            layer_actions_list,invalid_actions = self.get_greedy_type_action(data,history_t_plus_1)

        # random selection
        elif action_type == 'Stochastic':
            self.verbose_logger.info('Choosing action stochastic...')
            layer_actions_list,invalid_actions = self.get_stochastic_type_action(data,history_t_plus_1)

        else:
            raise NotImplementedError

        self.verbose_logger.debug('=' * 60)
        self.verbose_logger.debug('State')
        self.verbose_logger.debug(state)
        self.verbose_logger.debug('Action (%s)',action_type)
        self.verbose_logger.debug(layer_actions_list)
        self.verbose_logger.debug('=' * 60)

        if self.prev_action is not None and \
                        self.get_action_string(layer_actions_list) == self.get_action_string(self.prev_action):
            self.same_action_count += 1

        else:
            self.same_action_count = 0

        self.action_logger.info('%s,%s,%s,%.3f', action_type, state, layer_actions_list, self.epsilon)

        self.prev_action = layer_actions_list
        self.prev_state = state

        self.verbose_logger.info('\tSelected action: %s\n', layer_actions_list)

        return state, layer_actions_list, invalid_actions

    def get_current_q_vector(self):
        return self.current_q_for_actions

    def output_action(self, data):
        '''
        Output action acording to one of the below methods
        Explore: action during the exploration (network growth and netowrk shrinkage)
        Deterministic: action with highest q value
        Stochastic: action in a stochastic manner
        :param data:
        :return:
        '''

        invalid_actions = []
        # data => ['distMSE']['filter_counts']
        # ['filter_counts'] => depth_index : filter_count
        # State => Layer_Depth (w.r.t net), dist_MSE, number of filters in layer

        action_type = None  # for logging purpose
        state = []  # removed distMSE (believe doesn't have much value)
        state.extend(data['filter_counts_list'])

        self.verbose_logger.info('Data for (Depth Index,DistMSE,Filter Count) %s\n' % str(state))
        history_t_plus_1 = list(self.current_state_history)
        history_t_plus_1.append([state])

        self.verbose_logger.debug('Current state history: %s\n', self.current_state_history)
        self.verbose_logger.debug('history_t+1:%s\n', history_t_plus_1)
        self.verbose_logger.debug('Epsilons: %.3f\n', self.epsilon)
        self.verbose_logger.info('Trial phase: %.3f\n', self.trial_phase)

        n_conv_first_half = int(ceil(self.n_conv * 1.0 / 2.0))
        n_conv_last_half = int(self.n_conv - n_conv_first_half)
        assert n_conv_first_half + n_conv_last_half == self.n_conv
        if self.trial_phase < self.trial_phase_threshold:
            action_type = 'Explore'
            layer_actions_list,invalid_actions = self.get_explore_type_action(data,history_t_plus_1)

        # deterministic selection (if epsilon is not 1 or q is not empty)
        elif np.random.random() > self.epsilon and len(history_t_plus_1) == self.state_history_length:
            self.verbose_logger.info('Choosing action deterministic...')
            # we create this copy_actions in case we need to change the order the actions processed
            # without changing the original action space (self.actions)
            action_type = 'Greedy'
            layer_actions_list,invalid_actions = self.get_greedy_type_action(data,history_t_plus_1)

        # random selection
        else:
            self.verbose_logger.info('Choosing action stochastic...')
            action_type = 'Stochastic'

            layer_actions_list,invalid_actions = self.get_stochastic_type_action(data,history_t_plus_1)

        # decay epsilon
        if self.trial_phase >= self.trial_phase_threshold:
            self.epsilon = max(self.epsilon * 0.95, self.min_epsilon)

        self.verbose_logger.debug('=' * 60)
        self.verbose_logger.debug('State')
        self.verbose_logger.debug(state)
        self.verbose_logger.debug('Action (%s)',action_type)
        self.verbose_logger.debug(layer_actions_list)
        self.verbose_logger.debug('=' * 60)

        if self.prev_action is not None and \
                        self.get_action_string(layer_actions_list) == self.get_action_string(self.prev_action):
            self.same_action_count += 1

        else:
            self.same_action_count = 0

        self.action_logger.info('%s,%s,%s,%.3f', action_type, state, layer_actions_list, self.epsilon)

        self.prev_action = layer_actions_list
        self.prev_state = state

        self.verbose_logger.info('\tSelected action: %s\n', layer_actions_list)

        return state, layer_actions_list, invalid_actions

    def get_trial_phase_action_probs(self):
        '''
        We output action executing probabilities during the exploration stage
        :return:
        '''
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

        return trial_action_probs

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


    def get_complexity_penalty(self, curr_comp, prev_comp, filter_bound_vec,act_string):

        if self.num_classes>101:
            return 0.0

        # total gain should be negative for taking add action before half way througl a layer
        # total gain should be positve for taking add action after half way througl a layer
        total = 0
        split_factor = 0.6
        for l_i,(c_depth, p_depth, up_dept) in enumerate(zip(curr_comp,prev_comp,filter_bound_vec)):
            if up_dept>0 and abs(c_depth-p_depth) > 0:
                total += (((up_dept*split_factor)-c_depth)/(up_dept*split_factor))

        if 'add' in act_string:
            return - total * (1/self.num_classes)
        elif 'remove' in act_string:
            return total * (1/self.num_classes)
        else:
            return 0.0

    def get_grow_encouragement(self,affected_layer_idx,action_string, curr_comp, filter_bound_vec,top_k,split_fraction):

        growth_enc = (1+np.log(affected_layer_idx+1))*\
                     ((filter_bound_vec[affected_layer_idx]*split_fraction) - curr_comp[affected_layer_idx])*(top_k/self.num_classes)\
                     /(filter_bound_vec[affected_layer_idx])
        if 'add' in action_string:
            return growth_enc
        if 'remove' in action_string:
            return - growth_enc


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
                self.verbose_logger.info('Training the Q Approximator with Experience...')
                self.verbose_logger.debug('(Q) Total experience data: %d', len(self.experience))

                # =====================================================
                # Returns a batch of experience
                # ====================================================
                if len(self.experience) > self.batch_size:
                    exp_indices = np.random.randint(0, len(self.experience), (self.batch_size,))
                    self.verbose_logger.debug('Experience indices: %s', exp_indices)
                    x, y, r, next_state = self.get_xy_with_experince([self.experience[ei] for ei in exp_indices])
                else:
                    x, y, r, next_state = self.get_xy_with_experince(self.experience)

                if self.global_time_stamp < 5:
                    assert np.max(x) <= 1.0 and np.max(x) >= -1.0 and np.max(y) <= 1.0 and np.max(y) >= -1.0

                self.verbose_logger.debug('Summary of Structured Experience data')
                self.verbose_logger.debug('\tX:%s', x.shape)
                self.verbose_logger.debug('\tY:%s', y.shape)
                self.verbose_logger.debug('\tR:%s', r.shape)
                self.verbose_logger.debug('\tNextState:%s', next_state.shape)

                pred_q = self.session.run(self.tf_out_target_op, feed_dict={self.tf_state_input: x})
                self.verbose_logger.debug('\tPredicted %s:', pred_q.shape)
                target_q = r.flatten() + self.discount_rate * np.max(pred_q, axis=1).flatten()

                self.verbose_logger.debug('\tTarget Q %s:', target_q.shape)
                self.verbose_logger.debug('\tTarget Q Values %s:', target_q[:5])
                assert target_q.size <= self.batch_size

                # This gives y values by multiplying one-hot-encded actions (bxaction_size) in the experience tuples
                # with target q values (bx1)
                ohe_targets = np.multiply(y, target_q.reshape(-1, 1))

                # since the state contain layer id, let us make the layer id one-hot encoded
                self.verbose_logger.debug('X (shape): %s, Y (shape): %s', x.shape, y.shape)
                self.verbose_logger.debug('X: \n%s, Y: \n%s', str(x[:3, :]), str(y[:3]))

                _ = self.session.run([self.tf_optimize_op], feed_dict={
                    self.tf_state_input: x, self.tf_q_targets: ohe_targets, self.tf_q_mask: y
                })

                if self.global_time_stamp % self.target_update_rate == 0 and self.local_time_stamp % self.n_conv == 0:
                    self.verbose_logger.info('Coppying the Q approximator as the Target Network')
                    # self.target_network = self.regressor.partial_fit(x, y)
                    _ = self.session.run([self.tf_target_update_ops])

                if self.global_time_stamp > 0 and self.global_time_stamp % self.exp_clean_interval == 0:
                    self.clean_experience()

        si, ai_list, sj = data['prev_state'], data['prev_action'], data['curr_state']
        self.verbose_logger.debug('Si,Ai,Sj: %s,%s,%s', si, ai_list, sj)

        curr_action_string = self.get_action_string(ai_list)
        comp_gain = self.get_complexity_penalty(data['curr_state'], data['prev_state'], self.filter_bound_vec,
                                                curr_action_string)
        # Turned off 28/09/2017
        #mean_accuracy = (1.0 + ((data['pool_accuracy'] + data['prev_pool_accuracy'])/200.0)) *\
        #                ((data['pool_accuracy'] - data['prev_pool_accuracy']) / 100.0)
        mean_accuracy = (1.0/self.num_classes) if data['pool_accuracy'] > data['max_pool_accuracy'] else -(1.0/self.num_classes)
        #immediate_mean_accuracy = (1.0 + ((data['unseen_valid_accuracy'] + data['prev_unseen_valid_accuracy'])/200.0))*\
        #                          (data['unseen_valid_accuracy'] - data['prev_unseen_valid_accuracy']) / 100.0

        self.verbose_logger.info('Complexity penalty: %.5f', comp_gain)

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

        reward = mean_accuracy - comp_gain #+ 0.5*immediate_mean_accuracy # new
        curr_action_string = self.get_action_string(ai_list)

        # exponential magnifier to prevent from being taken consecutively
        # Turned off on 21/09/2017
        '''if self.trial_phase > 1.0:
            if ('add' in curr_action_string or 'remove' in curr_action_string) and self.same_action_count >= 1:
                self.verbose_logger.info('Reward before magnification: %.5f', reward)
                if reward > 0:
                    reward = reward  # /= min(self.same_action_count + 1,10)
                else:
                    reward *= min(self.same_action_count + 1, 10)
                self.verbose_logger.info('Reward after magnification: %.5f', reward)
            # encourage taking finetune action consecutively
            if 'finetune' in curr_action_string and self.same_action_count >= 1:
                self.verbose_logger.info('Reward before magnification: %.5f', reward)
                if reward > 0:
                    reward *= min(self.same_action_count + 1, 10)
                else:  # new
                    self.same_action_count = 0  # reset action count # new
                self.verbose_logger.info('Reward after magnification: %.5f', reward)'''

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
        self.verbose_logger.debug('History(t)')
        self.verbose_logger.debug('%s\n', history_t)

        assert len(history_t) <= self.state_history_length

        action_idx = self.index_from_action_list(ai_list)

        # update current state history
        self.current_state_history.append([si])
        self.current_state_history[-1].append([1 if action_idx == act else 0 for act in range(self.output_size)])

        if len(self.current_state_history) > self.state_history_length - 1:
            del self.current_state_history[0]
            assert len(self.current_state_history) == self.state_history_length - 1

        self.verbose_logger.debug('Current History')
        self.verbose_logger.debug('%s\n', self.current_state_history)

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
                #    self.verbose_logger.info('Updating reward for %s', prev_action_string)
                #    self.verbose_logger.info('\tFrom: %.3f',self.experience[-2][2])
                #    self.experience[-2][2] += 0.5*max(0,(data['pool_accuracy'] - self.prev_prev_pool_accuracy)/100.0)
                #    self.verbose_logger.info('\tTo: %.3f\n', self.experience[-2][2])

            for invalid_a in data['invalid_actions']:
                self.verbose_logger.debug('Adding the invalid action %s to experience', invalid_a)
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
                self.verbose_logger.debug('Latest Experience: ')
                self.verbose_logger.debug('\t%s\n', self.experience[-1])

        self.verbose_logger.info('Update Summary ')
        self.verbose_logger.info('\tState: %s', si)
        self.verbose_logger.info('\tAction: %d,%s', action_idx, ai_list)
        self.verbose_logger.info('\tReward: %.3f', reward)
        self.verbose_logger.info('\t\tReward (Mean Acc): %.4f', mean_accuracy)

        self.previous_reward = reward
        self.prev_prev_pool_accuracy = data['prev_pool_accuracy']

        self.local_time_stamp += 1
        self.global_time_stamp += 1

        self.verbose_logger.info('Global/Local time step: %d/%d\n', self.global_time_stamp, self.local_time_stamp)

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
            q_pred = self.session.run(self.tf_out_target_op, feed_dict={self.tf_state_input: x})
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

    def get_add_action_type(self):
        return 'Add'

    def get_remove_action_type(self):
        return 'Remove'

    def get_finetune_action_type(self):
        return 'Finetune'

    def get_donothing_action_type(self):
        return "DoNothing"

    def get_naivetrain_action_type(self):
        return "DoNothing"

    def get_action_type_with_action_list(self,action_list):
        for li,la in enumerate(action_list):
            if la is None:
                continue

            if la[0]=='add':
                return self.get_add_action_type()
            elif la[0]=='remove':
                return self.get_remove_action_type()
            elif la[0]=='finetune':
                return self.get_finetune_action_type()
            elif la[0]=='naive_train':
                return self.get_naivetrain_action_type()

        return self.get_donothing_action_type()