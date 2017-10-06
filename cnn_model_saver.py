def save_cnn_hyperparameters(main_dir,weight_sizes_dict,stride_dict, scope_list, hypeparam_filepath):

    if config.WEIGHT_SAVE_DIR and not os.path.exists(main_dir + os.sep + config.WEIGHT_SAVE_DIR):
        os.mkdir(main_dir + os.sep + config.WEIGHT_SAVE_DIR)

    hyperparam_dict = {'layers': config.TF_ANG_SCOPES,
                       'activations': config.ACTIVATION}

    for scope in scope_list:
        if 'fc' not in scope and 'out' not in scope:
            hyperparam_dict[scope] = {'weights_size': weight_sizes_dict[scope],
                                  'stride': stride_dict[scope]}
        else:
            hyperparam_dict[scope] = {'weights_size': weight_sizes_dict[scope]}

    pickle.dump(hyperparam_dict, open(main_dir + os.sep + config.WEIGHT_SAVE_DIR + os.sep + hypeparam_filepath, "wb"))


def save_cnn_weights_naive(main_dir, sess,model_filepath):
    var_dict = {}

    if config.WEIGHT_SAVE_DIR and not os.path.exists(main_dir + os.sep + config.WEIGHT_SAVE_DIR):
        os.mkdir(main_dir + os.sep + config.WEIGHT_SAVE_DIR)

    for scope in config.TF_ANG_SCOPES:

        if 'pool' not in scope:
            if 'out' not in scope:
                weights_name = scope + config.TF_SCOPE_DIVIDER + config.TF_WEIGHTS_STR
                bias_name = scope + config.TF_SCOPE_DIVIDER + config.TF_BIAS_STR
                with tf.variable_scope(scope,reuse=True):
                    var_dict[weights_name] = tf.get_variable(config.TF_WEIGHTS_STR)
                    var_dict[bias_name] = tf.get_variable(config.TF_BIAS_STR)

                    with open(main_dir + os.sep + config.WEIGHT_SAVE_DIR +os.sep + 'variable_names.txt','w') as f:
                        f.write(weights_name)
                        f.write(bias_name)

            else:
                with tf.variable_scope(scope, reuse=True):
                    for di in config.TF_DIRECTION_LABELS:
                        with tf.variable_scope(di, reuse=True):
                            weights_name = scope + config.TF_SCOPE_DIVIDER + di +\
                                           config.TF_SCOPE_DIVIDER + config.TF_WEIGHTS_STR
                            bias_name = scope + config.TF_SCOPE_DIVIDER + di +\
                                        config.TF_SCOPE_DIVIDER + config.TF_BIAS_STR

                            var_dict[weights_name] = tf.get_variable(config.TF_WEIGHTS_STR)
                            var_dict[bias_name] = tf.get_variable(config.TF_BIAS_STR)

                            with open(main_dir + os.sep + config.WEIGHT_SAVE_DIR + os.sep + 'variable_names.txt','w') as f:
                                f.write(weights_name)
                                f.write(bias_name)

    saver = tf.train.Saver(var_dict)
    saver.save(sess,main_dir + os.sep + config.WEIGHT_SAVE_DIR + os.sep +  model_filepath)

