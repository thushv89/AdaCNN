import xml.etree.ElementTree as ET
import tensorflow as tf
from math import ceil
import constants

TF_CONV_WEIGHT_SHAPE_STR = constants.TF_CONV_WEIGHT_SHAPE_STR
TF_FC_WEIGHT_IN_STR = constants.TF_FC_WEIGHT_IN_STR
def retrive_dictionary_from_xml(fname):
    '''
    Retrieves a dictionary from the xml file
    :param fname: Name of the xml file
    :return:
    '''
    dictionary = {}
    tree = ET.parse(fname)
    root = tree.getroot()
    for item in root.iter('entry'):
        values = []
        for sub_item in item.iter():
            dtype = sub_item.attrib('datatype')
            if sub_item.tag == 'key':

                 key = get_item_with_dtype(sub_item.text,dtype)
            else:
                 values.append(get_item_with_dtype(sub_item.text,dtype))

        # if the list only has one element
        # dictionary has a single value as value
        if len(values)==1:
            dictionary[key] = values[0]
        else:
            dictionary[key] = values

        break


# For the xml file
datatypes = ['string','int32','float32']


def get_item_with_dtype(value,dtype):
    '''
    Get a given string (value) with the given dtype
    :param value:
    :param dtype:
    :return:
    '''
    if dtype==datatypes[0]:
        return str(value)
    elif dtype==datatypes[1]:
        return int(value)
    elif dtype==datatypes[2]:
        return float(value)
    else:
        raise NotImplementedError


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def update_tf_hyperparameters(op, tf_weight_shape, tf_in_size):
    global cnn_ops, cnn_hyperparameters
    update_ops = []
    if 'conv' in op:
        with tf.variable_scope(op, reuse=True):
            update_ops.append(tf.assign(tf.get_variable(TF_CONV_WEIGHT_SHAPE_STR, dtype=tf.int32), tf_weight_shape))
    if 'fulcon' in op:
        with tf.variable_scope(op, reuse=True):
            update_ops.append(tf.assign(tf.get_variable(TF_FC_WEIGHT_IN_STR, dtype=tf.int32), tf_in_size))

    return update_ops

def get_ops_hyps_from_string(dataset_info,net_string):
    # E.g. String
    # Init,0,0,0#C,1,1,64#C,5,1,64#C,5,1,128#P,5,2,0#C,1,1,64#P,2,2,0#Terminate,0,0,0

    num_channels = dataset_info['n_channels']
    image_size = dataset_info['image_size']
    num_labels = dataset_info['n_labels']

    cnn_ops = []
    cnn_hyperparameters = {}
    prev_conv_hyp = None

    op_tokens = net_string.split('#')
    depth_index = 0
    fulcon_depth_index = 0  # makes implementations easy

    last_feature_map_depth = 3  # need this to calculate the fulcon layer in size
    last_fc_out = 0

    final_2d_width = image_size
    for token in op_tokens:
        # state (layer_depth,op=(type,kernel,stride,depth),out_size)
        token_tokens = token.split(',')
        # op => type,kernel,stride,depth
        op = (token_tokens[0], int(token_tokens[1]), int(token_tokens[2]), int(token_tokens[3]))
        if op[0] == 'C':
            op_id = 'conv_' + str(depth_index)
            if prev_conv_hyp is None:
                hyps = {'weights': [op[1], op[1], num_channels, op[3]], 'stride': [1, op[2], op[2], 1],
                        'padding': 'SAME'}
            else:
                hyps = {'weights': [op[1], op[1], prev_conv_hyp['weights'][3], op[3]], 'stride': [1, op[2], op[2], 1],
                        'padding': 'SAME'}

            cnn_ops.append(op_id)
            cnn_hyperparameters[op_id] = hyps
            prev_conv_hyp = hyps  # need this to set the input depth for a conv layer
            last_feature_map_depth = op[3]
            depth_index += 1
            final_2d_width = ceil(final_2d_width//hyps['stride'][2])

        elif op[0] == 'P':
            op_id = 'pool_' + str(depth_index)
            hyps = {'type': 'max', 'kernel': [1, op[1], op[1], 1], 'stride': [1, op[2], op[2], 1], 'padding': 'SAME'}
            cnn_ops.append(op_id)
            cnn_hyperparameters[op_id] = hyps
            depth_index += 1
            final_2d_width = ceil(final_2d_width // hyps['stride'][2])

        elif op[0] == 'PG':
            cnn_ops.append('pool_global')
            pg_hyps = {'type': 'max', 'kernel': [1, op[1], op[1], 1], 'stride': [1, op[2], op[2], 1], 'padding': 'SAME'}
            cnn_hyperparameters['pool_global'] = pg_hyps
            final_2d_width = ceil(final_2d_width // pg_hyps['stride'][2])

        elif op[0] == 'FC':

            op_id = 'fulcon_' + str(fulcon_depth_index)
            # for the first fulcon layer size comes from the last convolutional layer
            if fulcon_depth_index==0:
                hyps = {'in': final_2d_width * final_2d_width * last_feature_map_depth, 'out': op[1]}
            # all the other fulcon layers the size comes from the previous fulcon layer
            else:
                hyps = {'in': cnn_hyperparameters['fulcon_'+str(fulcon_depth_index-1)]['out'], 'out': op[1]}
            cnn_ops.append(op_id)
            cnn_hyperparameters[op_id] = hyps
            fulcon_depth_index += 1
            last_fc_out = op[1]

        elif op[0] == 'Terminate':

            # if no FCs are present
            if fulcon_depth_index == 0:

                op_id = 'fulcon_out'
                if fulcon_depth_index==0:
                    hyps = {'in': final_2d_width * final_2d_width * last_feature_map_depth, 'out': num_labels}
                else:
                    hyps = {'in':cnn_hyperparameters['fulcon_'+str(depth_index-1)]['out'],'out':num_labels}

                cnn_ops.append(op_id)
                cnn_hyperparameters[op_id] = hyps

            else:
                op_id = 'fulcon_out'
                hyps = {'in': last_fc_out, 'out': num_labels}
                cnn_ops.append(op_id)
                cnn_hyperparameters[op_id] = hyps

        elif op[0] == 'Init':
            continue
        else:
            print('=' * 40)
            print(op[0])
            print('=' * 40)
            raise NotImplementedError

    return cnn_ops, cnn_hyperparameters,final_2d_width


def get_cnn_string_from_ops(cnn_ops, cnn_hyps):
    current_cnn_string = ''
    for op in cnn_ops:
        if 'conv' in op:
            current_cnn_string += '#C,' + str(cnn_hyps[op]['weights'][0]) + ',' + str(
                cnn_hyps[op]['stride'][1]) + ',' + str(cnn_hyps[op]['weights'][3])
        elif 'pool' in op:
            current_cnn_string += '#P,' + str(cnn_hyps[op]['kernel'][0]) + ',' + str(
                cnn_hyps[op]['stride'][1]) + ',' + str(0)
        elif 'fulcon_out' in op:
            current_cnn_string += '#Terminate,0,0,0'
        elif 'fulcon' in op:
            current_cnn_string += '#FC,' + str(cnn_hyps[op]['in'])

    return current_cnn_string