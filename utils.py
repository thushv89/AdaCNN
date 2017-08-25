import xml.etree.ElementTree as ET
import tensorflow as tf


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