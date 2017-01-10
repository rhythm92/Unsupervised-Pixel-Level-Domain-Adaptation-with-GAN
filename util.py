"""Contains basic utilities such as layers or network building blocks.
   Some codes borrowed from https://github.com/yunjey/domain-transfer-network-tensorflow.git
"""
import tensorflow as tf

def conv(x, filters=64, ksize=(3,3), s=1, p='SAME', name=None):
    channel_in = x.get_shape()[-1]
    k_w, k_h = ksize

    with tf.variable_scope(name):
        w = tf.get_variable('w', shape=[k_w, k_h, channel_in, filters],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', shape=[channel_out],
                            initializer=tf.constant_initializer(0.0))
        return tf.nn.conv2d(x, w, strides=[1, s, s, 1], padding=p) + b

def maxpool(x,pool_size=2):
    return tf.nn.max_pool(x, ksize=[1,pool_size,pool_size,1],
                          strides=[1,pool_size,pool_size,1], padding='SAME')

def dense(x, dim_out = 10):
    dim_in = x.get_shape()[-1]

    with tf.variable_scope(name):
        w = tf.get_variable('w', shape=[dim_in, dim_out],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', shape=[dim_out],
                            initializer=tf.constant_initializer(0.0))
        return tf.matmul(x, w) + b

def batch_norm(x, name=None):
    """TODO: how to do the batch norm?
    """
    return tf.contrib.layers.batch_norm(x)

def relu(x):
    return tf.nn.relu(x)

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)

def res_block(x):
    """Residule block as described in the paper:
       n64s1 -> BN -> relu -> n64s1 -> BN -> + -> output
         |                                   |
         - - - - - - - - - - - - - - - - - - -

    Args:
        x: (batch_size, w, h, c) the input tensor
        name: (string)  the name space of the block

    Outs:
        (batch_size, w, h, c) tensor, same size as the output
    """
    pass

