"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""
import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

# checking to see if the TensorFlow version is compatible with tf.concat_v2 (which replaced tf.concat)
if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        # will use tf.concat_v2
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    # fall back to previous version
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)

# a batch normalization function for stablizing training 
def bn(x, is_training, scope):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9, # decay rate for MA (moving averages)
                                        updates_collections=None, #
                                        epsilon=1e-5, # small value, close to zero but not zero to avoid dividing by zero
                                        scale=True, # output is scaled
                                        is_training=is_training, # separates training and inference behavior
                                        scope=scope)

# compute the output size for convolution with stride, matching the feature map to a scale that works for other functions 
def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

# concatenate a "y vector" on the feature map x
def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape() # gets shape of input tensor x
    y_shapes = y.get_shape() # gets shape of the conditioning tensor y
    # concatenate along the axis of the feature map x
    return concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

# define a 2-dimensional convolutional layer 
def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        # creates a weight w with an initializer 
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        # using strides and the 2D layer, perfroms convolution
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        # adds a bias variable to convolution-- helps the model learn some flexibility, and allows us to shidt our activation function 
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        # reshapes the convolution with the new bias
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv
# defines a deconvolution, or the transpose of a normal 2D convolution. this is used for upsampling
def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, name="deconv2d", stddev=0.02, with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        # create another weight variable for the filter
        # w: Filter tensor with dimensions [filter_height, filter_width, output_channels, input_channels]
        # k_h and k_w: Height and width of the filter 
        # output_shape[-1]: The number of output channels (depth of the output feature map)
        # input_.get_shape()[-1]: The number of input channels (depth of the input feature map)
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            # perfrom deconvolution to upsample 
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            # fallback to an older TensorFlow function
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        # creates a bias with initializer set to zero, which gets added to the output of the deconvolution
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        # if the parameter is true, returns the ouput of deconvolution with weights and biases
        if with_w:
            return deconv, w, biases
        else: 
            # if with_w is not True, just returns the deconvolution. 
            return deconv
# defines a leaky ReLU activation function for some non-linearity
# "leak" allows for small gradient for negative input values 
# returns original output variable, negative values will not be zeroed like normal ReLU
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

# defines a fully linear layer
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list() # gets hape of the input tensor

    # defines a variable scope for the linear layer
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
        # creates a bias 
        
        bias = tf.get_variable("bias", [output_size],
        initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
            # will return layer with/without bias and weights applied, depending on if used.
