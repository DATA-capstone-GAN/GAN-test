# This is the Gated Recurrent Unit (GRU) used by WGAN_GRUI.py to capture the temporal information necessary for the GAN to generate good values for the missing data.
# It can process sequences with missing values, learn patterns and use them to make better predictions.
# The main purpose is to capture temporal dependencies so they can then be passed to the WGAN_GRUI.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 23:13:29 2018

@author: yonghong, luo
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn import RNNCell
import tensorflow as tf


if "1.5" in tf.__version__ or "1.7" in tf.__version__:                                    # Check tensorflow version 1.5 or 1.7.
    from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell                          # Used for creating custom Reccurrent Neural Network (RNN) cells.
    from tensorflow.python.layers import base as base_layer                               # Used to build foundational functionality for neural network layers.
    from tensorflow.python.ops import nn_ops                                              # Contains functions for use with neural networks, i.e. activation, normalization, and dropout.
    _BIAS_VARIABLE_NAME = "bias"                                                          # Create constant for use with bias.
    _WEIGHTS_VARIABLE_NAME = "kernel"                                                     # Create constant for use with weights.
    
    class MyGRUCell15(LayerRNNCell):
    # todo: Just change it directly; I've tested it, and it doesn't affect the calling of dynamic_rnn. (original comment)
      """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    
      Args:
        num_units: int, The number of units in the GRU cell.
        activation: Nonlinearity to use.  Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
         in an existing scope.  If not `True`, and the existing scope already has
         the given variables, an error is raised.
        kernel_initializer: (optional) The initializer to use for the weight and
        projection matrices.
        bias_initializer: (optional) The initializer to use for the bias.
        name: String, the name of the layer. Layers with the same name will
          share weights, but to avoid mistakes we require reuse=True in such
          cases.
      """
      def __init__(self,
                   num_units,                        # The number of hidden units in the RNN.
                   activation=None,                  # Activation function for RNN (default = tanh)
                   reuse=None,                       # Defines whether to reuse variables or not.
                   kernel_initializer=None,          # Parameter for the wieght matrices.
                   bias_initializer=None,            # Parameter the bias term.
                   name=None):                       # Layername (layers with the same name can share variables)

        # Call GRUCell for TF version 1.5               
        super(MyGRUCell15, self).__init__(_reuse=reuse, name=name)
    
        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)                  # Defines input cells as 2 dimensional (Normal for RNN).    
        self._num_units = num_units                                     # Establishes number of neruons in the RNN.
        self._activation = activation or math_ops.tanh                  # Establishes the activation function (tanh by default)
        self._kernel_initializer = kernel_initializer                   # Initalization of the weight matrix.
        self._bias_initializer = bias_initializer                       # Initialization of the bias term.
    
      @property
      def state_size(self):                   # Defines the size of the internal state of the GRU cell. 
        return self._num_units
    
      @property
      def output_size(self):                  # Defines the size of the output produced by the GRU cell.
        return self._num_units
    
      def build(self, inputs_shape):
        if inputs_shape[1].value is None:                                                # Checks if the input shape is correct and raises error if it is not.
          raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                           % inputs_shape)
    
        input_depth = inputs_shape[1].value-self._num_units                              # Calculates the number of input features (dimensions) that will be fed to the GRU.
        self._gate_kernel = self.add_variable(                                           # Creates the kernel (weights) for the gates in the GRU cell.
            "gates/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, 2 * self._num_units],
            initializer=self._kernel_initializer)
        self._gate_bias = self.add_variable(                                             # Creates the bias for the gate calculations.
            "gates/%s" % _BIAS_VARIABLE_NAME,
            shape=[2 * self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)))
        self._candidate_kernel = self.add_variable(                                      # Creates the kernel (weights) for the candidate hidden state.
            "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, self._num_units],
            initializer=self._kernel_initializer)
        self._candidate_bias = self.add_variable(                                        # Creates the bias for the candidate hidden state calculations.
            "candidate/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
    
        self.built = True                                                         # Indicates that the build process is complete.
    
      def call(self, inputs, state):                                              # Implements the forward pass of the GRU cell. Processing input data and previous hidden cell.
        """Gated recurrent unit (GRU) with nunits cells."""
        totalLength=inputs.get_shape().as_list()[1]                               # Retrieves the number of features of the input tensor.
        inputs_=inputs[:,0:totalLength-self._num_units]                           # Separates the actual input features from the reset gates.
        rth=inputs[:,totalLength-self._num_units:]                                # Extracts the reset gate values from the input, used to update previous state.
        inputs=inputs_                                                            # Updates the inputs variable to reflect only the relevant input features.
        state=math_ops.multiply(rth,state)                                        # Determine how much of the previous state to keep. A value of 0 will reset the state, a value of 1 will keep it unchanged.
        
        gate_inputs = math_ops.matmul(                                            # Calculates the input to the gates of the GRU cell.
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)               # Adds the bias to the gate inputs.
    
        value = math_ops.sigmoid(gate_inputs)                                     # Applies the sigmoid activation function to the gate inputs.
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)         # Splits the value tensor into two parts: the reset gate r and the update gate u.
    
        r_state = r * state                                                       # Scales previous state by the reset gate r, allows for selective forgetting of the previous information.
    
        candidate = math_ops.matmul(                                              # Calculates the candidate hidden state.
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)              # Adds the bias for the candidate state.
    
        c = self._activation(candidate)                                           # Applies the activation function to the candidate hidden state.
        new_h = u * state + (1 - u) * c                                           # Computes the new hidden state of the GRU cell.
        return new_h, new_h                                                       # Returns the updated hidden state for the next time step.

# TF version check and imports appropriate Linear Transformation function.
elif "1.4" in tf.__version__:
    from tensorflow.python.ops.rnn_cell_impl import _Linear
elif "1.2" in tf.__version__:
    from tensorflow.python.ops.rnn_cell_impl import _linear


# Call GRUCell for TF version 1.4. Uses RNNCell instead of LayerRNNCell.
class MyGRUCell4(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
  """

  def __init__(self,                                                          
               num_units,                                                      # The number of hidden units in the RNN.
               activation=None,                                                # Activation function for RNN (default = tanh)
               reuse=None,                                                     # Defines whether to reuse variables or not.
               kernel_initializer=None,                                        # Parameter for the wieght matrices.
               bias_initializer=None):                                         # Parameter for the bias term.
    super(MyGRUCell4, self).__init__(_reuse=reuse)
    self._num_units = num_units                                                # Establishes number of neruons in the RNN.
    self._activation = activation or math_ops.tanh                             # Establishes the activation function (tanh by default)
    self._kernel_initializer = kernel_initializer                              # Initalization of the weight matrix.
    self._bias_initializer = bias_initializer                                  # Initalization of the bias term.
    self._gate_linear = None                                                   # Hold the linear transformation (weight and bias) used for calculating the gate values. Control the flow of information.
    self._candidate_linear = None                                              # Responsible for the linear transformation for computing the candidate activation values, part of the GRU's mechanism for updating the hidden state.

  @property
  def state_size(self):                                                        # Defines the size of the internal state of the GRU cell.
    return self._num_units

  @property
  def output_size(self):                                                       # Defines the size of the output produced by the GRU cell.
    return self._num_units

  def call(self, inputs, state):                                               # inputs - input data, state - previous hidden state
    """Gated recurrent unit (GRU) with nunits cells."""
    # inputs = realinputs + m +rt
    # rt's length is self._num_units
    # state = rt * older state 
    # input = first 2 part
    totalLength=inputs.get_shape().as_list()[1]                                # Determines the total number of features in the input tensor.
    inputs_=inputs[:,0:totalLength-self._num_units]                            # Separates the actual input features from the reset gate values.
    rth=inputs[:,totalLength-self._num_units:]                                 # Contains the values for the reset gate, used in state updating.
    inputs=inputs_                                                             # Updates the inputs variable to reflect only the relevant input features.
    state=math_ops.multiply(rth,state)                                         # Updates the previous state by multiplying it with the reset gate. Allows the model to forget parts of the previous state.
    if self._gate_linear is None:                                              # Check if linear transformation has been completed on the gate.
      bias_ones = self._bias_initializer                                       # Sets up the bias term in the GRU Cell.
      if self._bias_initializer is None:                                       # Check if bias initializer is initialized, defaults to a constant initializer with a value of 1 if not.
        bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
      with vs.variable_scope("gates"):  # Reset gate and update gate.
        self._gate_linear = _Linear(
            [inputs, state],
            2 * self._num_units,
            True,
            bias_initializer=bias_ones,
            kernel_initializer=self._kernel_initializer)

    value = math_ops.sigmoid(self._gate_linear([inputs, state]))               # Determines how much of the previous state to keep and how much of the new input to incorporate.
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)          # Affects how much of the past information to discard.  Controls how much of the new candidate hidden state to incorporate.

    r_state = r * state                                                        # Enables the GRU to selectively forget parts of the previous hidden state. 
    if self._candidate_linear is None:
      with vs.variable_scope("candidate"):
        self._candidate_linear = _Linear(
            [inputs, r_state],                                                 # inputs - actual input features for the current time step. r_state - modified previous state of the GRU, scaled by the reset gate. Allows the GRU to forget parts of the previous state based on the reset gate value.
            self._num_units,                                                   # Specifies the number of output units (or neurons) for this linear transformation.
            True,                                                              # Indicates whether to include a bias term in this linear transformation.
            bias_initializer=self._bias_initializer,                           # Biases added to the linear transformations to allow the model to fit the training data more flexibly.
            kernel_initializer=self._kernel_initializer)                       # Control how much influence the inputs have on the output of each neuron in the layer.
    c = self._activation(self._candidate_linear([inputs, r_state]))            # Applies the activation function to the candidate hidden state.
    new_h = u * state + (1 - u) * c                                            # Computes the new hidden state of the GRU cell.
    return new_h, new_h                                                        # Returns the updated hidden state for the next time step.

# Call GRUCell for TF version 1.2
class MyGRUCell2(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
  """

  def __init__(self,
               num_units,                                                      # The number of hidden units in the RNN.
               activation=None,                                                # Activation function for RNN (default = tanh).
               reuse=None,                                                     # Defines whether to reuse variables or not.
               kernel_initializer=None,                                        # Parameter for the wieght matrices.
               bias_initializer=None):                                         # Parameter for the bias term.
    super(MyGRUCell2, self).__init__(_reuse=reuse)
    self._num_units = num_units                                                # Establishes number of neruons in the RNN.
    self._activation = activation or math_ops.tanh                             # Establishes the activation function (tanh by default).
    self._kernel_initializer = kernel_initializer                              # Initalization of the weight matrix.
    self._bias_initializer = bias_initializer                                  # Initalization of the bias term.
    self._gate_linear = None                                                   # Hold the linear transformation (weight and bias) used for calculating the gate values. Control the flow of information.
    self._candidate_linear = None                                              # Responsible for the linear transformation for computing the candidate activation values, part of the GRU's mechanism for updating the hidden state.

  @property
  def state_size(self):                                                        # Defines the size of the internal state of the GRU cell.
    return self._num_units

  @property
  def output_size(self):                                                       # Defines the size of the output produced by the GRU cell.
    return self._num_units

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    # inputs = realinputs + m +rt
    # rt's length is self._num_units
    # state = rt * older state 
    # input = first 2 part
    totalLength=inputs.get_shape().as_list()[1]                                # Determines the total number of features in the input tensor.
    inputs_=inputs[:,0:totalLength-self._num_units]                            # Separates the actual input features from the reset gate values.
    rth=inputs[:,totalLength-self._num_units:]                                 # Contains the values for the reset gate, used in state updating.
    inputs=inputs_                                                             # Updates the inputs variable to reflect only the relevant input features.
    state=math_ops.multiply(rth,state)                                         # Updates the previous state by multiplying it with the reset gate. Allows the model to forget parts of the previous state.
    with vs.variable_scope("gates"):                                           # Reset gate and update gate. Creates another variable scope specifically for calculating the candidate hidden state.
      # We start with bias of 1.0 to not reset and not update.
      bias_ones = self._bias_initializer                                       # Sets up the bias term in the GRU Cell.
      if self._bias_initializer is None:                                       # Check if bias initializer is initialized, defaults to a constant initializer with a value of 1 if not.
        dtype = [a.dtype for a in [inputs, state]][0]
        bias_ones = init_ops.constant_initializer(1.0, dtype=dtype)
      value = math_ops.sigmoid(                                                #  Determines how much of the previous state to keep and how much of the new candidate state to incorporate.
          _linear([inputs, state], 2 * self._num_units, True, bias_ones,
                  self._kernel_initializer))
      r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)         # Affects how much of the past information to discard.  Controls how much of the new candidate hidden state to incorporate.
    with vs.variable_scope("candidate"):
      c = self._activation(                                                     # Applies the activation function to the candidate hidden state.
          _linear([inputs, r * state], self._num_units, True,
                  self._bias_initializer, self._kernel_initializer))
    new_h = u * state + (1 - u) * c                                             # Computes the new hidden state of the GRU cell.
    return new_h, new_h                                                         # Returns the updated hidden state for the next time step.
