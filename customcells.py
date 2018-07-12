import numpy as np
import tensorflow as tf
import os

from helper_funcs import linear2, linear, kind_dict
from helper_funcs import DiagonalGaussianFromExisting

import collections
import hashlib
import numbers


from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell

class GRUCell(LayerRNNCell):
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
    dtype: Default dtype of the layer (default of `None` means use the type
      of the first input). Required when `build` is called before `call`.
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None,
               dtype=None,
               recurrent_collections=None):
    super(GRUCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._rec_collections = recurrent_collections

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value
    # MRK, changed the following to allow separate variables for input and recurrent weights
    self._gate_kernel_input = self.add_variable(
        "gates/%s_input" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth, 2 * self._num_units],
        initializer=self._kernel_initializer)
    self._gate_kernel_rec = self.add_variable(
        "gates/%s_rec" % _WEIGHTS_VARIABLE_NAME,
        shape=[self._num_units, 2 * self._num_units],
        initializer=self._kernel_initializer)


    self._gate_bias = self.add_variable(
        "gates/%s" % _BIAS_VARIABLE_NAME,
        shape=[2 * self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.constant_initializer(1.0, dtype=self.dtype)))
    
    # MRK, changed the following to allow separate variables for input and recurrent weights
    self._candidate_kernel_input = self.add_variable(
        "candidate/%s_input" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth, self._num_units],
        initializer=self._kernel_initializer)
    self._candidate_kernel_rec = self.add_variable(
        "candidate/%s_rec" % _WEIGHTS_VARIABLE_NAME,
        shape=[self._num_units, self._num_units],
        initializer=self._kernel_initializer)

    self._candidate_bias = self.add_variable(
        "candidate/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.zeros_initializer(dtype=self.dtype)))

    self.built = True
    # MRK, add the recurrent weights to collections for applying L2
    if self._rec_collections:
      tf.add_to_collection(self._rec_collections, self._gate_kernel_rec)
      tf.add_to_collection(self._rec_collections, self._candidate_kernel_rec)

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    # MRK, seperate matmul for input and recurrent weights    
    gate_inputs_input = math_ops.matmul(inputs, self._gate_kernel_input)
    gate_inputs_rec = math_ops.matmul(state, self._gate_kernel_rec)
    gate_inputs = gate_inputs_input + gate_inputs_rec 

    gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

    value = math_ops.sigmoid(gate_inputs)
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state

    #candidate = math_ops.matmul(
    #    array_ops.concat([inputs, r_state], 1), self._candidate_kernel)

    # MRK, seperate matmul for input and recurrent weights    
    candidate_input = math_ops.matmul(inputs, self._candidate_kernel_input)
    candidate_rec = math_ops.matmul(state, self._candidate_kernel_rec)
    candidate = candidate_input + candidate_rec

    candidate = nn_ops.bias_add(candidate, self._candidate_bias)

    c = self._activation(candidate)
    new_h = u * state + (1 - u) * c
    return new_h, new_h


class ComplexCell(LayerRNNCell):
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
    dtype: Default dtype of the layer (default of `None` means use the type
      of the first input). Required when `build` is called before `call`.
  """

  def __init__(self,
               num_units_con,
               num_units_gen,
               factors_dim,
               co_dim,
               ext_input_dim,
               inject_ext_input_to_gen,
               run_type,
               keep_prob,
               var_min=None,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None,
               dtype=None,):
    super(ComplexCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._gate_kernel_input = {}
    self._gate_kernel_rec = {}
    self._candidate_kernel_input = {}
    self._candidate_kernel_rec = {}
    self._candidate_bias = {}
    self._gate_bias = {}
    # make our custom inputs accessible to the class
    self._inject_ext_input_to_gen = inject_ext_input_to_gen
    self._var_min = var_min
    self._run_type = run_type
    self._num_units_gen = num_units_gen
    self._num_units_con = num_units_con
    self._co_dim = co_dim
    self._factors_dim = factors_dim
    self._ext_input_dim = ext_input_dim
    self._keep_prob = keep_prob


  @property
  def state_size(self):
    return self._num_units_con + self._num_units_gen + 3*self._co_dim + self._factors_dim

  @property
  def output_size(self):
    return self._num_units_con + self._num_units_gen + 3*self._co_dim + self._factors_dim

  def build(self, inputs_shape):
      # create GRU weight/bias tensors for generator and controller
      self.build_custom([None, self._co_dim + self._ext_input_dim],
                        cell_name='gen_gru', num_units=self._num_units_gen, rec_collections_name='l2_gen')
      con_inputs_shape = [None, inputs_shape[1].value +  self._factors_dim]
      self.build_custom(con_inputs_shape, cell_name='con_gru', num_units=self._num_units_con, rec_collections_name='l2_con')
      self.built = True

  def build_custom(self, inputs_shape, cell_name='', num_units=None, rec_collections_name=None):
    if isinstance(inputs_shape, list):
        input_depth = inputs_shape[1]
    else:
        if inputs_shape[1].value is None:
          raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                           % inputs_shape)

        input_depth = inputs_shape[1].value

    # MRK, changed the following to allow separate variables for input and recurrent weights
    self._gate_kernel_input[cell_name] = self.add_variable(
        "gates/%s_%s_input" % (cell_name, _WEIGHTS_VARIABLE_NAME),
        shape=[input_depth, 2 * num_units],
        initializer=self._kernel_initializer)
    self._gate_kernel_rec[cell_name] = self.add_variable(
        "gates/%s_%s_rec" % (cell_name, _WEIGHTS_VARIABLE_NAME),
        shape=[num_units, 2 * num_units],
        initializer=self._kernel_initializer)


    self._gate_bias[cell_name] = self.add_variable(
        "gates/%s_%s" % (cell_name, _BIAS_VARIABLE_NAME),
        shape=[2 * num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.constant_initializer(1.0, dtype=self.dtype)))
    
    # MRK, changed the following to allow separate variables for input and recurrent weights
    self._candidate_kernel_input[cell_name] = self.add_variable(
        "candidate/%s_%s_input" % (cell_name, _WEIGHTS_VARIABLE_NAME),
        shape=[input_depth, num_units],
        initializer=self._kernel_initializer)
    self._candidate_kernel_rec[cell_name] = self.add_variable(
        "candidate/%s_%s_rec" % (cell_name, _WEIGHTS_VARIABLE_NAME),
        shape=[num_units, num_units],
        initializer=self._kernel_initializer)

    self._candidate_bias[cell_name] = self.add_variable(
        "candidate/%s_%s" % (cell_name, _BIAS_VARIABLE_NAME),
        shape=[num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.zeros_initializer(dtype=self.dtype)))


    # MRK, add the recurrent weights to collections for applying L2
    if rec_collections_name:
      tf.add_to_collection(rec_collections_name, self._gate_kernel_rec)
      tf.add_to_collection(rec_collections_name, self._candidate_kernel_rec)

  def gru_block(self, inputs, state, cell_name=''):
    """Gated recurrent unit (GRU) with nunits cells."""
    # MRK, seperate matmul for input and recurrent weights    
    gate_inputs_input = math_ops.matmul(inputs, self._gate_kernel_input[cell_name])
    gate_inputs_rec = math_ops.matmul(state, self._gate_kernel_rec[cell_name])
    gate_inputs = gate_inputs_input + gate_inputs_rec 

    gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias[cell_name])

    value = math_ops.sigmoid(gate_inputs)
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state

    #candidate = math_ops.matmul(
    #    array_ops.concat([inputs, r_state], 1), self._candidate_kernel)

    # MRK, seperate matmul for input and recurrent weights    
    candidate_input = math_ops.matmul(inputs, self._candidate_kernel_input[cell_name])
    candidate_rec = math_ops.matmul(state, self._candidate_kernel_rec[cell_name])
    candidate = candidate_input + candidate_rec

    candidate = nn_ops.bias_add(candidate, self._candidate_bias[cell_name])

    c = self._activation(candidate)
    new_h = u * state + (1 - u) * c
    return new_h


  def call(self, inputs, state):
    # if external inputs are used split the inputs
    if self._ext_input_dim > 0:
      ext_inputs = inputs[:, -self._ext_input_dim:]
      con_i = inputs[:, :-self._ext_input_dim]
    else:
      con_i = inputs

    # split the state to get the gen and con states, and factor
    gen_s, con_s, _, _, _, fac_s = \
      tf.split(state, [self._num_units_con,
                       self._num_units_gen,
                       self._co_dim,
                       self._co_dim,
                       self._co_dim,
                       self._factors_dim],  axis=1)

    # input to the controller is (enc_con output and factors)
    con_inputs = tf.concat([con_i, fac_s], axis=1, )

    # controller GRU recursion, get new state
    # add dropout to controller inputs (MRK fix)
    con_inputs = tf.nn.dropout(con_inputs, self._keep_prob)
    con_s_new = self.gru_block(con_inputs, con_s, cell_name='con_gru')

    # calculate the inputs to the generator
    with tf.variable_scope("con_2_gen"):
        # transformation to mean and logvar of the posterior
        co_mean = linear(con_s_new, self._co_dim,
                               name="con_2_gen_transform_mean")
        co_logvar = linear(con_s_new, self._co_dim,
                                 name="con_2_gen_transform_logvar")

        if self._var_min:
            co_logvar = tf.log(tf.exp(co_logvar) + self._var_min)

        cos_posterior = DiagonalGaussianFromExisting(
            co_mean, co_logvar)

        # whether to sample the posterior or pass its mean
        # MRK, fixed the following
        do_posterior_sample = tf.logical_or(tf.equal(self._run_type, tf.constant(kind_dict("train"))),
                                            tf.equal(self._run_type, tf.constant(kind_dict("posterior_sample_and_average"))))
        co_out = tf.cond(do_posterior_sample, lambda: cos_posterior.sample(), lambda: cos_posterior.mean)

    # generator's inputs
    if self._ext_input_dim > 0 and self._inject_ext_input_to_gen:
        # passing external inputs along with controller output as generetor's input
        gen_inputs = tf.concat([co_out, ext_inputs], axis=1)
    elif self._ext_input_dim > 0 and not self._inject_ext_input_to_gen:
        assert 0, "Not Implemented!"
    else:
        # using only controller output as generator's input
        gen_inputs = co_out

    # generator GRU recursion, get the new state
    gen_s_new = self.gru_block(gen_inputs, gen_s, cell_name='gen_gru')

    # calculate the factors
    with tf.variable_scope("gen_2_fac"):
        # add dropout to gen output (MRK fix)
        gen_s_new_dropped = tf.nn.dropout(gen_s_new, self._keep_prob)
        fac_s_new = linear(gen_s_new_dropped, self._factors_dim,
                           name="gen_2_fac_transform",
                           #collections=self.col_names['fac']
                       )
    # pass the states and make other values accessible outside DynamicRNN
    state_concat = [gen_s_new, con_s_new, co_mean, co_logvar, co_out, fac_s_new]
    new_h = tf.concat(state_concat, axis=1)

    return new_h, new_h



class ComplexCell_old(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units_con, num_units_gen,
                 co_dim, factors_dim, fac_2_con_dim, ext_input_dim,
                 batch_size, var_min = 0.0,
                 clip_value = None,
                 state_is_tuple=False, reuse=None, kind=None,
                 forget_bias=1.0, inject_ext_input_to_gen=True):
        super(ComplexCell, self).__init__(_reuse=reuse)
        
        self._num_units_con = num_units_con
        self._num_units_gen = num_units_gen

        self._co_dim = co_dim
        self._factors_dim = factors_dim
        self._fac_2_con_dim = fac_2_con_dim
        self._state_is_tuple = state_is_tuple
        self._batch_size = batch_size
        self._var_min = var_min
        self._kind = kind
        self._clip_value = clip_value
        self._forget_bias = forget_bias
        # note that there are 3x co_dim
        # because it has mean and logvar params, and samples
        self._num_units_tot = self._num_units_con + \
                              self._num_units_gen + 3*self._co_dim + self._factors_dim
        self._output_size = self._num_units_tot
        self._state_size = self._num_units_con + \
                           self._num_units_gen + 3*self._co_dim + self._factors_dim
        self._noise_bxn = tf.random_normal([batch_size, self._co_dim])

        self.col_names = {'con': ['l2_con'],
                          'gen': ['l2_gen'],
                          'fac': ['l2_gen_2_factors'],}
        self.inject_ext_input_to_gen = inject_ext_input_to_gen
        self.ext_input_dim = ext_input_dim

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size
        

    def __call__(self, inputs, state, scope=None):
        name = type(self).__name__
        # inputs and state better be a 4-tuple
        with tf.variable_scope(scope or type(self).__name__):
            # get the inputs
            if self.ext_input_dim > 0:
                ext_inputs = inputs[:,-self.ext_input_dim:]
                con_i = inputs[:,:-self.ext_input_dim]
            else:
                con_i = inputs
            # split the state
            con_s, gen_s, co_mean_s, co_logvar_s, co_prev_out, fac_s = \
                            tf.split(state, [self._num_units_con,
                                             self._num_units_gen,
                                             self._co_dim,
                                             self._co_dim,
                                             self._co_dim,
                                             self._factors_dim],
                                                 axis=1)

            # the controller's full inputs are:
            #   inputs coming in from the ci_encoder
            #   a reduced_dim rep of the previous factors
            with tf.variable_scope("fac_2_con"):
                con_fac_i = linear(fac_s, self._fac_2_con_dim,
                                   name = "fac_2_con_transform")

            con_inputs = tf.concat(axis=1, values=[con_i, con_fac_i])

            ## implement standard GRU eqns for controller
            with tf.variable_scope("Controller_Gates"):
                #con_ru = _linear([con_inputs, con_s],
                #                               2 * self._num_units_con, True, 1.0)
                #con_ru = linear(tf.concat(axis=1, values=[con_inputs, con_s]),
                #                2 * self._num_units_con, name='con_ru_linear')
                #con_ru = tf.nn.sigmoid(con_ru)
                #con_r, con_u = tf.split( con_ru, 2, axis=1)
                # MRK, add collections
                r_x, u_x = tf.split(axis=1, num_or_size_splits=2, value=linear2(con_inputs,
                                                                                2 * self._num_units_con,
                                                                                alpha=1.0,  # input_weight_scale
                                                                                do_bias=False,
                                                                                name="input_con_ru_linear",
                                                                                normalized=False,
                                                                                collections=self.col_names['con']))

                r_h, u_h = tf.split(axis=1, num_or_size_splits=2, value=linear2(con_s,
                                                                                2 * self._num_units_con,
                                                                                do_bias=True,
                                                                                alpha=1.0,  # self._rec_weight_scale,
                                                                                name="state_con_ru_linear",
                                                                                collections=self.col_names['con']))
                r = r_x + r_h
                u = u_x + u_h
                con_r, con_u = tf.sigmoid(r), tf.sigmoid(u + self._forget_bias)

            with tf.variable_scope("Controller_Candidate"):
                #con_c = tf.nn.tanh( _linear([con_inputs, con_r * con_s],
                #                                          self._num_units_con, True))
            #con_new_h = con_u*con_s + (1 - con_u) * con_c
                c_x = linear2(con_inputs, self._num_units_con, name="input_con_c_linear", do_bias=False,
                              alpha=1.0,  # self._input_weight_scale,
                              normalized=False,
                              collections=self.col_names['con'])
                c_rh = linear2(con_r * con_s, self._num_units_con, name="state_con_c_linear", do_bias=True,
                               alpha=1.0,  # self._rec_weight_scale,
                               collections=self.col_names['con'])
                con_c = tf.tanh(c_x + c_rh)

            con_new_h = con_u * con_s + (1 - con_u) * con_c

            # calculate the controller outputs
            with tf.variable_scope("con_2_gen"):
                co_mean_new_h = linear(con_new_h, self._co_dim,
                                       name = "con_2_gen_transform_mean")
                co_logvar_new_h = linear(con_new_h, self._co_dim,
                                         name = "con_2_gen_transform_logvar")
                if self._var_min > 0.0:
                    co_logvar_new_h = tf.log(tf.exp(co_logvar_new_h) + self._var_min)

                cos_posterior = DiagonalGaussianFromExisting(
                    co_mean_new_h, co_logvar_new_h)
                
                # normally sample, but sometimes use the mean
                # MRK, fixed the following
                do_posterior_sample = tf.logical_or(tf.equal(self._kind, tf.constant(kind_dict("train"))),
                            tf.equal(self._kind, tf.constant(kind_dict("posterior_sample_and_average"))))
                co_out = tf.cond(do_posterior_sample, lambda:cos_posterior.sample(), lambda:cos_posterior.mean)

            # generator's inputs

            if self.ext_input_dim > 0 and self.inject_ext_input_to_gen:
                # passing external inputs along with controller output as generetor's input
                gen_inputs = tf.concat(
                    axis=1, values=[co_out, ext_inputs])
            elif self.ext_input_dim > 0 and not self.inject_ext_input_to_gen:
                assert 0, "Not Implemented!"
            else:
                # using only controller output as generator's input
                gen_inputs = co_out
            
            ## implement standard GRU eqns for generator
            with tf.variable_scope("Generator_Gates"):
                #gen_ru = linear(tf.concat(axis=1, values=[gen_inputs, gen_s]),
                #                2 * self._num_units_gen, name='gen_run_linear')
                #gen_ru = tf.nn.sigmoid(gen_ru)
                #gen_r, gen_u = tf.split(gen_ru, 2, axis=1)
                r_x, u_x = tf.split(axis=1, num_or_size_splits=2, value=linear2(gen_inputs,
                                                                                2 * self._num_units_gen,
                                                                                alpha=1.0,  # input_weight_scale
                                                                                do_bias=False,
                                                                                name="input_gen_ru_linear",
                                                                                normalized=False,
                                                                                collections=self.col_names['gen']))

                r_h, u_h = tf.split(axis=1, num_or_size_splits=2, value=linear2(gen_s,
                                                                                2 * self._num_units_gen,
                                                                                do_bias=True,
                                                                                alpha=1.0,  # self._rec_weight_scale,
                                                                                name="state_gen_ru_linear",
                                                                                collections=self.col_names['gen']))
                r = r_x + r_h
                u = u_x + u_h
                gen_r, gen_u = tf.sigmoid(r), tf.sigmoid(u + self._forget_bias)

            with tf.variable_scope("Generator_Candidate"):
                #gen_c = tf.nn.tanh( _linear([gen_inputs, gen_r * gen_s],
                #                            self._num_units_gen, True))
            #gen_new_h = gen_u*gen_s + (1 - gen_u) * gen_c
                c_x = linear2(gen_inputs, self._num_units_gen, name="input_gen__c_linear", do_bias=False,
                              alpha=1.0,  # self._input_weight_scale,
                              normalized=False,
                              collections=self.col_names['gen'])
                c_rh = linear2(gen_r * gen_s, self._num_units_gen, name="state_gen_c_linear", do_bias=True,
                               alpha=1.0,  # self._rec_weight_scale,
                               collections=self.col_names['gen'])
                c = tf.tanh(c_x + c_rh)

            gen_new_h = gen_u * gen_s + (1 - gen_u) * c

            # calculate the factors
            with tf.variable_scope("gen_2_fac"):
                fac_new_h = linear(gen_new_h, self._factors_dim,
                                   name = "gen_2_fac_transform",
                                   collections=self.col_names['fac'])
                #fac_new_h = linear2(gen_new_h, self._factors_dim,
                #                    name = "gen_2_fac_transform",
                 #                   collections=self.col_names['fac'])


            values = [con_new_h, gen_new_h, co_mean_new_h, co_logvar_new_h, co_out, fac_new_h]
            new_h = tf.concat(axis=1, values=values)

            # add a clip to prevent crazy gradients / nan-ing out
            if self._clip_value is not None:
                new_h = tf.clip_by_value(new_h, -self._clip_value, self._clip_value)
            
        return new_h, new_h



from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

def _linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
        dtype=dtype,
        initializer=kernel_initializer)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      if bias_initializer is None:
        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
      biases = vs.get_variable(
          _BIAS_VARIABLE_NAME, [output_size],
          dtype=dtype,
          initializer=bias_initializer)
    return nn_ops.bias_add(res, biases)
