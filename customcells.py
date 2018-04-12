import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
import os

from helper_funcs import linear2, linear, kind_dict
from helper_funcs import DiagonalGaussianFromExisting

import collections
import hashlib
import numbers



class CustomGRUCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, batch_size,
                 clip_value = None,
                 state_is_tuple = False,
                 reuse = None,
                 recurrent_collections=None,
                 input_collections=None,
                 forget_bias=1.0):
        super(CustomGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._state_is_tuple = state_is_tuple
        self._batch_size = batch_size
        self._clip_value = clip_value
        self._output_size = self._num_units
        self._state_size = self._num_units
        self._rec_collections = recurrent_collections
        self._input_collections = input_collections
        self._forget_bias = forget_bias # is not used in LFADS

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

            with tf.variable_scope("Gates"):
                #ru = linear(tf.concat(axis=1, values=[inputs, state]),
                #                2 * self._num_units, name='ru_linear')
                # MRK, note, hps.input_weight_scale is replaced with 1.0 for now
                # MRK, separate input and state weights collections
                r_x, u_x = tf.split(axis=1, num_or_size_splits=2, value=linear2(inputs,
                                                                               2 * self._num_units,
                                                                               alpha=1.0, # input_weight_scale
                                                                               do_bias=False,
                                                                               name="input_ru_linear",
                                                                               normalized=False,
                                                                               collections=self._input_collections))

                r_h, u_h = tf.split(axis=1, num_or_size_splits=2, value=linear2(state,
                                                                               2 * self._num_units,
                                                                               do_bias=True,
                                                                               alpha=1.0, #self._rec_weight_scale,
                                                                               name="state_ru_linear",
                                                                               collections=self._rec_collections))
                r = r_x + r_h
                u = u_x + u_h
                r, u = tf.sigmoid(r), tf.sigmoid(u + self._forget_bias)
                #ru = tf.nn.sigmoid(ru)
                #r, u = tf.split( ru, 2, axis=1)

            with tf.variable_scope("Candidate"):
                #c = tf.nn.tanh( _linear([inputs, r * state],
                #                                          self._num_units, True))
            # new_h = u * h + (1 - u) * c
                c_x = linear2(inputs, self._num_units, name="input_c_linear", do_bias=False,
                             alpha=1.0, #self._input_weight_scale,
                             normalized=False,
                             collections=self._input_collections)
                c_rh = linear2(r * state, self._num_units, name="state_c_linear", do_bias=True,
                          alpha=1.0, #self._rec_weight_scale,
                          collections=self._rec_collections)
                c = tf.tanh(c_x + c_rh)

            new_h = u*state + (1 - u) * c

            # clip if requested
            if self._clip_value is not None:
                new_h = tf.clip_by_value(new_h, -self._clip_value, self._clip_value)
            return new_h, new_h            

class ComplexCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units_con, num_units_gen,
                 co_dim, factors_dim, fac_2_con_dim,
                 batch_size, var_min = 0.0,
                 clip_value = None,
                 state_is_tuple=False, reuse=None, kind=None,
                 forget_bias=1.0):
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
                if self._kind in [kind_dict("train"), kind_dict("posterior_sample_and_average")]:
                    co_out = cos_posterior.sample()
                else:
                    co_out = cos_posterior.mean

            # generator's inputs are the controller's outputs
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
