import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell
from helper_funcs import linear, kind_dict
from helper_funcs import DiagonalGaussianFromExisting


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
               num_units_gen,
               num_units_con,
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
      self.build_custom(self._co_dim + self._ext_input_dim,
                        cell_name='gen_gru', num_units=self._num_units_gen, rec_collections_name='l2_gen')
      con_input_depth = inputs_shape[1].value +  self._factors_dim - self._ext_input_dim
      self.build_custom(con_input_depth, cell_name='con_gru', num_units=self._num_units_con, rec_collections_name='l2_con')
      self.built = True

  def build_custom(self, input_depth, cell_name='', num_units=None, rec_collections_name=None):
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
      tf.add_to_collection(rec_collections_name, self._gate_kernel_rec[cell_name])
      tf.add_to_collection(rec_collections_name, self._candidate_kernel_rec[cell_name])

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

    # MRK, separate matmul for input and recurrent weights
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

    # split the state to get the gen and con states, and factors
    gen_s, con_s, _, _, _, fac_s = \
      tf.split(state, [self._num_units_gen,
                       self._num_units_con,
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
    with tf.name_scope("con_2_gen"):
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




from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
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
